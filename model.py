"""
model.py
--------
AptamerClassifier — self-attention Transformer for binary aptamer binding
classification.

Architecture (ViT-style patch embedding):
    1. PatchEmbedding  : splits (B, embed_dim) → (B, N, d_model)
    2. [CLS] token     : learnable prefix token prepended to the patch sequence
    3. SinusoidalPE    : fixed positional encoding over (N+1) positions
    4. TransformerEncoder (Pre-LN, multi-head self-attention + GELU FFN) × L layers
    5. ClassificationHead : LayerNorm → Linear → GELU → Dropout → Linear(1)
                            applied to the [CLS] output token

Input  : (batch, embed_dim)  — mean-pooled NT embeddings from gen_embeddings.py
Output : (batch,)             — raw logit; apply sigmoid for probability

Config keys consumed (config.yaml → model section):
    embed_dim, num_patches, d_model, nhead, num_layers, dim_feedforward, dropout
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Splits a flat embedding vector into N equal patches and linearly projects
    each patch to d_model dimensions.

    Constraint: embed_dim % num_patches == 0

    Parameters
    ----------
    embed_dim   : int  — dimension of the incoming embedding (e.g. 1280)
    num_patches : int  — number of patches to split into (e.g. 16)
    d_model     : int  — output dimension per patch (transformer hidden size)
    """

    def __init__(self, embed_dim: int, num_patches: int, d_model: int) -> None:
        super().__init__()
        if embed_dim % num_patches != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_patches ({num_patches}). "
                f"Try num_patches in {[p for p in range(1, embed_dim+1) if embed_dim % p == 0]}."
            )
        self.num_patches = num_patches
        self.patch_size  = embed_dim // num_patches
        self.projection  = nn.Linear(self.patch_size, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, embed_dim)
        B = x.size(0)
        x = x.view(B, self.num_patches, self.patch_size)  # (B, N, patch_size)
        return self.projection(x)                          # (B, N, d_model)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard fixed sinusoidal positional encoding.
    Adds position information across the sequence dimension.

    Parameters
    ----------
    d_model : int   — must match the transformer hidden size
    max_len : int   — maximum sequence length (default 512 is more than enough)
    dropout : float — applied after adding the encoding
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ClassificationHead(nn.Module):
    """
    Two-layer MLP classification head applied to the [CLS] token.

        LayerNorm → Linear(d_model → d_model//2) → GELU → Dropout → Linear(→ 1)

    Returns a scalar logit per sample (no sigmoid — use BCEWithLogitsLoss).
    """

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, d_model)  → logit : (B,)
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class AptamerClassifier(nn.Module):
    """
    Self-attention Transformer for binary aptamer binding classification.

    Parameters
    ----------
    embed_dim       : int   — NT embedding dimension from gen_embeddings.py
    num_patches     : int   — number of patches to split the embedding into
    d_model         : int   — transformer hidden dimension
    nhead           : int   — number of self-attention heads (d_model % nhead == 0)
    num_layers      : int   — number of TransformerEncoderLayer stacks
    dim_feedforward : int   — FFN inner dimension inside each encoder layer
    dropout         : float — dropout probability (applied in attention, FFN, PE, head)

    Usage
    -----
        model = AptamerClassifier.from_config("config.yaml")
        logit = model(embedding_batch)          # (B,)
        prob  = torch.sigmoid(logit)            # probability of class 1
    """

    def __init__(
        self,
        embed_dim:       int   = 1280,
        num_patches:     int   = 16,
        d_model:         int   = 256,
        nhead:           int   = 8,
        num_layers:      int   = 4,
        dim_feedforward: int   = 512,
        dropout:         float = 0.1,
    ) -> None:
        super().__init__()

        # ── 1. Patch embedding ────────────────────────────────────────────────
        self.patch_embed = PatchEmbedding(embed_dim, num_patches, d_model)

        # ── 2. Learnable [CLS] token ─────────────────────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── 3. Positional encoding over (num_patches + 1) positions ──────────
        self.pos_enc = SinusoidalPositionalEncoding(
            d_model, max_len=num_patches + 1, dropout=dropout
        )

        # ── 4. Transformer encoder (Pre-LayerNorm for stable training) ────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,   # Pre-LN: more stable than Post-LN
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers           = num_layers,
            enable_nested_tensor = False,  # suppress PyTorch deprecation warning
        )

        # ── 5. Classification head on [CLS] token output ──────────────────────
        self.head = ClassificationHead(d_model, dropout)

        self._init_weights()

    # ── Weight initialisation ────────────────────────────────────────────────
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Forward pass ─────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (B, embed_dim)

        Returns
        -------
        logit : torch.Tensor  shape (B,)   — raw logit (no sigmoid applied)
        """
        B = x.size(0)

        # Step 1: patch embedding → (B, N, d_model)
        x = self.patch_embed(x)

        # Step 2: prepend [CLS] token → (B, N+1, d_model)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)

        # Step 3: positional encoding
        x = self.pos_enc(x)

        # Step 4: self-attention transformer
        x = self.transformer(x)

        # Step 5: extract [CLS] output → classification head → scalar logit
        return self.head(x[:, 0])

    # ── Convenience constructors ─────────────────────────────────────────────
    @classmethod
    def from_config(cls, config_path: str = "config.yaml") -> "AptamerClassifier":
        """Instantiate the model from a config.yaml file."""
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        m = cfg["model"]
        return cls(
            embed_dim       = m["embed_dim"],
            num_patches     = m["num_patches"],
            d_model         = m["d_model"],
            nhead           = m["nhead"],
            num_layers      = m["num_layers"],
            dim_feedforward = m["dim_feedforward"],
            dropout         = m["dropout"],
        )

    # ── Utilities ─────────────────────────────────────────────────────────────
    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        """Human-readable parameter count breakdown."""
        lines = ["AptamerClassifier parameter breakdown:"]
        total = 0
        for name, module in self.named_children():
            n = sum(p.numel() for p in module.parameters() if p.requires_grad)
            lines.append(f"  {name:<25} {n:>10,}")
            total += n
        lines.append(f"  {'TOTAL':<25} {total:>10,}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Loss function
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Binary Focal Loss (Lin et al., 2017  —  RetinaNet).

        FL(p_t) = -α_t · (1 − p_t)^γ · log(p_t)

    where
        p_t   = sigmoid(logit)      if y = 1,  else  1 − sigmoid(logit)
        α_t   = alpha               if y = 1,  else  1 − alpha

    The (1 − p_t)^γ term down-weights easy negatives so training focuses
    on hard examples rather than relying on explicit class reweighting.

    Parameters
    ----------
    alpha : float
        Balance factor between positive and negative classes.
        alpha=0.25 (Lin et al. default) gives a mild bias towards positives.
        alpha=0.5 treats both classes equally (all focusing comes from gamma).
    gamma : float
        Focusing exponent.  gamma=0 → standard BCE.  gamma=2 is standard.
        Higher values focus more aggressively on hard / misclassified examples.

    Usage
    -----
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        loss = criterion(logits, labels)   # logits: (B,), labels: (B,) ∈ {0,1}
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B,) raw model output (before sigmoid)
        targets : (B,) float tensor with values in {0.0, 1.0}
        """
        # Element-wise BCE: -[y log p + (1-y) log(1-p)]
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # p_t: probability assigned to the true class
        p_t = torch.exp(-bce)

        # α_t: per-sample alpha weight aligned with the true class
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        # Focal weight: suppress easy examples
        focal_weight = (1.0 - p_t).pow(self.gamma)

        return (alpha_t * focal_weight * bce).mean()

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, gamma={self.gamma}"


def build_criterion(t_cfg: dict, device: torch.device) -> nn.Module:
    """
    Construct the loss function from the training config section.

    Config keys:
        loss        : "focal" (default) | "bce"
        focal_alpha : float  (used when loss=focal, default 0.25)
        focal_gamma : float  (used when loss=focal, default 2.0)
    """
    name = t_cfg.get("loss", "focal").lower()

    if name == "focal":
        alpha = float(t_cfg.get("focal_alpha", 0.25))
        gamma = float(t_cfg.get("focal_gamma", 2.0))
        log_fn = __import__("logging").getLogger(__name__)
        log_fn.info(f"Loss: FocalLoss(alpha={alpha}, gamma={gamma})")
        return FocalLoss(alpha=alpha, gamma=gamma).to(device)

    if name == "bce":
        __import__("logging").getLogger(__name__).info("Loss: BCEWithLogitsLoss")
        return nn.BCEWithLogitsLoss().to(device)

    raise ValueError(f"Unknown loss '{name}'. Choose: focal | bce")
