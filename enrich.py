"""
enrich.py
---------
Reinforcement-learning aptamer sequence optimizer.

Starts from a parent DNA aptamer sequence (random or user-provided) and
iteratively mutates it using PPO to maximise a SELEX-like reward:

    reward = α · binder_confidence + β · plausibility

where:
    binder_confidence — sigmoid(AptamerClassifier(NT_mean_pool(seq)))
                        the probability that the mutant is a binder, computed
                        with the classifier checkpoint trained by `train.py`.
    plausibility      — DNA sanity score (valid bases, length, GC content,
                        no homopolymer runs, decent Shannon entropy).

Pretrained Nucleotide-Transformer (NT) embeddings are extracted on-the-fly
for every mutant sample, so the reward signal directly reflects the
predicted SELEX outcome.

Run output (saved to runs/enrich_<timestamp>/):
    config.yaml         copy of the config used
    best_checkpoint.pt  highest-reward checkpoint encountered
    last_checkpoint.pt  most recent checkpoint
    history.jsonl       per-iteration metrics
    top_sequences.csv   top-K mutants by reward
    enrich.log          full rich console transcript
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
    TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)
from rich.table import Table
from rich import box

# ── Console + logging (set up first so all logs flow through rich) ───────────
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%H:%M:%S]",
    handlers=[RichHandler(console=console, show_path=False,
                          markup=True, rich_tracebacks=True)],
)
log = logging.getLogger(__name__)

from model import AptamerClassifier   # noqa: E402  — needs rich logging set first


# ═════════════════════════════════════════════════════════════════════════════
# DNA constants
# ═════════════════════════════════════════════════════════════════════════════

NT_VOCAB   = list("ATCG")
NT_TO_IDX  = {nt: i for i, nt in enumerate(NT_VOCAB)}
IDX_TO_NT  = {i: nt for i, nt in enumerate(NT_VOCAB)}
NUM_NT     = 4


def seq_to_indices(seq: str) -> torch.Tensor:
    """ATCG string → LongTensor of indices."""
    return torch.tensor([NT_TO_IDX[c] for c in seq], dtype=torch.long)


def indices_to_seq(indices: torch.Tensor) -> str:
    """LongTensor of indices → ATCG string."""
    return "".join(IDX_TO_NT[int(i)] for i in indices)


def random_aptamer_sequence(length: int = 35, rng: random.Random | None = None) -> str:
    """Generate a uniformly random DNA sequence of the given length."""
    rng = rng or random
    return "".join(rng.choice(NT_VOCAB) for _ in range(length))


# ═════════════════════════════════════════════════════════════════════════════
# Plausibility scorer  (pure-Python; no gradients)
# ═════════════════════════════════════════════════════════════════════════════

class AptamerPlausibility:
    """
    DNA-specific plausibility score in [0, 1].

    Components (each ∈ [0, 1], averaged with equal weight by default):
        valid_bases  : 1.0 if all chars in {A,T,C,G}; else proportion of valid
        length       : 1.0 if min_length ≤ len ≤ max_length; else linear decay
        gc_content   : 1.0 if gc_low ≤ gc ≤ gc_high; else linear decay
        max_run      : 1.0 if no homopolymer run > max_run; else penalty
        composition  : 1.0 if no single base > max_composition; else linear decay
        entropy      : 1.0 if Shannon entropy ≥ entropy_min, else linear (max 2.0)
    """

    def __init__(
        self,
        min_length:       int   = 30,
        max_length:       int   = 40,
        gc_low:           float = 0.20,
        gc_high:          float = 0.80,
        max_run:          int   = 5,
        max_composition:  float = 0.50,
        entropy_min:      float = 1.4,
        weights:          dict[str, float] | None = None,
    ) -> None:
        self.min_length      = min_length
        self.max_length      = max_length
        self.gc_low          = gc_low
        self.gc_high         = gc_high
        self.max_run         = max_run
        self.max_composition = max_composition
        self.entropy_min     = entropy_min
        self.entropy_max     = math.log2(NUM_NT)   # 2.0 for DNA
        default = {
            "valid_bases": 1.0,
            "length":      1.0,
            "gc_content":  1.0,
            "max_run":     1.0,
            "composition": 1.0,
            "entropy":     1.0,
        }
        w = weights or default
        total = sum(w.values())
        self.weights = {k: v / total for k, v in w.items()}

    # ── component scorers ────────────────────────────────────────────────────
    def _score_valid_bases(self, seq: str) -> float:
        if not seq:
            return 0.0
        good = sum(1 for c in seq if c in NT_TO_IDX)
        return good / len(seq)

    def _score_length(self, seq: str) -> float:
        L = len(seq)
        if self.min_length <= L <= self.max_length:
            return 1.0
        # Linear decay outside the window
        if L < self.min_length:
            return max(0.0, 1.0 - (self.min_length - L) / max(self.min_length, 1))
        return max(0.0, 1.0 - (L - self.max_length) / max(self.max_length, 1))

    def _score_gc(self, seq: str) -> float:
        if not seq:
            return 0.0
        gc = (seq.count("G") + seq.count("C")) / len(seq)
        if self.gc_low <= gc <= self.gc_high:
            return 1.0
        dist = max(self.gc_low - gc, gc - self.gc_high)
        span = max(self.gc_high - self.gc_low, 1e-6)
        return max(0.0, 1.0 - dist / span)

    def _score_max_run(self, seq: str) -> float:
        if not seq:
            return 0.0
        runs = [m.group() for m in re.finditer(r"(.)\1+", seq)]
        excess = sum(max(0, len(r) - self.max_run) for r in runs)
        # Each excess base costs 0.1; clamp to [0, 1]
        return max(0.0, 1.0 - excess * 0.10)

    def _score_composition(self, seq: str) -> float:
        if not seq:
            return 0.0
        counts = Counter(seq)
        top_frac = counts.most_common(1)[0][1] / len(seq)
        if top_frac <= self.max_composition:
            return 1.0
        # Linear decay from max_composition → 1.0 (single-base sequence)
        return max(0.0, 1.0 - (top_frac - self.max_composition) /
                              max(1.0 - self.max_composition, 1e-6))

    def _score_entropy(self, seq: str) -> float:
        if not seq:
            return 0.0
        counts = Counter(seq)
        freqs  = [c / len(seq) for c in counts.values()]
        ent    = -sum(f * math.log2(f) for f in freqs if f > 0)
        if ent >= self.entropy_max:
            return 1.0
        if ent <= self.entropy_min:
            return max(0.0, ent / max(self.entropy_min, 1e-6) * 0.5)  # heavily penalised
        return (ent - self.entropy_min) / max(self.entropy_max - self.entropy_min, 1e-6)

    # ── public API ────────────────────────────────────────────────────────────
    def score(self, seq: str) -> dict[str, float]:
        comps = {
            "valid_bases": self._score_valid_bases(seq),
            "length":      self._score_length(seq),
            "gc_content":  self._score_gc(seq),
            "max_run":     self._score_max_run(seq),
            "composition": self._score_composition(seq),
            "entropy":     self._score_entropy(seq),
        }
        comps["weighted"] = sum(self.weights[k] * comps[k] for k in self.weights)
        return comps

    def batch_score(self, seqs: list[str]) -> torch.Tensor:
        return torch.tensor([self.score(s)["weighted"] for s in seqs],
                            dtype=torch.float32)


# ═════════════════════════════════════════════════════════════════════════════
# NT Embedder + Aptamer Scorer  (reward signal computation)
# ═════════════════════════════════════════════════════════════════════════════

class NTEmbedder(nn.Module):
    """
    Frozen Nucleotide Transformer wrapper.
    Returns mean-pooled embeddings — the same representation the classifier
    was trained on. Mirrors gen_embeddings.py exactly.
    """

    def __init__(
        self,
        model_id:   str = "InstaDeepAI/nucleotide-transformer-500m-1000g",
        layer:      int = -1,
        max_length: int = 128,
        device:     torch.device | None = None,
    ) -> None:
        super().__init__()
        from transformers import AutoTokenizer, AutoModelForMaskedLM

        self.device     = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.layer      = layer
        self.max_length = max_length

        log.info(f"Loading NT model: [bold]{model_id}[/bold]")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_id,
            output_hidden_states=True,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.embed_dim = self.model.config.hidden_size
        log.info(
            f"NT loaded — [dim]parameters: {sum(p.numel() for p in self.model.parameters()):,}  |  "
            f"hidden_size: {self.embed_dim}[/dim]"
        )

    @torch.no_grad()
    def mean_pool(self, sequences: list[str], batch_size: int = 32) -> torch.Tensor:
        """
        Returns mean-pooled embeddings of shape (B, embed_dim) on self.device.
        """
        all_chunks = []
        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start : start + batch_size]
            tokens = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
            input_ids       = tokens["input_ids"].to(self.device)
            attention_mask  = tokens["attention_mask"].to(self.device)

            out      = self.model(input_ids, attention_mask=attention_mask,
                                  output_hidden_states=True)
            hidden   = out.hidden_states[self.layer]               # (B, L, D)
            mask_exp = attention_mask.unsqueeze(-1).float()
            summed   = (hidden * mask_exp).sum(dim=1)              # (B, D)
            denom    = attention_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
            pooled   = (summed / denom).float()                    # cast back to fp32
            all_chunks.append(pooled)

        return torch.cat(all_chunks, dim=0)


class AptamerScorer(nn.Module):
    """
    Loads an AptamerClassifier checkpoint and produces binder confidence
    scores ∈ [0, 1] for batches of DNA sequences — the same pipeline as
    inference.py, just exposed as a callable.
    """

    def __init__(
        self,
        ckpt_path:  str | Path,
        nt:         NTEmbedder,
        device:     torch.device,
    ) -> None:
        super().__init__()
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"AptamerClassifier checkpoint not found: {ckpt_path}\n"
                f"Run train.py first to produce one."
            )

        ckpt = torch.load(ckpt_path, map_location=device, weights_only = False)
        ckpt_cfg = ckpt.get("config")
        if ckpt_cfg is None or "model" not in ckpt_cfg:
            raise ValueError(
                f"Checkpoint at {ckpt_path} is missing the embedded 'config.model' section."
            )
        m = ckpt_cfg["model"]

        self.classifier = AptamerClassifier(
            embed_dim       = m["embed_dim"],
            num_patches     = m["num_patches"],
            d_model         = m["d_model"],
            nhead           = m["nhead"],
            num_layers      = m["num_layers"],
            dim_feedforward = m["dim_feedforward"],
            dropout         = m["dropout"],
        ).to(device)
        self.classifier.load_state_dict(ckpt["state_dict"])
        self.classifier.eval()
        for p in self.classifier.parameters():
            p.requires_grad = False

        self.nt          = nt
        self.device      = device
        self.embed_dim   = nt.embed_dim
        self.ckpt_path   = ckpt_path
        self.label_map   = ckpt.get("label_map", {})
        self.val_metrics = ckpt.get("val_metrics", {})
        self.ckpt_epoch  = ckpt.get("epoch", "?")

        log.info(
            f"Classifier loaded — [bold]{ckpt_path}[/bold]  "
            f"[dim](epoch {self.ckpt_epoch}, "
            f"val_auc={self.val_metrics.get('auc', float('nan')):.4f})[/dim]"
        )

    @torch.no_grad()
    def confidence(
        self,
        sequences: list[str],
        embed_batch: int = 32,
    ) -> torch.Tensor:
        """
        Returns sigmoid(classifier_logit) ∈ [0, 1] for each sequence.
        """
        emb    = self.nt.mean_pool(sequences, batch_size=embed_batch)  # (B, D)
        logit  = self.classifier(emb)                                  # (B,)
        return torch.sigmoid(logit)


# ═════════════════════════════════════════════════════════════════════════════
# Aptamer Reward = α · confidence + β · plausibility
# ═════════════════════════════════════════════════════════════════════════════

class AptamerReward:
    """
    Combined SELEX-style reward function.

    reward_raw   = α · binder_confidence + β · plausibility   ∈ [0, 1]
    reward_delta = delta_scale · (reward_raw − parent_raw)    (if use_delta=True)
    """

    def __init__(
        self,
        scorer:                AptamerScorer,
        plausibility:          AptamerPlausibility,
        confidence_weight:     float = 0.7,
        plausibility_weight:   float = 0.3,
        use_delta:             bool  = True,
        delta_scale:           float = 5.0,
        embed_batch:           int   = 32,
    ) -> None:
        self.scorer        = scorer
        self.plausibility  = plausibility
        total              = confidence_weight + plausibility_weight
        self.w_conf        = confidence_weight   / total
        self.w_plaus       = plausibility_weight / total
        self.use_delta     = use_delta
        self.delta_scale   = delta_scale
        self.embed_batch   = embed_batch

        # Parent baseline (set by set_parent_baseline)
        self.parent_seq:     str | None     = None
        self.parent_conf:    float | None   = None
        self.parent_plaus:   float | None   = None
        self.parent_raw:     float | None   = None

    # ─────────────────────────────────────────────────────────────────────────
    def set_parent_baseline(self, parent_seq: str) -> dict[str, float]:
        """Compute and cache parent's confidence and plausibility."""
        conf  = float(self.scorer.confidence([parent_seq], embed_batch=self.embed_batch)[0].item())
        plaus_scores = self.plausibility.score(parent_seq)
        plaus = float(plaus_scores["weighted"])
        raw   = self.w_conf * conf + self.w_plaus * plaus

        self.parent_seq   = parent_seq
        self.parent_conf  = conf
        self.parent_plaus = plaus
        self.parent_raw   = raw

        return {
            "confidence":  conf,
            "plausibility": plaus,
            "raw":          raw,
            **{f"plaus_{k}": v for k, v in plaus_scores.items() if k != "weighted"},
        }

    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def batch_score(self, sequences: list[str]) -> dict[str, torch.Tensor]:
        """
        Compute confidence, plausibility, raw and (final) reward for each sequence.
        All returned tensors live on the scorer's device.
        """
        device       = self.scorer.device
        conf         = self.scorer.confidence(sequences, embed_batch=self.embed_batch)  # (B,)
        plaus        = self.plausibility.batch_score(sequences).to(device)              # (B,)
        raw          = self.w_conf * conf + self.w_plaus * plaus                        # (B,)

        if self.use_delta and self.parent_raw is not None:
            reward = (raw - self.parent_raw) * self.delta_scale
        else:
            # Centre raw [0,1] to [-1,1] when not using delta-reward
            reward = raw * 2.0 - 1.0

        return {
            "confidence":   conf,
            "plausibility": plaus,
            "raw":          raw,
            "reward":       reward,
        }


# ═════════════════════════════════════════════════════════════════════════════
# Self-attention building blocks (proposer internals)
# ═════════════════════════════════════════════════════════════════════════════

class SelfAttentionBlock(nn.Module):
    """Single transformer encoder block with relative positional bias."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1, max_rel_pos: int = 64) -> None:
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.scale    = self.head_dim ** -0.5

        self.norm1     = nn.LayerNorm(d_model)
        self.qkv       = nn.Linear(d_model, 3 * d_model)
        self.out_proj  = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

        self.max_rel_pos  = max_rel_pos
        self.rel_pos_bias = nn.Embedding(2 * max_rel_pos + 1, n_heads)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, L, D  = x.shape
        residual = x
        x = self.norm1(x)

        qkv     = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = [t.transpose(1, 2) for t in qkv.unbind(2)]
        attn    = (q @ k.transpose(-2, -1)) * self.scale

        pos  = torch.arange(L, device=x.device)
        rel  = (pos.unsqueeze(0) - pos.unsqueeze(1)).clamp(
                  -self.max_rel_pos, self.max_rel_pos) + self.max_rel_pos
        bias = self.rel_pos_bias(rel).permute(2, 0, 1).unsqueeze(0)
        attn = attn + bias

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)

        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out  = (attn @ v).transpose(1, 2).reshape(B, L, D)
        x    = residual + self.out_proj(out)
        x    = x + self.ffn(self.norm2(x))
        return x


class ContextualEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int,
                 d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            SelfAttentionBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            h = layer(h, mask)
        return self.final_norm(h)


# ═════════════════════════════════════════════════════════════════════════════
# Position selector head
# ═════════════════════════════════════════════════════════════════════════════

class PositionSelector(nn.Module):
    """
    Selects which positions to mutate. Operates autoregressively up to
    max_mutations steps, with a learned stop signal after min_mutations.
    """

    def __init__(self, d_model: int, max_mutations: int = 5, min_mutations: int = 1):
        super().__init__()
        self.max_mutations = max_mutations
        self.min_mutations = min_mutations

        self.score_head     = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.selected_token = nn.Parameter(torch.randn(d_model) * 0.02)
        self.stop_head      = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )
        self.update_attn    = SelfAttentionBlock(
            d_model, n_heads=4, d_ff=d_model * 2, dropout=0.1,
        )

    def forward(
        self,
        h:             torch.Tensor,
        mutable_mask:  torch.Tensor | None = None,
        temperature:   float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, D = h.shape
        device  = h.device

        selected   = torch.zeros(B, L, device=device)
        total_lp   = torch.zeros(B, device=device)
        total_ent  = torch.zeros(B, device=device)
        h_cur      = h.clone()

        available = torch.ones(B, L, device=device)
        if mutable_mask is not None:
            available = available * mutable_mask

        active = torch.ones(B, device=device, dtype=torch.bool)

        for step in range(self.max_mutations):
            logits = self.score_head(h_cur).squeeze(-1)
            logits = logits.masked_fill(available == 0, -1e9)
            p_pos  = F.softmax(logits / temperature, dim=-1)

            step_ent  = -(p_pos * torch.log(p_pos + 1e-10)).sum(-1)
            total_ent = total_ent + step_ent * active.float()

            if step >= self.min_mutations:
                g          = h_cur.mean(dim=1)
                stop_logit = self.stop_head(g).squeeze(-1)
                stop_p     = torch.sigmoid(stop_logit)
                stop       = (torch.bernoulli(stop_p).bool()
                              if self.training else (stop_p > 0.5))

                total_lp = total_lp + (
                    stop.float()  * torch.log(stop_p + 1e-10)
                  + (~stop).float() * torch.log(1 - stop_p + 1e-10)
                ) * active.float()

                active = active & ~stop
                if not active.any():
                    break
            else:
                stop = torch.zeros(B, device=device, dtype=torch.bool)

            idx = (torch.multinomial(p_pos, 1).squeeze(-1)
                   if self.training else p_pos.argmax(-1))

            lp       = torch.log(p_pos.gather(1, idx.unsqueeze(1)).squeeze(1) + 1e-10)
            total_lp = total_lp + lp * active.float()

            bidx = torch.arange(B, device=device)
            selected[bidx, idx] += active.float()
            available[bidx, idx] = 0

            # Mark the position as "selected" via a non-in-place update
            pos_onehot   = F.one_hot(idx, L).float()
            pos_onehot   = pos_onehot * active.float().unsqueeze(-1)
            token_delta  = pos_onehot.unsqueeze(-1) * self.selected_token.view(1, 1, -1)
            h_cur        = h_cur + token_delta
            h_cur        = self.update_attn(h_cur)

        selected = selected.clamp(0, 1)
        return selected, total_lp, total_ent


# ═════════════════════════════════════════════════════════════════════════════
# Nucleotide selector head (DNA equivalent of original AminoAcidSelector)
# ═════════════════════════════════════════════════════════════════════════════

class NucleotideSelector(nn.Module):
    """For each selected position, choose a new nucleotide (A/T/C/G)."""

    def __init__(self, d_model: int, n_heads: int = 4, neighborhood_k: int = 8):
        super().__init__()
        self.d_model = d_model
        self.k       = neighborhood_k

        self.cross_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=0.1,
        )

        self.nt_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, NUM_NT),
        )

        self.nt_embed = nn.Embedding(NUM_NT, d_model)

        self.update = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        h:            torch.Tensor,
        selected:     torch.Tensor,
        original_seq: torch.Tensor,
        temperature:  float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, D  = h.shape
        device   = h.device

        new_seq   = original_seq.clone()
        total_lp  = torch.zeros(B, device=device)
        total_ent = torch.zeros(B, device=device)
        h_cur     = h.clone()

        max_sel = int(selected.sum(-1).max().item()) if selected.sum() > 0 else 0
        if max_sel == 0:
            return new_seq, total_lp, total_ent

        for step in range(max_sel):
            active  = torch.zeros(B, device=device)
            pos_idx = torch.zeros(B, dtype=torch.long, device=device)

            for b in range(B):
                sel_positions = selected[b].nonzero(as_tuple=True)[0]
                if step < len(sel_positions):
                    pos_idx[b] = sel_positions[step]
                    active[b]  = 1.0

            if active.sum() == 0:
                break

            # Local neighbourhood context via cross-attention
            contexts = []
            for b in range(B):
                pos = pos_idx[b].item()
                k     = min(self.k, L)
                half  = k // 2
                start = max(0, pos - half)
                end   = min(L, start + k)
                start = max(0, end - k)
                nb_idx    = torch.arange(start, end, device=device)
                nb_embeds = h_cur[b, nb_idx].unsqueeze(0)
                query     = self.cross_norm(h_cur[b, pos].unsqueeze(0).unsqueeze(0))
                ctx, _    = self.cross_attn(query, nb_embeds, nb_embeds)
                contexts.append(ctx.squeeze(0).squeeze(0))
            contexts = torch.stack(contexts)

            bidx       = torch.arange(B, device=device)
            pos_embeds = h_cur[bidx, pos_idx]
            combined   = torch.cat([pos_embeds, contexts], dim=-1)
            logits     = self.nt_head(combined) / temperature
            probs      = F.softmax(logits, dim=-1)

            step_ent  = -(probs * torch.log(probs + 1e-10)).sum(-1)
            total_ent = total_ent + step_ent * active

            chosen = (torch.multinomial(probs, 1).squeeze(-1)
                      if self.training else probs.argmax(-1))

            lp       = torch.log(probs.gather(1, chosen.unsqueeze(1)).squeeze(1) + 1e-10)
            total_lp = total_lp + lp * active

            # Mutate sequence (LongTensor — safe, not in graph)
            for b in range(B):
                if active[b] > 0:
                    new_seq[b, pos_idx[b]] = chosen[b]

            # Non-in-place update of internal representation
            new_nt_emb        = self.nt_embed(chosen)
            pos_for_update    = h_cur[bidx, pos_idx]
            updated_embeds    = self.update(
                torch.cat([pos_for_update, new_nt_emb], dim=-1)
            )

            pos_onehot = F.one_hot(pos_idx, L).float()
            pos_onehot = pos_onehot * active.unsqueeze(-1)
            mask_3d    = pos_onehot.unsqueeze(-1)
            h_cur      = h_cur * (1.0 - mask_3d) + \
                         updated_embeds.unsqueeze(1) * mask_3d

        return new_seq, total_lp, total_ent


# ═════════════════════════════════════════════════════════════════════════════
# Mutation Proposer (policy network)
# ═════════════════════════════════════════════════════════════════════════════

class MutationProposal(NamedTuple):
    mutant_indices:     torch.Tensor
    mutant_seqs:        list[str]
    selected_positions: torch.Tensor
    position_log_prob:  torch.Tensor
    nt_log_prob:        torch.Tensor
    total_log_prob:     torch.Tensor
    entropy:            torch.Tensor


class MutationProposer(nn.Module):
    """
    Self-attention transformer that proposes mutations on a DNA sequence.
    Self-contained — uses its own learned base embedding rather than an
    external pretrained encoder, because Nucleotide Transformer tokenises
    at 6-mer granularity, which doesn't map cleanly to per-base mutations.
    """

    def __init__(
        self,
        seq_length:      int,
        d_model:         int = 256,
        n_heads:         int = 8,
        n_enc_layers:    int = 4,
        d_ff:            int = 512,
        max_mutations:   int = 5,
        min_mutations:   int = 1,
        neighborhood_k:  int = 8,
        dropout:         float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_length = seq_length

        self.base_embed = nn.Embedding(NUM_NT, d_model)
        self.pos_embed  = nn.Parameter(torch.zeros(1, seq_length, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.embed_norm = nn.LayerNorm(d_model)
        self.embed_drop = nn.Dropout(dropout)

        self.encoder = ContextualEncoder(d_model, n_heads, n_enc_layers, d_ff, dropout)

        self.pos_selector = PositionSelector(
            d_model, max_mutations=max_mutations, min_mutations=min_mutations,
        )
        self.nt_selector = NucleotideSelector(
            d_model, n_heads=max(1, n_heads // 2), neighborhood_k=neighborhood_k,
        )

    def encode(self, seq_indices: torch.Tensor) -> torch.Tensor:
        """seq_indices (B, L) → h (B, L, d_model)"""
        h = self.base_embed(seq_indices) + self.pos_embed[:, : seq_indices.size(1)]
        h = self.embed_drop(self.embed_norm(h))
        return self.encoder(h)

    def forward(
        self,
        seq_indices:  torch.Tensor,
        mutable_mask: torch.Tensor | None = None,
        pos_temp:     float = 1.0,
        nt_temp:      float = 1.0,
    ) -> MutationProposal:
        h = self.encode(seq_indices)
        sel, pos_lp, pos_ent = self.pos_selector(h, mutable_mask, pos_temp)
        new_idx, nt_lp, nt_ent = self.nt_selector(h, sel, seq_indices, nt_temp)

        mutant_seqs = [indices_to_seq(new_idx[b]) for b in range(new_idx.shape[0])]

        return MutationProposal(
            mutant_indices     = new_idx,
            mutant_seqs        = mutant_seqs,
            selected_positions = sel,
            position_log_prob  = pos_lp,
            nt_log_prob        = nt_lp,
            total_log_prob     = pos_lp + nt_lp,
            entropy            = pos_ent + nt_ent,
        )


# ═════════════════════════════════════════════════════════════════════════════
# Value network (critic)
# ═════════════════════════════════════════════════════════════════════════════

class ValueNetwork(nn.Module):
    def __init__(self, d_model: int, d_hidden: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden), nn.GELU(), nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_hidden // 2), nn.GELU(),
            nn.Linear(d_hidden // 2, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h.mean(dim=1)).squeeze(-1)


# ═════════════════════════════════════════════════════════════════════════════
# Rollout buffer
# ═════════════════════════════════════════════════════════════════════════════

class RolloutEntry(NamedTuple):
    sequences:           list[str]
    seq_indices:         torch.Tensor
    mutable_mask:        torch.Tensor
    selected_positions:  torch.Tensor
    total_log_prob:      torch.Tensor
    entropy:             torch.Tensor
    value:               torch.Tensor
    reward:              torch.Tensor
    advantage:           torch.Tensor
    returns:             torch.Tensor


class RolloutBuffer:
    def __init__(self) -> None:
        self.entries: list[RolloutEntry] = []

    def clear(self) -> None:
        self.entries = []

    def add(self, entry: RolloutEntry) -> None:
        self.entries.append(entry)

    def __len__(self) -> int:
        return len(self.entries)

    def get_flat(self) -> RolloutEntry:
        return RolloutEntry(
            sequences          = [s for e in self.entries for s in e.sequences],
            seq_indices        = torch.cat([e.seq_indices         for e in self.entries]),
            mutable_mask       = torch.cat([e.mutable_mask        for e in self.entries]),
            selected_positions = torch.cat([e.selected_positions  for e in self.entries]),
            total_log_prob     = torch.cat([e.total_log_prob      for e in self.entries]),
            entropy            = torch.cat([e.entropy             for e in self.entries]),
            value              = torch.cat([e.value               for e in self.entries]),
            reward             = torch.cat([e.reward              for e in self.entries]),
            advantage          = torch.cat([e.advantage           for e in self.entries]),
            returns            = torch.cat([e.returns             for e in self.entries]),
        )


# ═════════════════════════════════════════════════════════════════════════════
# PPO Trainer
# ═════════════════════════════════════════════════════════════════════════════

class PPOTrainer:
    def __init__(
        self,
        proposer:          MutationProposer,
        critic:            ValueNetwork,
        reward_fn:         AptamerReward,
        parent_sequence:   str,
        device:            torch.device,
        mutable_positions: list[int] | None = None,
        # PPO hyperparameters
        lr_policy:         float = 3e-4,
        lr_critic:         float = 1e-3,
        clip_eps:          float = 0.2,
        entropy_coef:      float = 0.05,
        value_coef:        float = 0.5,
        max_grad_norm:     float = 1.0,
        ppo_epochs:        int   = 4,
        minibatch_size:    int   = 16,
        rollout_batch_size: int  = 16,
        n_rollout_steps:   int   = 4,
        pos_temp:          float = 1.5,
        nt_temp:           float = 1.5,
        entropy_decay:     float = 0.9995,
        entropy_min:       float = 0.005,
    ) -> None:
        self.proposer        = proposer
        self.critic          = critic
        self.reward_fn       = reward_fn
        self.parent_sequence = parent_sequence
        self.device          = device

        L = len(parent_sequence)
        self.mutable_mask = torch.ones(L, device=device)
        if mutable_positions is not None:
            self.mutable_mask = torch.zeros(L, device=device)
            for p in mutable_positions:
                self.mutable_mask[p] = 1.0

        self.opt_policy = torch.optim.AdamW(
            proposer.parameters(), lr=lr_policy, weight_decay=1e-4,
        )
        self.opt_critic = torch.optim.Adam(critic.parameters(), lr=lr_critic)

        self.clip_eps          = clip_eps
        self.entropy_coef      = entropy_coef
        self.value_coef        = value_coef
        self.max_grad_norm     = max_grad_norm
        self.ppo_epochs        = ppo_epochs
        self.minibatch_size    = minibatch_size
        self.rollout_batch_size = rollout_batch_size
        self.n_rollout_steps   = n_rollout_steps
        self.pos_temp          = pos_temp
        self.nt_temp           = nt_temp
        self.entropy_decay     = entropy_decay
        self.entropy_min       = entropy_min

        self.buffer       = RolloutBuffer()
        self.global_step  = 0
        self.best_reward  = -float("inf")
        self.best_sequence       = parent_sequence
        self.best_confidence     = (reward_fn.parent_conf  if reward_fn.parent_conf  is not None else 0.0)
        self.best_plausibility   = (reward_fn.parent_plaus if reward_fn.parent_plaus is not None else 0.0)
        self.top_k_records: list[dict] = []

    # ─── rollout ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def collect_rollouts(self) -> dict:
        self.proposer.eval()
        self.critic.eval()
        self.buffer.clear()

        all_rewards: list[float] = []
        all_conf:    list[float] = []
        all_plaus:   list[float] = []

        iter_best_reward     = -float("inf")
        iter_best_seq        = self.parent_sequence
        iter_best_mutations  : list[tuple[int, str, str]] = []
        iter_best_confidence = 0.0
        iter_best_plaus      = 0.0

        for _ in range(self.n_rollout_steps):
            B = self.rollout_batch_size
            seq_indices  = (
                seq_to_indices(self.parent_sequence)
                .unsqueeze(0).expand(B, -1).to(self.device)
            )
            mutable_mask = self.mutable_mask.unsqueeze(0).expand(B, -1)

            self.proposer.train()
            proposal = self.proposer(
                seq_indices, mutable_mask,
                pos_temp=self.pos_temp, nt_temp=self.nt_temp,
            )
            self.proposer.eval()

            # ── Reward computation: NT + classifier on mutants ──────────────
            batch_scores = self.reward_fn.batch_score(proposal.mutant_seqs)
            rewards      = batch_scores["reward"]
            confidence   = batch_scores["confidence"]
            plausibility = batch_scores["plausibility"]

            h = self.proposer.encode(seq_indices)
            values = self.critic(h)

            advantages = rewards - values
            returns    = rewards
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            self.buffer.add(RolloutEntry(
                sequences          = proposal.mutant_seqs,
                seq_indices        = seq_indices,
                mutable_mask       = mutable_mask,
                selected_positions = proposal.selected_positions,
                total_log_prob     = proposal.total_log_prob,
                entropy            = proposal.entropy,
                value              = values,
                reward             = rewards,
                advantage          = advantages,
                returns            = returns,
            ))

            all_rewards.extend(rewards.tolist())
            all_conf.extend(confidence.tolist())
            all_plaus.extend(plausibility.tolist())

            # ── Best-tracking ────────────────────────────────────────────────
            top_idx = int(rewards.argmax().item())
            top_r   = float(rewards[top_idx].item())
            top_seq = proposal.mutant_seqs[top_idx]
            top_c   = float(confidence[top_idx].item())
            top_p   = float(plausibility[top_idx].item())

            if top_r > self.best_reward:
                self.best_reward       = top_r
                self.best_sequence     = top_seq
                self.best_confidence   = top_c
                self.best_plausibility = top_p

            if top_r > iter_best_reward:
                iter_best_reward     = top_r
                iter_best_seq        = top_seq
                iter_best_confidence = top_c
                iter_best_plaus      = top_p
                iter_best_mutations  = [
                    (i, p_nt, m_nt)
                    for i, (p_nt, m_nt) in enumerate(zip(self.parent_sequence, top_seq))
                    if p_nt != m_nt
                ]

            # Track top-K records for the final CSV
            for r, c, p, s in zip(rewards.tolist(), confidence.tolist(),
                                  plausibility.tolist(), proposal.mutant_seqs):
                self.top_k_records.append({
                    "sequence": s, "reward": r,
                    "confidence": c, "plausibility": p,
                })

        return {
            "reward_mean":         sum(all_rewards) / len(all_rewards),
            "reward_max":          max(all_rewards),
            "reward_min":          min(all_rewards),
            "confidence_mean":     sum(all_conf) / len(all_conf),
            "plausibility_mean":   sum(all_plaus) / len(all_plaus),
            "best_ever_reward":    self.best_reward,
            "iter_best_reward":    iter_best_reward,
            "iter_best_seq":       iter_best_seq,
            "iter_best_confidence": iter_best_confidence,
            "iter_best_plausibility": iter_best_plaus,
            "iter_best_mutations": iter_best_mutations,
        }

    # ─── PPO update ─────────────────────────────────────────────────────────
    def update(self) -> dict[str, float]:
        self.proposer.train()
        self.critic.train()

        data = self.buffer.get_flat()
        N    = len(data.reward)
        old_log_probs = data.total_log_prob.detach()
        advantages    = data.advantage.detach()
        returns       = data.returns.detach()

        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0
        total_clip_frac   = 0.0
        n_updates         = 0

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(N, device=self.device)
            for start in range(0, N, self.minibatch_size):
                end    = min(start + self.minibatch_size, N)
                mb_idx = perm[start:end]

                mb_seq    = data.seq_indices[mb_idx]
                mb_mask   = data.mutable_mask[mb_idx]
                mb_old_lp = old_log_probs[mb_idx]
                mb_adv    = advantages[mb_idx]
                mb_ret    = returns[mb_idx]

                proposal   = self.proposer(
                    mb_seq, mb_mask, pos_temp=self.pos_temp, nt_temp=self.nt_temp,
                )
                new_lp     = proposal.total_log_prob
                entropy    = proposal.entropy
                h_mb       = self.proposer.encode(mb_seq)
                new_values = self.critic(h_mb)

                ratio  = torch.exp(new_lp - mb_old_lp)
                surr1  = ratio * mb_adv
                surr2  = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                policy_loss   = -torch.min(surr1, surr2).mean()
                value_loss    = F.mse_loss(new_values, mb_ret)
                entropy_bonus = -entropy.mean()

                loss = (policy_loss
                        + self.value_coef * value_loss
                        + self.entropy_coef * entropy_bonus)

                self.opt_policy.zero_grad()
                self.opt_critic.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.proposer.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(),   self.max_grad_norm)
                self.opt_policy.step()
                self.opt_critic.step()

                with torch.no_grad():
                    clip_frac          = ((ratio - 1.0).abs() > self.clip_eps).float().mean()
                    total_policy_loss += policy_loss.item()
                    total_value_loss  += value_loss.item()
                    total_entropy     += entropy.mean().item()
                    total_clip_frac   += clip_frac.item()
                    n_updates         += 1

        self.entropy_coef = max(self.entropy_min, self.entropy_coef * self.entropy_decay)
        self.global_step += 1

        return {
            "policy_loss":   total_policy_loss / max(n_updates, 1),
            "value_loss":    total_value_loss  / max(n_updates, 1),
            "entropy":       total_entropy     / max(n_updates, 1),
            "clip_fraction": total_clip_frac   / max(n_updates, 1),
            "entropy_coef":  self.entropy_coef,
        }

    def step(self) -> dict:
        rollout_logs = self.collect_rollouts()
        update_logs  = self.update()
        return {**rollout_logs, **update_logs}


# ═════════════════════════════════════════════════════════════════════════════
# Display helpers (rich)
# ═════════════════════════════════════════════════════════════════════════════

def _print_start_banner(
    run_dir:    Path,
    parent:     str,
    parent_baseline: dict[str, float],
    cfg:        dict,
    device:     torch.device,
    n_proposer: int,
    n_critic:   int,
    classifier_path: str | Path,
    classifier_val_auc: float,
) -> None:
    dev_str = str(device)
    if device.type == "cuda":
        dev_str = f"cuda  —  {torch.cuda.get_device_name(0)}"

    ec     = cfg["enrich"]
    ppo    = ec["ppo"]
    pr     = ec["reward"]
    prop   = ec["proposer"]
    n_iter = ppo["n_iterations"]
    per_step_samples = ppo["n_rollout_steps"] * ppo["rollout_batch_size"]

    rows = [
        f"  [bold cyan]Run dir[/bold cyan]       {run_dir}",
        f"  [bold cyan]Classifier[/bold cyan]    {classifier_path}  "
        f"[dim](val_auc={classifier_val_auc:.4f})[/dim]",
        f"  [bold cyan]NT model[/bold cyan]      {ec['nt']['model_id']}",
        f"  [bold cyan]Device[/bold cyan]        {dev_str}",
        "",
        f"  [bold magenta]Parent[/bold magenta]        [white]{parent}[/white]  "
        f"[dim](L={len(parent)})[/dim]",
        f"  [bold magenta]Baseline[/bold magenta]      "
        f"reward=[yellow]{parent_baseline['raw']:.4f}[/yellow]  "
        f"conf=[green]{parent_baseline['confidence']:.4f}[/green]  "
        f"plaus=[green]{parent_baseline['plausibility']:.4f}[/green]",
        "",
        f"  [bold cyan]Reward[/bold cyan]        "
        f"α·conf + β·plaus  →  α={pr['confidence_weight']}  β={pr['plausibility_weight']}  "
        f"delta={'on' if pr['use_delta'] else 'off'} (×{pr['delta_scale']})",
        f"  [bold cyan]Proposer[/bold cyan]      "
        f"d_model={prop['d_model']}  layers={prop['n_enc_layers']}  "
        f"heads={prop['n_heads']}  [dim]{n_proposer:,} params[/dim]",
        f"  [bold cyan]Critic[/bold cyan]        d_hidden={ec['critic']['d_hidden']}  "
        f"[dim]{n_critic:,} params[/dim]",
        f"  [bold cyan]PPO[/bold cyan]           "
        f"{n_iter} iters × ({ppo['n_rollout_steps']} rollouts × "
        f"{ppo['rollout_batch_size']} batch = {per_step_samples} samples) × "
        f"{ppo['ppo_epochs']} epochs",
        f"  [bold cyan]Mutations[/bold cyan]     "
        f"{prop['min_mutations']}-{prop['max_mutations']} per rollout  "
        f"temps: pos={ppo['pos_temp']} nt={ppo['nt_temp']}",
    ]
    console.print(
        Panel("\n".join(rows), title="[bold]AptaBind Enrich[/bold]",
              border_style="magenta", padding=(0, 1))
    )
    console.print()


def _annotate_sequence(parent: str, mutant: str) -> str:
    """Render mutant with [bold red] markup on changed positions."""
    out = []
    for p, m in zip(parent, mutant):
        if p == m:
            out.append(m)
        else:
            out.append(f"[bold red on grey15]{m}[/bold red on grey15]")
    if len(mutant) > len(parent):
        out.append(f"[bold red on grey15]{mutant[len(parent):]}[/bold red on grey15]")
    return "".join(out)


def _format_mutations(muts: list[tuple[int, str, str]]) -> str:
    if not muts:
        return "(none)"
    return "  ".join(f"[yellow]{old}{pos}{new}[/yellow]" for pos, old, new in muts)


def _step_line(iteration: int, n_iter: int, logs: dict, parent: str) -> None:
    """Permanent per-iteration summary printed via rich console."""
    muts     = logs["iter_best_mutations"]
    n_mut    = len(muts)
    mut_seq  = logs["iter_best_seq"]
    annotated = _annotate_sequence(parent, mut_seq)

    is_new_best = (abs(logs["best_ever_reward"] - logs["iter_best_reward"]) < 1e-9
                   and logs["iter_best_reward"] > 0)
    new_best_tag = "  [bold green]★ NEW BEST[/bold green]" if is_new_best else ""

    header = (
        f"Step [cyan]{iteration:4d}[/cyan]/{n_iter}  "
        f"R_avg=[yellow]{logs['reward_mean']:+.4f}[/yellow]  "
        f"R_max=[yellow]{logs['reward_max']:+.4f}[/yellow]  "
        f"conf_mean=[green]{logs['confidence_mean']:.4f}[/green]  "
        f"plaus_mean=[green]{logs['plausibility_mean']:.4f}[/green]  "
        f"best_ever=[bold yellow]{logs['best_ever_reward']:+.4f}[/bold yellow]"
        f"{new_best_tag}"
    )
    ppo_line = (
        f"  [dim]π_loss={logs['policy_loss']:.4f}  "
        f"V_loss={logs['value_loss']:.4f}  "
        f"H={logs['entropy']:.3f}  "
        f"clip={logs['clip_fraction']:.3f}  "
        f"H_coef={logs['entropy_coef']:.5f}[/dim]"
    )
    mutant_line = (
        f"  [bold magenta]Mutant[/bold magenta] "
        f"([cyan]{n_mut} mut[/cyan], R=[yellow]{logs['iter_best_reward']:+.4f}[/yellow], "
        f"conf=[green]{logs['iter_best_confidence']:.4f}[/green], "
        f"plaus=[green]{logs['iter_best_plausibility']:.4f}[/green])  "
        f"{annotated}"
    )
    mut_list   = f"  [dim]Mutations:[/dim] {_format_mutations(muts)}"

    console.print(header)
    console.print(ppo_line)
    console.print(mutant_line)
    console.print(mut_list)
    console.print()


def _make_progress_bar() -> Progress:
    return Progress(
        SpinnerColumn(style="magenta"),
        TextColumn("[bold magenta]{task.description}"),
        BarColumn(bar_width=38, style="magenta", complete_style="bold magenta"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("[dim]{task.fields[stats]}[/dim]"),
        console=console,
        transient=False,        # keep the bar visible while step summaries scroll
        refresh_per_second=6,
    )


def _print_final_summary(
    parent:              str,
    parent_baseline:     dict[str, float],
    best_seq:            str,
    best_reward:         float,
    best_confidence:     float,
    best_plausibility:   float,
    total_time:          float,
    run_dir:             Path,
) -> None:
    mutations = [
        (i, p_nt, m_nt)
        for i, (p_nt, m_nt) in enumerate(zip(parent, best_seq))
        if p_nt != m_nt
    ]

    tbl = Table(box=None, show_header=False, pad_edge=False, padding=(0, 2))
    tbl.add_column("Field",  style="bold cyan",  min_width=18)
    tbl.add_column("Value",  justify="left")

    tbl.add_row("Parent", f"[white]{parent}[/white]")
    tbl.add_row("Parent reward",        f"{parent_baseline['raw']:.4f}")
    tbl.add_row("Parent confidence",    f"{parent_baseline['confidence']:.4f}")
    tbl.add_row("Parent plausibility",  f"{parent_baseline['plausibility']:.4f}")
    tbl.add_row("[dim]──────────────────[/dim]", "")
    tbl.add_row("Best mutant", _annotate_sequence(parent, best_seq))
    tbl.add_row("Best reward",       f"[bold yellow]{best_reward:+.4f}[/bold yellow]")
    tbl.add_row("Best confidence",   f"[bold green]{best_confidence:.4f}[/bold green]")
    tbl.add_row("Best plausibility", f"[bold green]{best_plausibility:.4f}[/bold green]")
    tbl.add_row("# mutations", f"{len(mutations)}")
    tbl.add_row("Mutations", _format_mutations(mutations))
    tbl.add_row("[dim]──────────────────[/dim]", "")
    tbl.add_row("Total time", f"{total_time/60:.1f} min")
    tbl.add_row("Run dir",    f"[dim]{run_dir}[/dim]")

    console.print()
    console.print(
        Panel(tbl, title="[bold green]Optimization Complete[/bold green]",
              border_style="green", padding=(0, 1))
    )


# ═════════════════════════════════════════════════════════════════════════════
# Run manager (timestamped output dir, checkpoints, history, top sequences)
# ═════════════════════════════════════════════════════════════════════════════

class RunManager:
    """
    Handles run directory creation, checkpointing, JSONL history, and CSV output.
    Each invocation creates runs/enrich_<YYYY-MM-DD_HH-MM-SS>/ so nothing is overwritten.
    """

    def __init__(self, base_dir: str | Path, cfg: dict) -> None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.dir = Path(base_dir) / f"enrich_{ts}"
        self.dir.mkdir(parents=True, exist_ok=True)

        # Save a copy of the config used
        with open(self.dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        # File handles
        self.history_path = self.dir / "history.jsonl"
        self.history_fh   = open(self.history_path, "w", buffering=1)   # line-buffered
        self.last_ckpt    = self.dir / "last_checkpoint.pt"
        self.best_ckpt    = self.dir / "best_checkpoint.pt"
        self.top_csv      = self.dir / "top_sequences.csv"

        # Add a file handler so the rich console transcript also gets dumped
        self._log_path  = self.dir / "enrich.log"
        plain = logging.FileHandler(self._log_path)
        plain.setFormatter(logging.Formatter("%(asctime)s %(message)s",
                                             datefmt="%H:%M:%S"))
        logging.getLogger().addHandler(plain)

        log.info(f"Run directory: [bold]{self.dir}[/bold]")

    # ─────────────────────────────────────────────────────────────────────────
    def log_history(self, iteration: int, logs: dict) -> None:
        """Append one JSON line per iteration (skip non-JSON-serializable types)."""
        rec = {"iteration": iteration}
        for k, v in logs.items():
            if k == "iter_best_mutations":
                rec[k] = [(int(i), str(o), str(n)) for (i, o, n) in v]
            elif isinstance(v, (int, float, str, bool)) or v is None:
                rec[k] = v
            elif isinstance(v, torch.Tensor):
                rec[k] = v.tolist() if v.numel() < 64 else None
            else:
                rec[k] = str(v)
        self.history_fh.write(json.dumps(rec) + "\n")

    # ─────────────────────────────────────────────────────────────────────────
    def save_checkpoint(
        self,
        path:       Path,
        iteration:  int,
        trainer:    PPOTrainer,
        cfg:        dict,
    ) -> None:
        torch.save({
            "iteration":           iteration,
            "proposer_state_dict": trainer.proposer.state_dict(),
            "critic_state_dict":   trainer.critic.state_dict(),
            "opt_policy_state":    trainer.opt_policy.state_dict(),
            "opt_critic_state":    trainer.opt_critic.state_dict(),
            "best_reward":         trainer.best_reward,
            "best_sequence":       trainer.best_sequence,
            "best_confidence":     trainer.best_confidence,
            "best_plausibility":   trainer.best_plausibility,
            "parent_sequence":     trainer.parent_sequence,
            "entropy_coef":        trainer.entropy_coef,
            "config":              cfg,
        }, path)

    def save_last(self, iteration: int, trainer: PPOTrainer, cfg: dict) -> None:
        self.save_checkpoint(self.last_ckpt, iteration, trainer, cfg)

    def save_best(self, iteration: int, trainer: PPOTrainer, cfg: dict) -> None:
        self.save_checkpoint(self.best_ckpt, iteration, trainer, cfg)

    # ─────────────────────────────────────────────────────────────────────────
    def save_top_sequences(self, records: list[dict], k: int) -> None:
        """
        Dump the top-K unique mutants by reward to a CSV file.
        """
        # Deduplicate by sequence, keep the highest-reward instance
        best_by_seq: dict[str, dict] = {}
        for r in records:
            s = r["sequence"]
            if s not in best_by_seq or r["reward"] > best_by_seq[s]["reward"]:
                best_by_seq[s] = r

        sorted_recs = sorted(best_by_seq.values(), key=lambda x: x["reward"], reverse=True)[:k]

        with open(self.top_csv, "w") as f:
            f.write("rank,sequence,reward,confidence,plausibility\n")
            for rank, r in enumerate(sorted_recs, 1):
                f.write(f"{rank},{r['sequence']},"
                        f"{r['reward']:.6f},{r['confidence']:.6f},{r['plausibility']:.6f}\n")
        log.info(f"Saved top {len(sorted_recs)} sequences → [bold]{self.top_csv}[/bold]")

    def close(self) -> None:
        self.history_fh.close()


# ═════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═════════════════════════════════════════════════════════════════════════════

def get_device() -> torch.device:
    if torch.cuda.is_available():
        d = torch.device("cuda")
        log.info(f"GPU: [bold]{torch.cuda.get_device_name(0)}[/bold]")
    elif torch.backends.mps.is_available():
        d = torch.device("mps")
        log.info("Device: [bold]Apple MPS[/bold]")
    else:
        d = torch.device("cpu")
        log.warning("No GPU detected — using [bold]CPU[/bold]. Will be slow.")
    return d


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if "enrich" not in cfg:
        raise KeyError(
            f"'enrich' section not found in {args.config}. "
            f"Add the enrich section to your config.yaml."
        )
    ec = cfg["enrich"]

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed = ec.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── Parent sequence ──────────────────────────────────────────────────────
    parent = args.parent or ec.get("parent_sequence")
    if not parent:
        parent = random_aptamer_sequence(length=ec.get("parent_length", 35))
        log.info(f"Generated random parent sequence ([dim]L={len(parent)}[/dim])")
    parent = parent.upper().strip()
    if any(c not in NT_TO_IDX for c in parent):
        raise ValueError(
            f"Parent contains characters outside ATCG: "
            f"{set(parent) - set(NT_TO_IDX.keys())}"
        )

    # ── Device + run dir ─────────────────────────────────────────────────────
    device = get_device()
    runs   = RunManager(ec.get("run_dir", "runs"), cfg)

    # ── Reward components ─────────────────────────────────────────────────────
    nt = NTEmbedder(
        model_id   = ec["nt"]["model_id"],
        layer      = ec["nt"].get("layer", -1),
        max_length = ec["nt"].get("max_length", 128),
        device     = device,
    )
    scorer = AptamerScorer(
        ckpt_path = args.classifier or ec["classifier_ckpt"],
        nt        = nt,
        device    = device,
    )
    plaus_cfg     = ec["reward"]["plaus"]
    plausibility  = AptamerPlausibility(
        min_length      = plaus_cfg["min_length"],
        max_length      = plaus_cfg["max_length"],
        gc_low          = plaus_cfg["gc_low"],
        gc_high         = plaus_cfg["gc_high"],
        max_run         = plaus_cfg["max_run"],
        max_composition = plaus_cfg["max_composition"],
        entropy_min     = plaus_cfg["entropy_min"],
    )
    reward_fn = AptamerReward(
        scorer              = scorer,
        plausibility        = plausibility,
        confidence_weight   = ec["reward"]["confidence_weight"],
        plausibility_weight = ec["reward"]["plausibility_weight"],
        use_delta           = ec["reward"]["use_delta"],
        delta_scale         = ec["reward"]["delta_scale"],
        embed_batch         = ec["nt"].get("embed_batch", 32),
    )
    parent_baseline = reward_fn.set_parent_baseline(parent)

    # ── Models ────────────────────────────────────────────────────────────────
    prop_cfg = ec["proposer"]
    proposer = MutationProposer(
        seq_length     = len(parent),
        d_model        = prop_cfg["d_model"],
        n_heads        = prop_cfg["n_heads"],
        n_enc_layers   = prop_cfg["n_enc_layers"],
        d_ff           = prop_cfg["d_ff"],
        max_mutations  = prop_cfg["max_mutations"],
        min_mutations  = prop_cfg["min_mutations"],
        neighborhood_k = prop_cfg["neighborhood_k"],
        dropout        = prop_cfg["dropout"],
    ).to(device)
    critic = ValueNetwork(
        d_model  = prop_cfg["d_model"],
        d_hidden = ec["critic"]["d_hidden"],
    ).to(device)

    n_proposer = sum(p.numel() for p in proposer.parameters() if p.requires_grad)
    n_critic   = sum(p.numel() for p in critic.parameters()   if p.requires_grad)

    # ── PPO trainer ───────────────────────────────────────────────────────────
    ppo = ec["ppo"]
    trainer = PPOTrainer(
        proposer           = proposer,
        critic             = critic,
        reward_fn          = reward_fn,
        parent_sequence    = parent,
        device             = device,
        mutable_positions  = ec.get("mutable_positions"),
        lr_policy          = float(ppo["lr_policy"]),
        lr_critic          = float(ppo["lr_critic"]),
        clip_eps           = ppo["clip_eps"],
        entropy_coef       = ppo["entropy_coef"],
        value_coef         = ppo["value_coef"],
        max_grad_norm      = ppo["max_grad_norm"],
        ppo_epochs         = ppo["ppo_epochs"],
        minibatch_size     = ppo["minibatch_size"],
        rollout_batch_size = ppo["rollout_batch_size"],
        n_rollout_steps    = ppo["n_rollout_steps"],
        pos_temp           = ppo["pos_temp"],
        nt_temp            = ppo["nt_temp"],
        entropy_decay      = ppo["entropy_decay"],
        entropy_min        = ppo["entropy_min"],
    )

    # ── Start banner ──────────────────────────────────────────────────────────
    _print_start_banner(
        run_dir            = runs.dir,
        parent             = parent,
        parent_baseline    = parent_baseline,
        cfg                = cfg,
        device             = device,
        n_proposer         = n_proposer,
        n_critic           = n_critic,
        classifier_path    = scorer.ckpt_path,
        classifier_val_auc = float(scorer.val_metrics.get("auc", float("nan"))),
    )

    # ── Optimization loop ─────────────────────────────────────────────────────
    n_iter     = ppo["n_iterations"]
    save_every = ec.get("save_every", 25)
    log_every  = ec.get("log_every",   1)

    total_start = time.perf_counter()
    prev_best   = trainer.best_reward

    with _make_progress_bar() as progress:
        task = progress.add_task("PPO optimization", total=n_iter, stats="")

        for iteration in range(1, n_iter + 1):
            logs = trainer.step()
            runs.log_history(iteration, logs)

            # Per-step rich panel
            if iteration % log_every == 0:
                _step_line(iteration, n_iter, logs, parent)

            # Save best checkpoint when reward improves
            if trainer.best_reward > prev_best:
                runs.save_best(iteration, trainer, cfg)
                prev_best = trainer.best_reward

            # Save last checkpoint periodically
            if iteration % save_every == 0 or iteration == n_iter:
                runs.save_last(iteration, trainer, cfg)

            # Progress bar postfix
            progress.update(
                task, advance=1,
                stats=(
                    f"R_avg={logs['reward_mean']:+.3f}  "
                    f"best={trainer.best_reward:+.3f}  "
                    f"conf={logs['confidence_mean']:.3f}"
                ),
            )

    total_time = time.perf_counter() - total_start

    # ── Final summary + outputs ───────────────────────────────────────────────
    _print_final_summary(
        parent            = parent,
        parent_baseline   = parent_baseline,
        best_seq          = trainer.best_sequence,
        best_reward       = trainer.best_reward,
        best_confidence   = trainer.best_confidence,
        best_plausibility = trainer.best_plausibility,
        total_time        = total_time,
        run_dir           = runs.dir,
    )

    runs.save_top_sequences(trainer.top_k_records, ec.get("top_k_sequences", 20))
    runs.save_last(n_iter, trainer, cfg)
    runs.close()
    log.info("Done.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RL aptamer optimizer (PPO) with NT + AptamerClassifier reward.",
    )
    p.add_argument("--config", "-c", default="config.yaml",
                   help="Path to config YAML (default: config.yaml)")
    p.add_argument("--parent", "-p", default=None,
                   help="Override parent DNA sequence (otherwise read from config)")
    p.add_argument("--classifier", "-m", default=None,
                   help="Override AptamerClassifier checkpoint path")
    return p.parse_args()


if __name__ == "__main__":
    main()
