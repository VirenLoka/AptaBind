"""
extract_nt_embeddings.py
------------------------
Extracts per-sequence embeddings from the Nucleotide Transformer
for a list of DNA aptamer sequences.

Produces three outputs:
  - mean pooled embedding  (one vector per sequence, recommended)
  - CLS token embedding    (one vector per sequence, alternative)
  - per-layer embeddings   (optional, for layer probing)

Installation:
    pip install --upgrade git+https://github.com/huggingface/transformers.git
    pip install torch
"""

import torch
import numpy as np
import argparse
import os
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# MODEL OPTIONS
# ─────────────────────────────────────────────
MODEL_OPTIONS = {
    "500m-human":    "InstaDeepAI/nucleotide-transformer-500m-human-ref",
    "500m-1000g":    "InstaDeepAI/nucleotide-transformer-500m-1000g",
    "2.5b-1000g":    "InstaDeepAI/nucleotide-transformer-2.5b-1000g",
    "2.5b-multi":    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
}


def load_sequences(filepath: str) -> tuple[list[str], list[str], list[int]]:
    """
    Load sequences from TSV (sequence<tab>score) or plain FASTA/text.
    Returns (sequences, labels, original_indices).
    Filters out sequences with ambiguous bases before embedding.
    """
    sequences, labels = [], []
    skipped = 0

    with open(filepath) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Handle TSV (seq<tab>score or seq<tab>label)
            parts = line.split("\t")
            seq = parts[0].upper()
            label = parts[1] if len(parts) > 1 else "unknown"

            # Skip ambiguous sequences — NT tokenizer handles N poorly
            if not all(b in "ATCG" for b in seq):
                log.warning(f"Line {lineno}: skipping '{seq[:20]}...' — ambiguous bases")
                skipped += 1
                continue

            sequences.append(seq)
            labels.append(label)

    log.info(f"Loaded {len(sequences):,} sequences ({skipped} skipped)")
    return sequences, labels


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("Using Apple MPS")
    else:
        device = torch.device("cpu")
        log.warning("No GPU found — using CPU. Expect slow extraction on large datasets.")
    return device


def load_model(model_key: str, device: torch.device):
    """Load tokenizer and model from HuggingFace."""
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    model_id = MODEL_OPTIONS.get(model_key, model_key)  # allow raw HF ID too
    log.info(f"Loading model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(
        model_id,
        output_hidden_states=True,   # we need all hidden states
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    model = model.to(device)
    model.eval()

    log.info(
        f"Model loaded — "
        f"parameters: {sum(p.numel() for p in model.parameters()):,}  |  "
        f"hidden size: {model.config.hidden_size}"
    )
    return tokenizer, model


def extract_embeddings(
    sequences: list[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int = 16,
    layer: int = -1,          # which layer to extract from (-1 = last)
    max_length: int = 128,    # 35 nt aptamers need far less than model max
) -> dict[str, np.ndarray]:
    """
    Extract embeddings for all sequences in batches.

    Returns dict with:
        'mean_pool'  : (N, hidden_size) — attention-masked mean over tokens
        'cls'        : (N, hidden_size) — CLS token only
        'sequences'  : list of input sequences (in order)
    """
    all_mean_pool = []
    all_cls = []

    n_batches = (len(sequences) + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in range(n_batches):
            batch_seqs = sequences[batch_idx * batch_size : (batch_idx + 1) * batch_size]

            # ── Tokenize ──────────────────────────────────────────────
            # The NT tokenizer uses 6-mer tokens.
            # padding="max_length" pads all seqs to the same length.
            # max_length=128 is far more than enough for 35-nt aptamers
            # (35 nt / 6 ≈ 6 tokens + CLS + padding).
            tokens = tokenizer.batch_encode_plus(
                batch_seqs,
                return_tensors="pt",
                padding="max_length",
                max_length=max_length,
                truncation=True,
            )

            input_ids = tokens["input_ids"].to(device)

            # Attention mask: 1 for real tokens, 0 for padding
            # NT pads with pad_token_id — build mask from that
            attention_mask = (input_ids != tokenizer.pad_token_id).to(device)

            # ── Forward pass ──────────────────────────────────────────
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            # hidden_states: tuple of (n_layers + 1) tensors, each (B, T, H)
            # Index 0 is the embedding layer, -1 is the last transformer layer
            hidden = outputs.hidden_states[layer]  # (B, T, H)

            # ── CLS token embedding ───────────────────────────────────
            # Position 0 is always the CLS token in BERT-style models
            cls_emb = hidden[:, 0, :]  # (B, H)

            # ── Attention-masked mean pooling ─────────────────────────
            # Exclude padding tokens from the mean.
            # Expand mask to match hidden dim: (B, T) -> (B, T, H)
            mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
            sum_hidden = torch.sum(hidden * mask_expanded, dim=1)  # (B, H)
            sum_mask = attention_mask.sum(dim=1, keepdim=True).float()  # (B, 1)
            mean_emb = sum_hidden / sum_mask.clamp(min=1e-9)  # (B, H)

            # Move to CPU and store
            all_mean_pool.append(mean_emb.cpu().float().numpy())
            all_cls.append(cls_emb.cpu().float().numpy())

            if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
                log.info(f"  Processed batch {batch_idx + 1}/{n_batches}")

    return {
        "mean_pool": np.vstack(all_mean_pool),   # (N, H)
        "cls":       np.vstack(all_cls),          # (N, H)
        "sequences": sequences,
    }


def extract_all_layers(
    sequences: list[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int = 8,
    max_length: int = 128,
) -> np.ndarray:
    """
    Extract mean-pooled embeddings from EVERY layer.
    Useful for layer probing — finding which layer captures binding-relevant
    features best. Returns array of shape (N, n_layers, hidden_size).
    Memory-intensive — use small batch_size.
    """
    n_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
    all_layers = []

    with torch.no_grad():
        for batch_idx in range(0, len(sequences), batch_size):
            batch_seqs = sequences[batch_idx : batch_idx + batch_size]

            tokens = tokenizer.batch_encode_plus(
                batch_seqs,
                return_tensors="pt",
                padding="max_length",
                max_length=max_length,
                truncation=True,
            )

            input_ids = tokens["input_ids"].to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).to(device)

            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            # Stack all layers: (n_layers, B, T, H)
            hidden_all = torch.stack(outputs.hidden_states, dim=0)

            mask_expanded = attention_mask.unsqueeze(0).unsqueeze(-1).float()
            sum_hidden = torch.sum(hidden_all * mask_expanded, dim=2)
            sum_mask = attention_mask.sum(dim=1).float().unsqueeze(0).unsqueeze(-1)
            mean_all = sum_hidden / sum_mask.clamp(min=1e-9)
            # mean_all: (n_layers, B, H) -> (B, n_layers, H)
            mean_all = mean_all.permute(1, 0, 2)

            all_layers.append(mean_all.cpu().float().numpy())
            log.info(f"  Layer probe batch {batch_idx // batch_size + 1} done")

    return np.vstack(all_layers)  # (N, n_layers, H)


def save_outputs(
    embeddings: dict,
    labels: list[str],
    output_dir: str,
    model_key: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Save embeddings as .npy (fast, preserves float precision)
    mean_path = os.path.join(output_dir, f"embeddings_mean_{model_key}.npy")
    cls_path  = os.path.join(output_dir, f"embeddings_cls_{model_key}.npy")
    seq_path  = os.path.join(output_dir, f"sequences_{model_key}.txt")
    lab_path  = os.path.join(output_dir, f"labels_{model_key}.txt")

    np.save(mean_path, embeddings["mean_pool"])
    np.save(cls_path,  embeddings["cls"])

    with open(seq_path, "w") as f:
        f.write("\n".join(embeddings["sequences"]))
    with open(lab_path, "w") as f:
        f.write("\n".join(labels))

    log.info(f"Saved mean-pool embeddings : {mean_path}  shape={embeddings['mean_pool'].shape}")
    log.info(f"Saved CLS embeddings       : {cls_path}   shape={embeddings['cls'].shape}")
    log.info(f"Saved sequences            : {seq_path}")
    log.info(f"Saved labels               : {lab_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract Nucleotide Transformer embeddings for aptamer sequences."
    )
    p.add_argument("--input",   "-i", required=True,
                   help="TSV or text file of sequences")
    p.add_argument("--output",  "-o", default="nt_embeddings",
                   help="Output directory (default: nt_embeddings/)")
    p.add_argument("--model",   "-m", default="500m-1000g",
                   choices=list(MODEL_OPTIONS.keys()),
                   help="Which NT model to use (default: 500m-1000g)")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Batch size (reduce if OOM, default: 16)")
    p.add_argument("--layer",   type=int, default=-1,
                   help="Which hidden layer to extract (-1=last, default: -1)")
    p.add_argument("--max-length", type=int, default=128,
                   help="Tokenizer max length (default: 128, enough for 35-nt aptamers)")
    p.add_argument("--all-layers", action="store_true",
                   help="Extract all layers for layer probing (slow, memory-heavy)")
    return p.parse_args()


def main():
    args = parse_args()

    sequences, labels = load_sequences(args.input)
    if not sequences:
        log.error("No sequences loaded. Exiting.")
        return

    device = get_device()
    tokenizer, model = load_model(args.model, device)

    log.info(f"\nExtracting embeddings for {len(sequences):,} sequences...")
    log.info(f"  Model          : {MODEL_OPTIONS[args.model]}")
    log.info(f"  Layer          : {args.layer}")
    log.info(f"  Batch size     : {args.batch_size}")
    log.info(f"  Max token len  : {args.max_length}")

    embeddings = extract_embeddings(
        sequences, tokenizer, model, device,
        batch_size=args.batch_size,
        layer=args.layer,
        max_length=args.max_length,
    )
    embeddings["sequences"] = sequences

    if args.all_layers:
        log.info("Extracting all-layer embeddings for layer probing...")
        all_layer_embs = extract_all_layers(
            sequences, tokenizer, model, device,
            batch_size=max(1, args.batch_size // 4),
            max_length=args.max_length,
        )
        all_layer_path = os.path.join(
            args.output, f"embeddings_all_layers_{args.model}.npy"
        )
        os.makedirs(args.output, exist_ok=True)
        np.save(all_layer_path, all_layer_embs)
        log.info(f"Saved all-layer embeddings: {all_layer_path}  shape={all_layer_embs.shape}")

    save_outputs(embeddings, labels, args.output, args.model)
    log.info("\nDone.")


if __name__ == "__main__":
    main()