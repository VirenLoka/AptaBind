"""
extract_nt_embeddings.py
------------------------
Extracts per-sequence embeddings from the Nucleotide Transformer
for a list of DNA aptamer sequences.

Input format (TSV with header):
    sequence_id  sequence  label  enrichment_score

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
import pickle
import logging

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


def load_sequences(filepath: str) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Load sequences from a TSV with header:
        sequence_id  sequence  label  enrichment_score

    Returns (sequence_ids, sequences, labels, scores).
    Filters out the header row and any sequences with ambiguous bases.
    """
    sequence_ids, sequences, labels, scores = [], [], [], []
    skipped = 0

    # Resolve expected column indices from the header
    col_id, col_seq, col_label, col_score = 0, 1, 2, 3  # defaults

    with open(filepath) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")

            # Parse header row to map column names → indices robustly
            if lineno == 1 or (parts[0].lower() == "sequence_id" and "sequence" in line.lower()):
                header = [h.strip().lower() for h in parts]
                try:
                    col_id    = header.index("sequence_id")
                    col_seq   = header.index("sequence")
                    col_label = header.index("label")
                    col_score = header.index("enrichment_score")
                except ValueError as e:
                    log.warning(f"Header missing expected column: {e}. Using defaults (0,1,2,3).")
                continue

            if len(parts) <= col_seq:
                log.warning(f"Line {lineno}: too few columns, skipping.")
                skipped += 1
                continue

            seq_id = parts[col_id].strip()
            seq    = parts[col_seq].strip().upper()
            label  = parts[col_label].strip() if len(parts) > col_label else "unknown"
            score  = parts[col_score].strip() if len(parts) > col_score else "0.0"

            # Skip ambiguous sequences — NT tokenizer handles N poorly
            if not all(b in "ATCG" for b in seq):
                log.warning(f"Line {lineno} ({seq_id}): skipping '{seq[:20]}...' — ambiguous bases")
                skipped += 1
                continue

            sequence_ids.append(seq_id)
            sequences.append(seq)
            labels.append(label)
            scores.append(score)

    log.info(f"Loaded {len(sequences):,} sequences ({skipped} skipped)")
    return sequence_ids, sequences, labels, scores


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
        output_hidden_states=True,
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
    layer: int = -1,
    max_length: int = 128,
) -> dict[str, np.ndarray]:
    """
    Extract embeddings for all sequences in batches.
    Returns dict with mean_pool, cls, and sequence tracking.
    """
    all_mean_pool = []
    all_cls = []

    n_batches = (len(sequences) + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in range(n_batches):
            batch_seqs = sequences[batch_idx * batch_size : (batch_idx + 1) * batch_size]

            tokens = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding="max_length",
                max_length=max_length,
                truncation=True,
            )

            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)

            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            hidden = outputs.hidden_states[layer]
            cls_emb = hidden[:, 0, :]

            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_hidden = torch.sum(hidden * mask_expanded, dim=1)
            sum_mask = attention_mask.sum(dim=1, keepdim=True).float()
            mean_emb = sum_hidden / sum_mask.clamp(min=1e-9)

            all_mean_pool.append(mean_emb.cpu().float().numpy())
            all_cls.append(cls_emb.cpu().float().numpy())

            if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
                log.info(f"  Processed batch {batch_idx + 1}/{n_batches}")

    return {
        "mean_pool": np.vstack(all_mean_pool),
        "cls":       np.vstack(all_cls),
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
    Extract mean-pooled embeddings from EVERY layer for layer probing.
    """
    all_layers = []

    with torch.no_grad():
        for batch_idx in range(0, len(sequences), batch_size):
            batch_seqs = sequences[batch_idx : batch_idx + batch_size]

            tokens = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding="max_length",
                max_length=max_length,
                truncation=True,
            )

            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)

            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            hidden_all = torch.stack(outputs.hidden_states, dim=0)

            mask_expanded = attention_mask.unsqueeze(0).unsqueeze(-1).float()
            sum_hidden = torch.sum(hidden_all * mask_expanded, dim=2)
            sum_mask = attention_mask.sum(dim=1).float().unsqueeze(0).unsqueeze(-1)
            mean_all = sum_hidden / sum_mask.clamp(min=1e-9)
            mean_all = mean_all.permute(1, 0, 2)

            all_layers.append(mean_all.cpu().float().numpy())
            log.info(f"  Layer probe batch {batch_idx // batch_size + 1} done")

    return np.vstack(all_layers)


def save_pickle(
    embeddings: dict,
    sequence_ids: list[str],
    sequences: list[str],
    labels: list[str],
    scores: list[str],
    output_path: str,
) -> None:
    """
    Save a nested dict keyed by seq_id:
        { seq_id: { sequence, embedding, label, enrichment_score } }
    """
    mean_pool = embeddings["mean_pool"]  # (N, hidden_size)

    result = {
        seq_id: {
            "sequence":         sequences[i],
            "embedding":        mean_pool[i],
            "label":            labels[i],
            "enrichment_score": scores[i],
        }
        for i, seq_id in enumerate(sequence_ids)
    }

    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    log.info(f"Saved pickle: {output_path}  ({len(result):,} sequences, embedding shape: {mean_pool[0].shape})")


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract Nucleotide Transformer embeddings for aptamer sequences."
    )
    p.add_argument("--input",   "-i", required=True,
                   help="TSV file with header: sequence_id, sequence, label, enrichment_score")
    p.add_argument("--output",  "-o", default="nt_embeddings.pkl",
                   help="Output pickle file (default: nt_embeddings.pkl)")
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

    sequence_ids, sequences, labels, scores = load_sequences(args.input)
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
        all_layer_path = os.path.splitext(args.output)[0] + f"_all_layers_{args.model}.npy"
        np.save(all_layer_path, all_layer_embs)
        log.info(f"Saved all-layer embeddings: {all_layer_path}  shape={all_layer_embs.shape}")

    save_pickle(embeddings, sequence_ids, sequences, labels, scores, args.output)
    log.info("\nDone.")


if __name__ == "__main__":
    main()