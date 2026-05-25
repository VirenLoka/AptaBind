"""
inference.py
------------
Run the trained AptamerClassifier on any embedding pickle file.

Usage:
    # Use paths from config.yaml:
    python inference.py

    # Override input/output at the command line:
    python inference.py --input new_embeddings.pkl --output preds.csv

    # Custom confidence threshold:
    python inference.py --threshold 0.3

Output CSV columns:
    seq_id | sequence | true_label | enrichment_score | probability | predicted_class
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from model import AptamerClassifier, build_criterion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        d = torch.device("cuda")
        log.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        d = torch.device("mps")
        log.info("Using Apple MPS")
    else:
        d = torch.device("cpu")
        log.info("Using CPU")
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Core inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model:      AptamerClassifier,
    records:    list[dict],
    device:     torch.device,
    batch_size: int,
    threshold:  float,
) -> list[dict]:
    """
    Run the model over all records in batches.

    Parameters
    ----------
    model      : trained AptamerClassifier
    records    : list of dicts, each must have 'embedding' (np.ndarray)
    device     : torch device
    batch_size : number of records per forward pass
    threshold  : sigmoid probability threshold for predicting class 1

    Returns
    -------
    List of result dicts with prediction columns appended.
    """
    model.eval()
    results = []
    n_total = len(records)

    for start in range(0, n_total, batch_size):
        batch = records[start : start + batch_size]

        embeddings = torch.tensor(
            np.stack([r["embedding"] for r in batch]),
            dtype=torch.float32,
            device=device,
        )
        logits = model(embeddings)                          # (B,)
        probs  = torch.sigmoid(logits).cpu().numpy()        # (B,) ∈ [0,1]

        for rec, prob in zip(batch, probs):
            results.append({
                "seq_id":           rec.get("seq_id", ""),
                "sequence":         rec.get("sequence", ""),
                "true_label":       rec.get("label", "unknown"),
                "enrichment_score": rec.get("enrichment_score", ""),
                "probability":      round(float(prob), 6),
                "predicted_class":  int(prob >= threshold),
            })

        if (start + len(batch)) % (batch_size * 20) == 0 or start + len(batch) == n_total:
            log.info(f"  Processed {start + len(batch):,} / {n_total:,}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Optional evaluation (when true labels are available)
# ─────────────────────────────────────────────────────────────────────────────

def _try_evaluate(df: pd.DataFrame, label_map: dict[str, int]) -> None:
    """
    If the input data has known labels that exist in label_map,
    compute and log classification metrics.
    """
    if "true_label" not in df.columns:
        return

    known = df["true_label"].isin(label_map.keys())
    if known.sum() == 0:
        return

    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        f1_score,
        accuracy_score,
        classification_report,
    )

    sub       = df[known].copy()
    y_true    = sub["true_label"].map(label_map).values
    y_prob    = sub["probability"].values
    y_pred    = sub["predicted_class"].values
    n_labeled = len(sub)

    log.info(f"\nEvaluation on {n_labeled:,} labeled samples:")
    try:
        auc   = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
        f1    = f1_score(y_true, y_pred, zero_division=0)
        acc   = accuracy_score(y_true, y_pred)
        log.info(f"  AUC-ROC : {auc:.4f}")
        log.info(f"  AUPRC   : {auprc:.4f}")
        log.info(f"  F1      : {f1:.4f}")
        log.info(f"  Accuracy: {acc:.4f}")
        log.info("\n" + classification_report(y_true, y_pred, zero_division=0))
    except ValueError as e:
        log.warning(f"Could not compute metrics: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def infer(
    config_path:   str,
    input_pickle:  str | None = None,
    output_csv:    str | None = None,
    threshold:     float | None = None,
) -> pd.DataFrame:
    """
    Load model, run inference on a pickle file, save CSV, return DataFrame.

    Parameters override the corresponding values in config.yaml when provided.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    inf_cfg    = cfg["inference"]
    ckpt_path  = Path(inf_cfg["checkpoint_path"])
    pickle_in  = Path(input_pickle  or inf_cfg["input_pickle"])
    csv_out    = Path(output_csv    or inf_cfg["output_csv"])
    thr        = threshold if threshold is not None else float(inf_cfg.get("threshold", 0.5))
    batch_size = cfg["dataloader"]["batch_size"]

    device = get_device()

    # ── Load checkpoint & model ───────────────────────────────────────────────
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run train.py first to produce a checkpoint."
        )

    ckpt      = torch.load(ckpt_path, map_location=device)
    label_map = ckpt.get("label_map", {})

    # Build model from the config embedded in the checkpoint (most reliable),
    # falling back to the current config.yaml if the key is missing.
    ckpt_cfg  = ckpt.get("config", cfg)
    m         = ckpt_cfg["model"]
    model     = AptamerClassifier(
        embed_dim       = m["embed_dim"],
        num_patches     = m["num_patches"],
        d_model         = m["d_model"],
        nhead           = m["nhead"],
        num_layers      = m["num_layers"],
        dim_feedforward = m["dim_feedforward"],
        dropout         = m["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])

    val_metrics = ckpt.get("val_metrics", {})
    log.info(
        f"Loaded checkpoint — epoch {ckpt.get('epoch', '?')}  "
        f"val_auc={val_metrics.get('auc', float('nan')):.4f}  "
        f"val_auprc={val_metrics.get('auprc', float('nan')):.4f}"
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    if not pickle_in.exists():
        raise FileNotFoundError(f"Input pickle not found: {pickle_in}")

    with open(pickle_in, "rb") as f:
        raw: dict = pickle.load(f)

    records = [{"seq_id": sid, **entry} for sid, entry in raw.items()]
    log.info(f"Loaded {len(records):,} records from {pickle_in}")

    # ── Run inference ─────────────────────────────────────────────────────────
    log.info(f"Running inference  (threshold={thr}, batch_size={batch_size})...")
    results = run_inference(model, records, device, batch_size, thr)

    # ── Build output DataFrame ────────────────────────────────────────────────
    df = pd.DataFrame(results)

    # Prediction summary
    n_pos = (df["predicted_class"] == 1).sum()
    n_neg = (df["predicted_class"] == 0).sum()
    log.info(
        f"Predictions — positive: {n_pos:,}  negative: {n_neg:,}  "
        f"pos_rate: {n_pos / len(df):.4f}"
    )

    # Optional evaluation if labels are known
    if label_map:
        _try_evaluate(df, label_map)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)
    log.info(f"Predictions saved → {csv_out}  ({len(df):,} rows)")

    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference with AptamerClassifier")
    p.add_argument(
        "--config", "-c", default="config.yaml",
        help="Path to config YAML (default: config.yaml)",
    )
    p.add_argument(
        "--input", "-i", default=None,
        help="Input embedding pickle (overrides inference.input_pickle in config)",
    )
    p.add_argument(
        "--output", "-o", default=None,
        help="Output CSV path (overrides inference.output_csv in config)",
    )
    p.add_argument(
        "--threshold", "-t", type=float, default=None,
        help="Sigmoid probability threshold for class 1 (overrides config)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    infer(args.config, args.input, args.output, args.threshold)
