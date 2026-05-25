"""
inference.py
------------
Run the trained AptamerClassifier on any embedding pickle file.

Usage:
    python inference.py                              # use paths from config.yaml
    python inference.py --input new_embeddings.pkl   # override input pickle
    python inference.py --output preds.csv           # override output CSV
    python inference.py --threshold 0.3              # override decision threshold

Terminal output:
    • Start banner  — checkpoint / input / device summary
    • Progress bar  — live batch processing with running pos/neg tally
    • Summary panel — prediction counts + evaluation metrics (if labels present)

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
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
    TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)
from rich.table import Table
from rich import box
import yaml

# ── Console + logging ────────────────────────────────────────────────────────
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%H:%M:%S]",
    handlers=[RichHandler(console=console, show_path=False,
                          markup=True, rich_tracebacks=True)],
)
log = logging.getLogger(__name__)

from model import AptamerClassifier, build_criterion   # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        d = torch.device("cuda")
        log.info(f"GPU: [bold]{torch.cuda.get_device_name(0)}[/bold]")
    elif torch.backends.mps.is_available():
        d = torch.device("mps")
        log.info("Device: [bold]Apple MPS[/bold]")
    else:
        d = torch.device("cpu")
        log.info("Device: [bold]CPU[/bold]")
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_inference_banner(
    ckpt_path:   Path,
    pickle_path: Path,
    n_records:   int,
    threshold:   float,
    device:      torch.device,
    ckpt:        dict,
) -> None:
    vm  = ckpt.get("val_metrics", {})
    ep  = ckpt.get("epoch", "?")
    dev_str = str(device)
    if device.type == "cuda":
        dev_str = f"cuda  —  {torch.cuda.get_device_name(0)}"

    rows = [
        f"  [bold cyan]Checkpoint[/bold cyan]  {ckpt_path}  [dim](epoch {ep})[/dim]",
        f"  [bold cyan]Input[/bold cyan]       {pickle_path}",
        f"  [bold cyan]Sequences[/bold cyan]   [green]{n_records:,}[/green]",
        f"  [bold cyan]Threshold[/bold cyan]   {threshold}",
        f"  [bold cyan]Device[/bold cyan]      {dev_str}",
        f"  [bold cyan]Val AUC[/bold cyan]     "
        f"{vm.get('auc', float('nan')):.4f}  "
        f"[dim]AUPRC {vm.get('auprc', float('nan')):.4f}  "
        f"F1 {vm.get('f1', float('nan')):.4f}[/dim]",
    ]
    console.print(
        Panel("\n".join(rows), title="[bold]AptaBind Inference[/bold]",
              border_style="green", padding=(0, 1))
    )
    console.print()


def _print_summary_panel(
    df:           pd.DataFrame,
    csv_out:      Path,
    eval_metrics: dict | None,
) -> None:
    n_total  = len(df)
    n_pos    = int((df["predicted_class"] == 1).sum())
    n_neg    = n_total - n_pos
    pos_rate = n_pos / n_total if n_total else 0.0

    tbl = Table(box=None, show_header=False, pad_edge=False, padding=(0, 2))
    tbl.add_column("Metric", style="bold cyan",  min_width=22)
    tbl.add_column("Value",  style="bold",        justify="right", min_width=12)

    tbl.add_row("Total sequences",    f"{n_total:,}")
    tbl.add_row("Predicted positive", f"[green]{n_pos:,}[/green]  [dim]({pos_rate:.1%})[/dim]")
    tbl.add_row("Predicted negative", f"{n_neg:,}  [dim]({1-pos_rate:.1%})[/dim]")
    tbl.add_row("Output CSV",         f"[dim]{csv_out}[/dim]")

    if eval_metrics:
        tbl.add_row("", "")
        tbl.add_row("[dim]─── Evaluation (labeled data)[/dim]", "")
        tbl.add_row(
            f"[dim]  Labeled samples[/dim]",
            f"[dim]{eval_metrics['n_labeled']:,}[/dim]",
        )
        tbl.add_row("  AUC-ROC",  f"[bold green]{eval_metrics['auc']:.4f}[/bold green]")
        tbl.add_row("  AUPRC",    f"[bold green]{eval_metrics['auprc']:.4f}[/bold green]")
        tbl.add_row("  F1",       f"{eval_metrics['f1']:.4f}")
        tbl.add_row("  Accuracy", f"{eval_metrics['accuracy']:.4f}")

    console.print(
        Panel(tbl, title="[bold green]Prediction Summary[/bold green]",
              border_style="green", padding=(0, 1))
    )


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
    Batch-process all records through the model.
    Shows a persistent rich progress bar (stays visible after completion).

    Returns a list of result dicts ready for pd.DataFrame().
    """
    model.eval()
    results: list[dict] = []
    n_total = len(records)
    n_pos   = 0

    with Progress(
        SpinnerColumn(style="green"),
        TextColumn("[bold green]{task.description}"),
        BarColumn(bar_width=45, style="green", complete_style="bold green"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("[dim]{task.fields[stats]}[/dim]"),
        console=console,
        transient=False,        # keep the bar visible after inference finishes
        refresh_per_second=10,
    ) as progress:
        task = progress.add_task(
            "Running inference",
            total=n_total,
            stats="",
        )

        for start in range(0, n_total, batch_size):
            batch = records[start : start + batch_size]

            embeddings = torch.tensor(
                np.stack([r["embedding"] for r in batch]),
                dtype=torch.float32,
                device=device,
            )
            logits = model(embeddings)
            probs  = torch.sigmoid(logits).cpu().numpy()

            for rec, prob in zip(batch, probs):
                pred = int(prob >= threshold)
                n_pos += pred
                results.append({
                    "seq_id":           rec.get("seq_id",            ""),
                    "sequence":         rec.get("sequence",          ""),
                    "true_label":       rec.get("label",       "unknown"),
                    "enrichment_score": rec.get("enrichment_score",  ""),
                    "probability":      round(float(prob), 6),
                    "predicted_class":  pred,
                })

            done = start + len(batch)
            progress.update(
                task, advance=len(batch),
                stats=(
                    f"pos={n_pos:,}  neg={done - n_pos:,}  "
                    f"rate={n_pos/done:.3f}"
                ),
            )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Optional evaluation (when labels are available)
# ─────────────────────────────────────────────────────────────────────────────

def _try_evaluate(df: pd.DataFrame, label_map: dict[str, int]) -> dict | None:
    """Return evaluation metrics dict if true labels are available; else None."""
    if "true_label" not in df.columns:
        return None
    known = df["true_label"].isin(label_map.keys())
    if known.sum() == 0:
        return None

    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score, accuracy_score,
    )
    sub    = df[known].copy()
    y_true = sub["true_label"].map(label_map).values
    y_prob = sub["probability"].values
    y_pred = sub["predicted_class"].values

    try:
        return {
            "n_labeled": len(sub),
            "auc":       roc_auc_score(y_true, y_prob),
            "auprc":     average_precision_score(y_true, y_prob),
            "f1":        f1_score(y_true, y_pred, zero_division=0),
            "accuracy":  accuracy_score(y_true, y_pred),
        }
    except ValueError as e:
        log.warning(f"Could not compute evaluation metrics: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def infer(
    config_path:  str,
    input_pickle: str | None = None,
    output_csv:   str | None = None,
    threshold:    float | None = None,
) -> pd.DataFrame:
    """
    Load model, run inference on a pickle file, save CSV, return DataFrame.
    CLI arguments override the corresponding config.yaml values when provided.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    inf_cfg    = cfg["inference"]
    ckpt_path  = Path(inf_cfg["checkpoint_path"])
    pickle_in  = Path(input_pickle or inf_cfg["input_pickle"])
    csv_out    = Path(output_csv   or inf_cfg["output_csv"])
    thr        = threshold if threshold is not None else float(inf_cfg.get("threshold", 0.5))
    batch_size = cfg["dataloader"]["batch_size"]

    device = get_device()

    # ── Load checkpoint ───────────────────────────────────────────────────────
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run train.py first to produce a checkpoint."
        )
    ckpt      = torch.load(ckpt_path, map_location=device)
    label_map = ckpt.get("label_map", {})

    # Rebuild model from the config embedded inside the checkpoint
    ckpt_cfg = ckpt.get("config", cfg)
    m        = ckpt_cfg["model"]
    model    = AptamerClassifier(
        embed_dim       = m["embed_dim"],
        num_patches     = m["num_patches"],
        d_model         = m["d_model"],
        nhead           = m["nhead"],
        num_layers      = m["num_layers"],
        dim_feedforward = m["dim_feedforward"],
        dropout         = m["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # ── Load data ─────────────────────────────────────────────────────────────
    if not pickle_in.exists():
        raise FileNotFoundError(f"Input pickle not found: {pickle_in}")

    log.info(f"Loading [bold]{pickle_in}[/bold]…")
    with open(pickle_in, "rb") as f:
        raw: dict = pickle.load(f)
    records = [{"seq_id": sid, **entry} for sid, entry in raw.items()]
    log.info(f"Loaded [green]{len(records):,}[/green] sequences")

    # ── Banner ────────────────────────────────────────────────────────────────
    _print_inference_banner(ckpt_path, pickle_in, len(records), thr, device, ckpt)

    # ── Inference ─────────────────────────────────────────────────────────────
    results = run_inference(model, records, device, batch_size, thr)
    df      = pd.DataFrame(results)

    # ── Optional evaluation ───────────────────────────────────────────────────
    eval_metrics = _try_evaluate(df, label_map) if label_map else None

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)

    # ── Summary panel ─────────────────────────────────────────────────────────
    console.print()
    _print_summary_panel(df, csv_out, eval_metrics)

    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference with AptamerClassifier")
    p.add_argument("--config",    "-c", default="config.yaml",
                   help="Config YAML path (default: config.yaml)")
    p.add_argument("--input",     "-i", default=None,
                   help="Input pickle  (overrides inference.input_pickle in config)")
    p.add_argument("--output",    "-o", default=None,
                   help="Output CSV    (overrides inference.output_csv in config)")
    p.add_argument("--threshold", "-t", type=float, default=None,
                   help="Decision threshold  (overrides inference.threshold in config)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    infer(args.config, args.input, args.output, args.threshold)
