"""
train.py
--------
Training loop for AptamerClassifier with rich progress bars and stat display.

Usage:
    python train.py
    python train.py --config config.yaml

Terminal output:
    • Start banner   — model / loss / data / device summary
    • Per-epoch bar  — live batch progress (transient, disappears after epoch)
    • Per-epoch line — colourised train/val stats printed permanently
    • History table  — all epochs in a rich table at training end
    • Test panel     — final held-out test results

Outputs:
    checkpoints/best_model.pt   — best validation metric checkpoint
    checkpoints/last_model.pt   — final epoch checkpoint
"""

import argparse
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, accuracy_score,
)
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
    TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text
from rich import box
import yaml

# ── Console + logging (configured before project imports so RichHandler
#    captures messages from dataset.py / model.py too) ────────────────────────
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%H:%M:%S]",
    handlers=[RichHandler(console=console, show_path=False,
                          markup=True, rich_tracebacks=True)],
)
log = logging.getLogger(__name__)

from dataset import get_dataloaders                    # noqa: E402
from model import AptamerClassifier, build_criterion   # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
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
        log.warning("No GPU detected — using [bold]CPU[/bold]. Training will be slow.")
    return d


def build_optimizer(model: nn.Module, t_cfg: dict) -> torch.optim.Optimizer:
    name = t_cfg.get("optimizer", "adamw").lower()
    lr, wd = t_cfg["learning_rate"], t_cfg.get("weight_decay", 1e-5)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"Unknown optimizer '{name}'. Choose: adamw | adam | sgd")


def build_scheduler(optimizer, t_cfg: dict, steps_per_epoch: int):
    name         = t_cfg.get("scheduler", "cosine").lower()
    epochs       = t_cfg["epochs"]
    warmup_steps = t_cfg.get("warmup_epochs", 5) * steps_per_epoch
    total_steps  = epochs * steps_per_epoch

    if name == "cosine":
        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if name == "step":
        step_size = t_cfg.get("step_size", 20) * steps_per_epoch
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=t_cfg.get("gamma", 0.5)
        )

    if name == "none":
        return None

    raise ValueError(f"Unknown scheduler '{name}'. Choose: cosine | step | none")


def _metric_is_better(current: float, best: float, name: str) -> bool:
    return current < best if name == "val_loss" else current > best

def _metric_init(name: str) -> float:
    return float("inf") if name == "val_loss" else float("-inf")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Full pass over `loader`; returns loss, auc, auprc, f1, accuracy."""
    model.eval()
    total_loss, n = 0.0, 0
    all_logits, all_labels = [], []

    for emb, lbl in loader:
        emb, lbl = emb.to(device), lbl.to(device)
        logits     = model(emb)
        loss       = criterion(logits, lbl)
        total_loss += loss.item() * len(lbl)
        n          += len(lbl)
        all_logits.append(logits.cpu())
        all_labels.append(lbl.cpu())

    logits_np = torch.cat(all_logits).float().numpy()
    labels_np = torch.cat(all_labels).float().numpy()
    probs_np  = 1.0 / (1.0 + np.exp(-logits_np))
    preds_np  = (probs_np >= threshold).astype(int)

    metrics: dict[str, float] = {"val_loss": total_loss / n}
    try:
        metrics["auc"]   = roc_auc_score(labels_np, probs_np)
    except ValueError:
        metrics["auc"]   = float("nan")
    try:
        metrics["auprc"] = average_precision_score(labels_np, probs_np)
    except ValueError:
        metrics["auprc"] = float("nan")
    metrics["accuracy"] = accuracy_score(labels_np, preds_np)
    metrics["f1"]       = f1_score(labels_np, preds_np, zero_division=0)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_start_banner(
    model, criterion, t_cfg: dict,
    n_train: int, n_val: int, n_test: int,
    device: torch.device,
) -> None:
    """Rich panel shown once at the start of training."""
    dev_str = str(device)
    if device.type == "cuda":
        dev_str = f"cuda  —  {torch.cuda.get_device_name(0)}"

    rows = [
        f"  [bold cyan]Model[/bold cyan]       AptamerClassifier   "
        f"[dim]{model.count_parameters():,} parameters[/dim]",

        f"  [bold cyan]Loss[/bold cyan]        {criterion}",

        f"  [bold cyan]Optimizer[/bold cyan]   {t_cfg.get('optimizer','adamw').upper()}  "
        f"lr={t_cfg['learning_rate']:.1e}  wd={t_cfg.get('weight_decay',1e-5):.1e}",

        f"  [bold cyan]Scheduler[/bold cyan]   {t_cfg.get('scheduler','cosine')}  "
        f"({t_cfg.get('warmup_epochs',5)} warmup epochs)",

        f"  [bold cyan]Data[/bold cyan]        "
        f"train [green]{n_train:,}[/green]  "
        f"val [green]{n_val:,}[/green]  "
        f"test [green]{n_test:,}[/green]",

        f"  [bold cyan]Device[/bold cyan]      {dev_str}",

        f"  [bold cyan]Epochs[/bold cyan]      up to {t_cfg['epochs']}  "
        f"[dim](early stop patience={t_cfg.get('early_stopping_patience',15)}, "
        f"metric={t_cfg.get('best_metric','auc')})[/dim]",
    ]
    console.print(
        Panel("\n".join(rows), title="[bold]AptaBind Training[/bold]",
              border_style="cyan", padding=(0, 1))
    )
    console.print()


def _epoch_line(
    epoch: int, epochs: int,
    train_loss: float,
    val: dict[str, float],
    lr: float,
    epoch_time: float,
    is_best: bool,
) -> None:
    """Single colourised summary line printed after each epoch."""
    c_metric = "bold green" if is_best else "white"
    best_tag  = "  [bold green]✓ best[/bold green]" if is_best else ""
    console.print(
        f"  Epoch [cyan]{epoch:3d}[/cyan]/{epochs}  "
        f"trn=[yellow]{train_loss:.4f}[/yellow]  "
        f"val=[yellow]{val['val_loss']:.4f}[/yellow]  "
        f"auc=[{c_metric}]{val['auc']:.4f}[/{c_metric}]  "
        f"auprc=[{c_metric}]{val['auprc']:.4f}[/{c_metric}]  "
        f"f1={val['f1']:.4f}  "
        f"acc={val['accuracy']:.4f}  "
        f"lr=[dim]{lr:.2e}[/dim]  "
        f"[dim]{epoch_time:.1f}s[/dim]"
        f"{best_tag}"
    )


def _print_history_table(history: list[dict]) -> None:
    """Rich table with one row per completed epoch, best row highlighted."""
    tbl = Table(
        title="Training History",
        box=box.ROUNDED,
        header_style="bold cyan",
        border_style="dim",
        show_lines=False,
        row_styles=["", "dim"],          # alternating row shading
    )
    tbl.add_column("Epoch",   justify="right",  style="cyan",   no_wrap=True)
    tbl.add_column("TrnLoss", justify="right")
    tbl.add_column("ValLoss", justify="right")
    tbl.add_column("AUC-ROC", justify="right")
    tbl.add_column("AUPRC",   justify="right")
    tbl.add_column("F1",      justify="right")
    tbl.add_column("Acc",     justify="right")
    tbl.add_column("LR",      justify="right", style="dim")
    tbl.add_column("Time",    justify="right", style="dim")

    for h in history:
        style  = "bold green on dark_green" if h["is_best"] else ""
        marker = " ✓" if h["is_best"] else ""
        tbl.add_row(
            f"{h['epoch']}{marker}",
            f"{h['train_loss']:.4f}",
            f"{h['val_loss']:.4f}",
            f"{h['auc']:.4f}",
            f"{h['auprc']:.4f}",
            f"{h['f1']:.4f}",
            f"{h['accuracy']:.4f}",
            f"{h['lr']:.2e}",
            f"{h['epoch_time']:.1f}s",
            style=style,
            end_section=(h is history[-1]),
        )

    console.print()
    console.print(tbl)


def _print_test_panel(metrics: dict[str, float], best_epoch: int, total_time: float) -> None:
    """Rich panel with final held-out test results."""
    rows_table = Table(box=None, show_header=False, pad_edge=False, padding=(0, 2))
    rows_table.add_column("Metric",  style="bold cyan",  min_width=12)
    rows_table.add_column("Value",   style="bold green",  justify="right", min_width=8)

    rows_table.add_row("AUC-ROC",  f"{metrics['auc']:.4f}")
    rows_table.add_row("AUPRC",    f"{metrics['auprc']:.4f}")
    rows_table.add_row("F1",       f"{metrics['f1']:.4f}")
    rows_table.add_row("Accuracy", f"{metrics['accuracy']:.4f}")
    rows_table.add_row("Loss",     f"{metrics['val_loss']:.4f}")
    rows_table.add_row("[dim]──────────────────[/dim]", "")
    rows_table.add_row("[dim]Best epoch[/dim]",  f"[dim]{best_epoch}[/dim]")
    rows_table.add_row("[dim]Total time[/dim]",  f"[dim]{total_time/60:.1f} min[/dim]")

    console.print(
        Panel(rows_table,
              title=f"[bold green]Test Results[/bold green]",
              border_style="green",
              padding=(0, 1))
    )


def _make_progress_bar() -> Progress:
    """Transient progress bar for the inner batch loop."""
    return Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=38, style="cyan", complete_style="bold cyan"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("[dim]{task.fields[stats]}[/dim]"),
        console=console,
        transient=True,       # bar disappears after the epoch ends
        refresh_per_second=8,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    t_cfg = cfg["training"]

    device = get_device()

    # ── Data ─────────────────────────────────────────────────────────────────
    log.info("Building dataloaders…")
    train_loader, val_loader, test_loader, label_map = get_dataloaders(config_path)
    cc = train_loader.dataset.class_counts
    log.info(
        f"Label map: {label_map}  |  "
        f"Train — pos: [green]{cc.get(1,0):,}[/green]  "
        f"neg: {cc.get(0,0):,}"
    )

    # ── Loss, model, optimizer, scheduler ────────────────────────────────────
    criterion = build_criterion(t_cfg, device)
    model     = AptamerClassifier.from_config(config_path).to(device)
    optimizer = build_optimizer(model, t_cfg)
    scheduler = build_scheduler(optimizer, t_cfg, steps_per_epoch=len(train_loader))

    epochs           = t_cfg["epochs"]
    patience         = t_cfg.get("early_stopping_patience", 15)
    grad_clip        = t_cfg.get("grad_clip", 1.0)
    best_metric_name = t_cfg.get("best_metric", "auc")

    _print_start_banner(
        model, criterion, t_cfg,
        n_train=len(train_loader.dataset),
        n_val=len(val_loader.dataset),
        n_test=len(test_loader.dataset),
        device=device,
    )

    # ── Checkpointing ────────────────────────────────────────────────────────
    ckpt_dir  = Path(t_cfg.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_model.pt"
    last_path = ckpt_dir / "last_model.pt"

    best_val   = _metric_init(best_metric_name)
    no_improve = 0
    best_epoch = 1
    history: list[dict] = []
    total_start = time.perf_counter()

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, n = 0.0, 0
        epoch_start   = time.perf_counter()

        # Inner batch loop with a transient rich progress bar
        with _make_progress_bar() as progress:
            task = progress.add_task(
                f"Epoch {epoch:3d}/{epochs}",
                total=len(train_loader),
                stats="",
            )
            for emb, lbl in train_loader:
                emb, lbl = emb.to(device), lbl.to(device)

                optimizer.zero_grad()
                logits = model(emb)
                loss   = criterion(logits, lbl)
                loss.backward()

                if grad_clip and grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                epoch_loss += loss.item() * len(lbl)
                n          += len(lbl)

                progress.update(
                    task, advance=1,
                    stats=(
                        f"loss={epoch_loss/n:.4f}  "
                        f"lr={optimizer.param_groups[0]['lr']:.2e}"
                    ),
                )

        train_loss = epoch_loss / n
        epoch_time = time.perf_counter() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Validation ───────────────────────────────────────────────────────
        val_metrics    = evaluate(model, val_loader, criterion, device)
        current_metric = val_metrics[best_metric_name]
        is_best        = _metric_is_better(current_metric, best_val, best_metric_name)

        if is_best:
            best_val   = current_metric
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "epoch":       epoch,
                    "state_dict":  model.state_dict(),
                    "optimizer":   optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "label_map":   label_map,
                    "config":      cfg,
                },
                best_path,
            )
        else:
            no_improve += 1

        # ── Permanent per-epoch summary line ─────────────────────────────────
        _epoch_line(epoch, epochs, train_loss, val_metrics, current_lr, epoch_time, is_best)

        history.append({
            "epoch":      epoch,
            "train_loss": train_loss,
            "is_best":    is_best,
            "epoch_time": epoch_time,
            "lr":         current_lr,
            **val_metrics,
        })

        if no_improve >= patience:
            console.print(
                f"\n[yellow]Early stopping: {best_metric_name} did not improve "
                f"for {patience} consecutive epochs.[/yellow]\n"
            )
            break

    # ── Save last checkpoint ──────────────────────────────────────────────────
    torch.save(
        {
            "epoch":      epoch,
            "state_dict": model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "label_map":  label_map,
            "config":     cfg,
        },
        last_path,
    )

    total_time = time.perf_counter() - total_start
    log.info(
        f"Training finished in [bold]{total_time/60:.1f} min[/bold]  |  "
        f"Best epoch: [cyan]{best_epoch}[/cyan]  |  "
        f"Checkpoints → [dim]{ckpt_dir}/[/dim]"
    )

    # ── History table ─────────────────────────────────────────────────────────
    _print_history_table(history)

    # ── Test evaluation ───────────────────────────────────────────────────────
    log.info("Loading best checkpoint for held-out test evaluation…")
    ckpt = torch.load(best_path, map_location=device,weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)
    _print_test_panel(test_metrics, best_epoch, total_time)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AptamerClassifier")
    p.add_argument("--config", "-c", default="config.yaml",
                   help="Path to config YAML (default: config.yaml)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args().config)
