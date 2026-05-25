"""
train.py
--------
Training loop for AptamerClassifier.

Usage:
    python train.py
    python train.py --config config.yaml

All hyperparameters are read from config.yaml.

Outputs:
    checkpoints/best_model.pt   — checkpoint with the best validation metric
    checkpoints/last_model.pt   — checkpoint after the final epoch
"""

import argparse
import logging
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
)
import yaml

from dataset import get_dataloaders
from model import AptamerClassifier, build_criterion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
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
        log.warning("No GPU found — using CPU. Training may be slow.")
    return d


def build_optimizer(model: nn.Module, t_cfg: dict) -> torch.optim.Optimizer:
    name = t_cfg.get("optimizer", "adamw").lower()
    lr   = t_cfg["learning_rate"]
    wd   = t_cfg.get("weight_decay", 1e-5)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"Unknown optimizer '{name}'. Choose: adamw | adam | sgd")


def build_scheduler(
    optimizer,
    t_cfg: dict,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Build a per-step LR scheduler."""
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
        gamma     = t_cfg.get("gamma", 0.5)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    if name == "none":
        return None

    raise ValueError(f"Unknown scheduler '{name}'. Choose: cosine | step | none")




# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:     AptamerClassifier,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Run model on `loader` and return a dict of metrics:
        loss, accuracy, auc, auprc, f1
    """
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
    probs_np  = 1.0 / (1.0 + np.exp(-logits_np))   # sigmoid without torch overhead
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


def _metric_is_better(current: float, best: float, metric_name: str) -> bool:
    """Higher is better for all tracked metrics except val_loss."""
    if metric_name == "val_loss":
        return current < best
    return current > best


def _metric_init_value(metric_name: str) -> float:
    return float("inf") if metric_name == "val_loss" else float("-inf")


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    t_cfg = cfg["training"]

    device = get_device()

    # ── Dataloaders ──────────────────────────────────────────────────────────
    log.info("Building dataloaders...")
    train_loader, val_loader, test_loader, label_map = get_dataloaders(config_path)
    log.info(f"Label map: {label_map}")
    log.info(f"Train class counts: {train_loader.dataset.class_counts}")

    # ── Loss ─────────────────────────────────────────────────────────────────
    criterion = build_criterion(t_cfg, device)

    # ── Model ────────────────────────────────────────────────────────────────
    model = AptamerClassifier.from_config(config_path).to(device)
    log.info(f"\n{model.summary()}\n")

    # ── Optimizer & LR scheduler ─────────────────────────────────────────────
    optimizer = build_optimizer(model, t_cfg)
    scheduler = build_scheduler(optimizer, t_cfg, steps_per_epoch=len(train_loader))

    # ── Checkpointing setup ───────────────────────────────────────────────────
    ckpt_dir    = Path(t_cfg.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path   = ckpt_dir / "best_model.pt"
    last_path   = ckpt_dir / "last_model.pt"

    best_metric_name = t_cfg.get("best_metric", "auc")
    best_val         = _metric_init_value(best_metric_name)
    patience         = t_cfg.get("early_stopping_patience", 15)
    no_improve       = 0
    log_every        = t_cfg.get("log_every_n_batches", 100)

    log.info(
        f"Training for up to {t_cfg['epochs']} epochs  "
        f"(early stop after {patience} epochs without {best_metric_name} improvement)"
    )

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(1, t_cfg["epochs"] + 1):
        model.train()
        epoch_loss, n = 0.0, 0

        for batch_idx, (emb, lbl) in enumerate(train_loader, 1):
            emb, lbl = emb.to(device), lbl.to(device)

            optimizer.zero_grad()
            logits = model(emb)
            loss   = criterion(logits, lbl)
            loss.backward()

            gc = t_cfg.get("grad_clip", 1.0)
            if gc and gc > 0:
                nn.utils.clip_grad_norm_(model.parameters(), gc)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item() * len(lbl)
            n          += len(lbl)

            if batch_idx % log_every == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                log.info(
                    f"  Epoch {epoch}/{t_cfg['epochs']}  "
                    f"batch {batch_idx}/{len(train_loader)}  "
                    f"lr={lr_now:.2e}  "
                    f"train_loss={epoch_loss / n:.4f}"
                )

        train_loss = epoch_loss / n

        # ── Validation ───────────────────────────────────────────────────────
        val_metrics = evaluate(model, val_loader, criterion, device)
        log.info(
            f"Epoch {epoch:3d}/{t_cfg['epochs']}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_metrics['val_loss']:.4f}  "
            f"auc={val_metrics['auc']:.4f}  "
            f"auprc={val_metrics['auprc']:.4f}  "
            f"f1={val_metrics['f1']:.4f}  "
            f"acc={val_metrics['accuracy']:.4f}"
        )

        # ── Best model tracking & early stopping ──────────────────────────────
        current_metric = val_metrics[best_metric_name]
        if _metric_is_better(current_metric, best_val, best_metric_name):
            best_val   = current_metric
            no_improve = 0
            torch.save(
                {
                    "epoch":      epoch,
                    "state_dict": model.state_dict(),
                    "optimizer":  optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "label_map":  label_map,
                    "config":     cfg,
                },
                best_path,
            )
            log.info(
                f"  ✓ New best {best_metric_name}={current_metric:.4f} "
                f"— checkpoint saved → {best_path}"
            )
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(
                    f"Early stopping: no improvement in {best_metric_name} "
                    f"for {patience} epochs."
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
    log.info(f"Last checkpoint saved → {last_path}")

    # ── Final test evaluation ─────────────────────────────────────────────────
    log.info("\nLoading best checkpoint for held-out test evaluation...")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)

    log.info(
        f"\n{'='*60}\n"
        f"  TEST RESULTS (best epoch {ckpt['epoch']})\n"
        f"{'─'*60}\n"
        f"  loss  : {test_metrics['val_loss']:.4f}\n"
        f"  AUC   : {test_metrics['auc']:.4f}\n"
        f"  AUPRC : {test_metrics['auprc']:.4f}\n"
        f"  F1    : {test_metrics['f1']:.4f}\n"
        f"  Acc   : {test_metrics['accuracy']:.4f}\n"
        f"{'='*60}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AptamerClassifier")
    p.add_argument(
        "--config", "-c", default="config.yaml",
        help="Path to config YAML (default: config.yaml)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config)
