"""
dataset.py
----------
AptamerDataset — loads the nested pickle produced by gen_embeddings.py
and returns ready-to-use PyTorch DataLoaders.

Pickle format (produced by gen_embeddings.py):
    {
        seq_id: {
            "sequence":         str,
            "embedding":        np.ndarray  shape (embed_dim,),
            "label":            str,          e.g. "1" or "0"
            "enrichment_score": str,
        },
        ...
    }

Public API:
    get_dataloaders(config_path) → (train_loader, val_loader, test_loader, label_map)
    load_config(config_path)     → dict
"""

import pickle
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import yaml

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class AptamerDataset(Dataset):
    """
    Wraps a list of aptamer records into a PyTorch Dataset.

    Parameters
    ----------
    records   : list[dict]
        Each dict must have keys 'embedding' (np.ndarray) and 'label' (str).
    label_map : dict[str, int]
        Maps raw label strings to integer class indices (0 or 1).
    """

    def __init__(self, records: list[dict], label_map: dict[str, int]) -> None:
        self.records   = records
        self.label_map = label_map

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec       = self.records[idx]
        embedding = torch.tensor(rec["embedding"], dtype=torch.float32)
        label     = torch.tensor(self.label_map[rec["label"]], dtype=torch.float32)
        return embedding, label

    @property
    def labels(self) -> list[int]:
        """Integer label for every record — useful for pos_weight calculation."""
        return [self.label_map[r["label"]] for r in self.records]

    @property
    def class_counts(self) -> dict[int, int]:
        """Count of each integer class in this split."""
        counts: dict[int, int] = {}
        for lbl in self.labels:
            counts[lbl] = counts.get(lbl, 0) + 1
        return counts


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Load and return a YAML config as a plain dict."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def _build_label_map(records: list[dict], positive_label: str) -> dict[str, int]:
    """
    Build a {label_str: int} map for binary classification.
    The label matching `positive_label` → 1; everything else → 0.
    Warns if the number of unique labels is not exactly 2.
    """
    unique = sorted({r["label"] for r in records})
    if len(unique) != 2:
        log.warning(
            f"Expected exactly 2 unique labels, found {len(unique)}: {unique}. "
            f"'{positive_label}' → 1, all others → 0."
        )
    label_map = {lbl: (1 if lbl == positive_label else 0) for lbl in unique}
    log.info(f"Label map: {label_map}")
    return label_map


def _stratified_split(
    records:     list[dict],
    int_labels:  list[int],
    train_ratio: float,
    val_ratio:   float,
    test_ratio:  float,
    stratify:    bool,
    seed:        int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split records into (train, val, test) subsets.
    Ratios must sum to 1.0 ± 1e-6.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train/val/test ratios must sum to 1.0, got {total:.6f}")

    indices        = list(range(len(records)))
    strat_arg      = int_labels if stratify else None

    # First split: train vs (val + test)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size   = 1.0 - train_ratio,
        stratify    = strat_arg,
        random_state= seed,
    )

    # Second split: val vs test (from the temp portion)
    temp_labels = [int_labels[i] for i in temp_idx] if stratify else None
    val_frac    = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size    = 1.0 - val_frac,
        stratify     = temp_labels,
        random_state = seed,
    )

    return (
        [records[i] for i in train_idx],
        [records[i] for i in val_idx],
        [records[i] for i in test_idx],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    config_path: str = "config.yaml",
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    """
    Build and return (train_loader, val_loader, test_loader, label_map).

    All split / loader parameters are read from `config_path`.

    Returns
    -------
    train_loader : DataLoader  — shuffled training batches
    val_loader   : DataLoader  — ordered validation batches
    test_loader  : DataLoader  — ordered test batches
    label_map    : dict[str, int]  — maps raw label strings to 0 / 1
    """
    cfg    = load_config(config_path)
    d_cfg  = cfg["data"]
    dl_cfg = cfg["dataloader"]

    # ── Load pickle ──────────────────────────────────────────────────────────
    pickle_path = Path(d_cfg["pickle_path"])
    if not pickle_path.exists():
        raise FileNotFoundError(
            f"Embedding pickle not found: {pickle_path}\n"
            f"Run gen_embeddings.py first to generate it."
        )

    with open(pickle_path, "rb") as f:
        raw: dict = pickle.load(f)

    records = [{"seq_id": sid, **entry} for sid, entry in raw.items()]
    log.info(f"Loaded {len(records):,} records from {pickle_path}")

    # ── Label encoding ───────────────────────────────────────────────────────
    label_map  = _build_label_map(records, d_cfg["positive_label"])
    int_labels = [label_map[r["label"]] for r in records]

    n_pos = sum(int_labels)
    n_neg = len(int_labels) - n_pos
    log.info(
        f"Class distribution — positive: {n_pos:,}  negative: {n_neg:,}  "
        f"pos_ratio: {n_pos / len(int_labels):.4f}  "
        f"imbalance_ratio: {n_neg / max(n_pos, 1):.1f}:1"
    )

    # ── Train / val / test split ─────────────────────────────────────────────
    train_recs, val_recs, test_recs = _stratified_split(
        records,
        int_labels,
        train_ratio = d_cfg.get("train_ratio", 0.80),
        val_ratio   = d_cfg.get("val_ratio",   0.10),
        test_ratio  = d_cfg.get("test_ratio",  0.10),
        stratify    = d_cfg.get("stratify",    True),
        seed        = d_cfg.get("seed",        42),
    )
    log.info(
        f"Split → train: {len(train_recs):,}  "
        f"val: {len(val_recs):,}  "
        f"test: {len(test_recs):,}"
    )

    # ── Build Datasets ───────────────────────────────────────────────────────
    train_ds = AptamerDataset(train_recs, label_map)
    val_ds   = AptamerDataset(val_recs,   label_map)
    test_ds  = AptamerDataset(test_recs,  label_map)

    # ── Build DataLoaders ────────────────────────────────────────────────────
    common_kwargs = dict(
        batch_size  = dl_cfg["batch_size"],
        num_workers = dl_cfg.get("num_workers", 4),
        pin_memory  = dl_cfg.get("pin_memory",  True),
    )
    train_loader = DataLoader(train_ds, shuffle=dl_cfg.get("shuffle_train", True),  **common_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **common_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **common_kwargs)

    return train_loader, val_loader, test_loader, label_map
