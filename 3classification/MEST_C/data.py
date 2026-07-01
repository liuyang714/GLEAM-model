"""
MEST-C shared data module: parameterized Dataset for all five components
"""
from __future__ import annotations

import os
from collections import Counter, defaultdict

import pandas as pd
import torch
from torch.utils.data import Dataset


# Task -> (label_col_train, label_col_val, num_classes)
TASK_CONFIG = {
    "M": ("train_label_m", "val_label_m", 2),
    "E": ("train_label_e", "val_label_e", 2),
    "S": ("train_label_s", "val_label_s", 2),
    "T": ("train_label_t", "val_label_t", 3),
    "C": ("train_label_c", "val_label_c", 3),
}


class MESTCDataset(Dataset):
    """
    Unified dataset for MEST-C components.

    Parameters
    ----------
    csv_file : path to partition CSV
    pt_dir : directory of .pt feature files
    task : one of "M", "E", "S", "T", "C"
    split : "train" or "val"
    """

    def __init__(self, csv_file: str, pt_dir: str, task: str, split: str = "train"):
        if task not in TASK_CONFIG:
            raise ValueError(f"Unknown task '{task}'. Must be one of {list(TASK_CONFIG.keys())}")

        label_col_train, label_col_val, self.num_classes = TASK_CONFIG[task]
        self.task = task

        df = pd.read_csv(csv_file)

        if split == "train":
            valid = df[["train", label_col_train]].dropna()
            self.ids = valid["train"].astype(str).tolist()
            self.labels = valid[label_col_train].astype(int).tolist()
        elif split == "val":
            valid = df[["val", label_col_val]].dropna()
            self.ids = valid["val"].astype(str).tolist()
            self.labels = valid[label_col_val].astype(int).tolist()
        else:
            raise ValueError(f"Unknown split: {split}")

        self.pt_dir = pt_dir

        dist = Counter(self.labels)
        print(f"[{task}] {split}: {len(self.ids)} samples, distribution: {dict(dist)}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_id = self.ids[idx]
        pt_path = os.path.join(self.pt_dir, f"{sample_id}.pt")
        bag = torch.load(pt_path, weights_only=True)

        if isinstance(bag, dict) and "tokens" in bag:
            bag = bag["tokens"]

        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return bag, y


def image_collate_fn(batch: list) -> tuple[list, torch.Tensor]:
    """Collate function for variable-length bags"""
    Xs, ys = zip(*batch)
    return list(Xs), torch.tensor(ys, dtype=torch.long)
