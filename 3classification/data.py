"""
Shared data module: partition reading, oversampling, Dataset classes
"""
from __future__ import annotations

import os
import random
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ============================================================
#  Partition Reading
# ============================================================
def make_split(
    df: pd.DataFrame, id_col: str, label_col: str
) -> tuple[list[str], list[int]]:
    """Extract id and label lists from DataFrame"""
    subset = df[df[id_col].notna()]
    ids = subset[id_col].astype(str).tolist()
    labels = subset[label_col].astype(int).tolist()
    return ids, labels


def build_text_maps(
    text_excel: str,
) -> tuple[dict[str, str], dict[str, float], dict[str, int]]:
    """
    Build text_map, age_map, sex_map from Excel.

    Parameters
    ----------
    text_excel : Excel file path

    Returns
    -------
    text_map : {pid: immunofluorescence_report text}
    age_map  : {pid: age (float)}
    sex_map  : {pid: sex encoding (int, male=-1 female=1)}
    """
    df = pd.read_excel(text_excel)
    df["ID"] = df["ID"].astype(str)

    text_map = dict(zip(df["ID"], df["immunofluorescence_report"].astype(str)))
    sex_map = dict(
        zip(df["ID"], df["sex"].map({"male": -1, "female": 1}).astype(int))
    )
    age_map = dict(zip(df["ID"], df["age"].astype(float)))

    return text_map, age_map, sex_map


# ============================================================
#  Oversampling Balance
# ============================================================
def balance_by_oversample(
    ids: list[str],
    labels: list[int],
    extra_cols: dict[str, list] | None = None,
) -> tuple[list[str], list[int], dict[str, list]]:
    """
    Oversample minority classes so each class has the same count as the largest.

    Parameters
    ----------
    ids : sample ID list
    labels : label list
    extra_cols : additional columns to oversample in sync {col_name: list}

    Returns
    -------
    Balanced (ids, labels, extra_cols)
    """
    label_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    max_n = max(len(v) for v in label_to_indices.values())

    balanced_indices: list[int] = []
    for label, indices in label_to_indices.items():
        if not indices:
            continue
        times = (max_n // len(indices)) + 1
        repeated = (indices * times)[:max_n]
        balanced_indices.extend(repeated)

    random.shuffle(balanced_indices)

    balanced_ids = [ids[i] for i in balanced_indices]
    balanced_labels = [labels[i] for i in balanced_indices]
    balanced_extra = {}
    if extra_cols:
        for key, col in extra_cols.items():
            balanced_extra[key] = [col[i] for i in balanced_indices]

    dist = Counter(balanced_labels)
    print(f"[Oversample] Balanced class distribution: {dict(dist)}")
    return balanced_ids, balanced_labels, balanced_extra


# ============================================================
#  BERT Text Dataset
# ============================================================
class BertTextDataset(Dataset):
    """BERT text classification dataset (supports oversampling)"""

    def __init__(
        self,
        ids: list[str],
        labels: list[int],
        tokenizer: Any,
        text_map: dict[str, str],
        age_map: dict[str, float],
        sex_map: dict[str, int],
        max_len: int = 256,
        balance: bool = False,
    ):
        self.tokenizer = tokenizer
        self.text_map = text_map
        self.age_map = age_map
        self.sex_map = sex_map
        self.max_len = max_len

        if balance:
            ids, labels, _ = balance_by_oversample(ids, labels)

        self.ids = ids
        self.labels = labels

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pid = str(self.ids[idx])
        label = self.labels[idx]

        text = self.text_map[pid]
        age = self.age_map[pid]
        sex = self.sex_map[pid]

        tokenized = self.tokenizer(
            str(text),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "pid": pid,
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "age": torch.tensor([float(age)], dtype=torch.float),
            "sex": torch.tensor([float(sex)], dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.long),
        }


# ============================================================
#  GAAN Image Feature Dataset
# ============================================================
class ImageFeatureDataset(Dataset):
    """GAAN image classification dataset (loads pre-extracted features from .pt files)"""

    def __init__(
        self,
        csv_path: str,
        pt_dir: str,
        split: str = "train",
        balance: bool = False,
    ):
        df = pd.read_csv(csv_path)

        if split == "train":
            valid = df[["train", "train_label"]].dropna()
            self.ids = valid["train"].astype(str).tolist()
            self.labels = valid["train_label"].astype(int).tolist()
        elif split == "val":
            valid = df[["val", "val_label"]].dropna()
            self.ids = valid["val"].astype(str).tolist()
            self.labels = valid["val_label"].astype(int).tolist()
        else:
            raise ValueError(f"Unknown split: {split}")

        self.pt_dir = pt_dir

        if balance and split == "train":
            self._balance()

    def _balance(self) -> None:
        balanced_ids, balanced_labels, _ = balance_by_oversample(self.ids, self.labels)
        self.ids = balanced_ids
        self.labels = balanced_labels

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
    """GAAN DataLoader collate_fn (variable-length bags)"""
    Xs, ys = zip(*batch)
    return list(Xs), torch.tensor(ys, dtype=torch.long)


# ============================================================
#  Fusion Dataset (text + image)
# ============================================================
class FusionDataset(Dataset):
    """Fusion model dataset: loads both text and image features"""

    def __init__(
        self,
        ids: list[str],
        labels: list[int],
        text_map: dict[str, str],
        age_map: dict[str, float],
        sex_map: dict[str, int],
        img_dir: str,
        tokenizer: Any,
        max_len: int = 256,
        balance: bool = False,
    ):
        self.ids = ids
        self.labels = labels
        self.text_map = text_map
        self.age_map = age_map
        self.sex_map = sex_map
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.texts = [text_map[str(id_val)] for id_val in ids]
        self.ages = [age_map[str(id_val)] for id_val in ids]
        self.sexes = [sex_map[str(id_val)] for id_val in ids]

        if balance:
            self._balance()

    def _balance(self) -> None:
        label_to_samples: dict[int, list] = defaultdict(list)
        for idx, label in enumerate(self.labels):
            sample = (self.ids[idx], self.texts[idx], self.ages[idx], self.sexes[idx], label)
            label_to_samples[label].append(sample)

        max_samples = max(len(v) for v in label_to_samples.values())
        balanced: list = []
        for label, samples in label_to_samples.items():
            repeat = (max_samples // len(samples)) + 1
            balanced.extend((samples * repeat)[:max_samples])

        self.ids, self.texts, self.ages, self.sexes, self.labels = zip(*balanced)
        self.ids = list(self.ids)
        self.texts = list(self.texts)
        self.ages = list(self.ages)
        self.sexes = list(self.sexes)
        self.labels = list(self.labels)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[tuple, torch.Tensor, torch.Tensor]:
        pid = str(self.ids[idx])
        label = self.labels[idx]
        text = self.text_map[pid]

        # Text encoding
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        age = torch.tensor(self.age_map[pid], dtype=torch.float)
        sex = torch.tensor(self.sex_map[pid], dtype=torch.float)

        text_inputs = (input_ids, attention_mask, age, sex)

        # Image features
        feat_path = os.path.join(self.img_dir, f"{self.ids[idx]}.pt")
        img_feat = torch.load(feat_path, map_location="cpu", weights_only=True)
        if isinstance(img_feat, dict) and "tokens" in img_feat:
            img_feat = img_feat["tokens"]

        return text_inputs, img_feat, torch.tensor(label, dtype=torch.long)
