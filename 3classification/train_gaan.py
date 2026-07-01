"""
GAAN image classification training entry
Usage: python -m renal_cls.train_gaan [--config path/to/config.yaml]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from .config import load_config, get_device
from .data import ImageFeatureDataset, image_collate_fn
from .models import GaanClassifier
from .utils import (
    set_seed, seed_worker, FocalLoss,
    plot_multiclass_roc, evaluate,
)


def train(cfg: dict) -> None:
    common = cfg["common"]
    data_cfg = cfg["data"]
    task_cfg = cfg["gaan"]

    set_seed(common["seed"])
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    device = get_device(cfg)
    num_classes = common["num_labels"]
    class_names = common["class_names"]

    save_dir = task_cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # ---- Data ----
    csv_path = data_cfg["partition_csv"]
    pt_dir = data_cfg["image_feat_dir"]

    train_ds = ImageFeatureDataset(csv_path, pt_dir, split="train", balance=True)
    val_ds = ImageFeatureDataset(csv_path, pt_dir, split="val", balance=False)

    g = torch.Generator()
    g.manual_seed(common["seed"])

    train_loader = DataLoader(
        train_ds, batch_size=task_cfg["batch_size"],
        sampler=RandomSampler(train_ds),
        collate_fn=image_collate_fn,
        worker_init_fn=seed_worker, generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=task_cfg["batch_size"],
        collate_fn=image_collate_fn,
        worker_init_fn=seed_worker, generator=g,
    )

    # ---- Model ----
    sample_x, _ = next(iter(train_ds))
    feature_dim = sample_x.shape[-1]

    model = GaanClassifier(feature_dim=feature_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=task_cfg["lr"])

    # ---- Training Loop ----
    best_val_acc = 0.0
    log_path = os.path.join(save_dir, "training_log.txt")

    for epoch in range(1, task_cfg["epochs"] + 1):
        # Train
        model.train()
        total_loss = total_correct = total_samples = 0

        for Xs, ys in tqdm(train_loader, desc=f"Epoch {epoch}/{task_cfg['epochs']}", leave=False):
            if ys.dim() == 0:
                ys = ys.unsqueeze(0)
            ys = ys.to(device)

            logits = model(Xs)
            loss = criterion(logits, ys)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(Xs)
            total_correct += (logits.argmax(dim=1) == ys).sum().item()
            total_samples += len(Xs)

        train_loss = total_loss / max(total_samples, 1)
        train_acc = total_correct / max(total_samples, 1)

        # Validate
        result = evaluate(
            model, val_loader, device, num_classes,
            save_csv=os.path.join(save_dir, f"val_epoch{epoch}_predictions.csv"),
            dataset=val_ds,
        )

        # Compute val_loss
        model.eval()
        val_loss_sum = val_count = 0
        with torch.no_grad():
            for Xs, ys in val_loader:
                if ys.dim() == 0:
                    ys = ys.unsqueeze(0)
                ys = ys.to(device)
                logits = model(Xs)
                val_loss_sum += F.cross_entropy(logits, ys, reduction="sum").item()
                val_count += len(ys)
        val_loss = val_loss_sum / max(val_count, 1)

        # ROC
        roc_path = os.path.join(save_dir, f"val_epoch{epoch}_roc.png")
        plot_multiclass_roc(result["labels"], result["probs"], class_names, roc_path)

        print(
            f"Epoch {epoch}/{task_cfg['epochs']}  "
            f"Train Loss={train_loss:.4f}  Train Acc={train_acc:.4f} | "
            f"Val Loss={val_loss:.4f}  Val Acc={result['acc']:.4f}  AUC={result['auc']:.4f}  F1={result['f1']:.4f}"
        )
        print(f"Confusion Matrix:\n{result['cm']}")

        # Write to unified log file
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{'='*60}\n")
            f.write(f"Epoch {epoch}/{task_cfg['epochs']}\n")
            f.write(f"Train Loss={train_loss:.4f}  Train Acc={train_acc:.4f} | "
                    f"Val Loss={val_loss:.4f}  Val Acc={result['acc']:.4f}  AUC={result['auc']:.4f}  F1={result['f1']:.4f}\n")
            f.write(f"Confusion Matrix:\n{result['cm']}\n\n")

        # Save model
        epoch_path = os.path.join(save_dir, f"epoch_{epoch}_model.pt")
        torch.save(model.state_dict(), epoch_path)

        if result["acc"] > best_val_acc:
            best_val_acc = result["acc"]
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
            print(f"  >> Saved best model (val_acc={best_val_acc:.4f})")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"  >> Best model saved (val_acc={best_val_acc:.4f})\n")

    print(f"\nTraining finished. Best Val Acc = {best_val_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="GAAN image classification training")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
