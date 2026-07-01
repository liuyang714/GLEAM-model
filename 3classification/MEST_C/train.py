"""
MEST-C component training entry
Usage: python -m renal_cls.mestc.train --task M --config path/to/config.yaml
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from ..config import load_config, get_device
from ..models import GaanClassifier
from ..utils import set_seed, seed_worker, plot_multiclass_roc, plot_confusion_matrix
from .data import MESTCDataset, image_collate_fn, TASK_CONFIG


def _plot_binary_roc(all_labels, all_probs, task_name, save_path):
    """Binary ROC: single curve using positive class probability"""
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    pos_probs = all_probs[:, 1]

    fpr, tpr, _ = roc_curve(all_labels, pos_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{task_name} Component - ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def train(cfg: dict, task: str) -> None:
    common = cfg["common"]
    data_cfg = cfg["data"]
    task_cfg = cfg["mestc"]

    set_seed(common["seed"])
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    device = get_device(cfg)

    _, _, num_classes = TASK_CONFIG[task]
    save_dir = os.path.join(task_cfg["save_dir"], task)
    os.makedirs(save_dir, exist_ok=True)

    # ---- Data ----
    csv_path = data_cfg["partition_csv"]
    pt_dir = data_cfg["image_feat_dir"]

    train_ds = MESTCDataset(csv_path, pt_dir, task=task, split="train")
    val_ds = MESTCDataset(csv_path, pt_dir, task=task, split="val")

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

    # Compute class weights from training data distribution
    from collections import Counter
    train_label_counts = Counter(train_ds.labels)
    total = sum(train_label_counts.values())
    class_weights = torch.tensor(
        [total / train_label_counts[i] for i in range(num_classes)],
        dtype=torch.float, device=device,
    )
    class_weights = class_weights / class_weights.sum() * num_classes  # normalize
    print(f"[{task}] Class weights: {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=task_cfg["lr"])

    # ---- Training Loop ----
    best_val_acc = 0.0
    log_path = os.path.join(save_dir, "training_log.txt")

    for epoch in range(1, task_cfg["epochs"] + 1):
        # Train
        model.train()
        total_loss = total_correct = total_samples = 0

        for Xs, ys in tqdm(train_loader, desc=f"[{task}] Epoch {epoch}/{task_cfg['epochs']}", leave=False):
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

        # Validate — use argmax (NOT threshold)
        model.eval()
        correct = total = 0
        val_loss_sum = val_count = 0
        all_labels, all_preds, all_probs = [], [], []

        with torch.no_grad():
            for Xs, ys in val_loader:
                if ys.dim() == 0:
                    ys = ys.unsqueeze(0)
                ys = ys.to(device)

                logits = model(Xs)
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)

                val_loss_sum += F.cross_entropy(logits, ys, reduction="sum").item()
                val_count += len(ys)
                correct += (preds == ys).sum().item()
                total += len(Xs)

                all_labels.extend(ys.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())

        val_acc = correct / max(total, 1)
        val_loss = val_loss_sum / max(val_count, 1)

        # Confusion matrix — argmax based
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

        print(
            f"[{task}] Epoch {epoch}/{task_cfg['epochs']}  "
            f"Train Loss={train_loss:.4f}  Train Acc={train_acc:.4f} | "
            f"Val Loss={val_loss:.4f}  Val Acc={val_acc:.4f}"
        )
        print(f"Confusion Matrix:\n{cm}")
        report = classification_report(all_labels, all_preds, digits=4, zero_division=0)
        print(report)

        # Write to unified log file
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{'='*60}\n")
            f.write(f"Epoch {epoch}/{task_cfg['epochs']}\n")
            f.write(f"Train Loss={train_loss:.4f}  Train Acc={train_acc:.4f} | "
                    f"Val Loss={val_loss:.4f}  Val Acc={val_acc:.4f}\n")
            f.write(f"Confusion Matrix:\n{cm}\n")
            f.write(f"{report}\n")

        # Save model
        torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch}.pt"))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
            print(f"  >> Saved best model (val_acc={best_val_acc:.4f})")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"  >> Best model saved (val_acc={best_val_acc:.4f})\n")

        # ROC
        roc_path = os.path.join(save_dir, f"roc_epoch{epoch}.png")
        if num_classes == 2:
            # Binary: single ROC curve
            _plot_binary_roc(all_labels, all_probs, task, roc_path)
        else:
            # Multiclass: per-class ROC curves
            class_names = [f"{task}{i}" for i in range(num_classes)]
            plot_multiclass_roc(all_labels, all_probs, class_names, roc_path)

        # Confusion Matrix
        cm_path = os.path.join(save_dir, f"cm_epoch{epoch}.png")
        class_names = [f"{task}{i}" for i in range(num_classes)]
        plot_confusion_matrix(all_labels, all_preds, class_names, cm_path)

    print(f"\n[{task}] Training finished. Best Val Acc = {best_val_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="MEST-C component training")
    parser.add_argument("--task", type=str, required=True, choices=["M", "E", "S", "T", "C"],
                        help="Which component to train")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, args.task)


if __name__ == "__main__":
    main()
