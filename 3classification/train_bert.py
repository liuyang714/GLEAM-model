"""
BERT text classification training entry
Usage: python -m renal_cls.train_bert [--config path/to/config.yaml]
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import BertTokenizer

from .config import load_config, get_device
from .data import make_split, build_text_maps, BertTextDataset
from .models import BertClassifier
from .utils import set_seed, FocalLoss, plot_multiclass_roc, plot_confusion_matrix, evaluate


def train(cfg: dict) -> None:
    # ---- Parse config ----
    common = cfg["common"]
    data_cfg = cfg["data"]
    model_cfg = cfg["models"]
    task_cfg = cfg["bert"]

    set_seed(common["seed"])
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    device = get_device(cfg)
    class_names = common["class_names"]
    num_labels = common["num_labels"]

    save_dir = task_cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # ---- Data ----
    import pandas as pd

    df_part = pd.read_csv(data_cfg["partition_csv"])
    train_ids, train_labels = make_split(df_part, "train", "train_label")
    val_ids, val_labels = make_split(df_part, "val", "val_label")
    text_map, age_map, sex_map = build_text_maps(data_cfg["text_excel"])

    print(f"Train: {len(train_ids)} samples | Val: {len(val_ids)} samples")
    print(f"Original train distribution: {dict(__import__('collections').Counter(train_labels))}")

    tokenizer = BertTokenizer.from_pretrained(model_cfg["pretrained_bert"])

    train_ds = BertTextDataset(
        train_ids, train_labels, tokenizer, text_map, age_map, sex_map,
        max_len=task_cfg["max_len"], balance=task_cfg["use_oversampling"],
    )
    val_ds = BertTextDataset(
        val_ids, val_labels, tokenizer, text_map, age_map, sex_map,
        max_len=task_cfg["max_len"], balance=False,
    )

    # Sampler
    if task_cfg.get("use_weighted_sampler") and not task_cfg["use_oversampling"]:
        from collections import Counter
        counts = Counter(train_labels)
        total = sum(counts.values())
        weights = [total / counts[l] for l in train_labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=task_cfg["batch_size"],
            sampler=sampler, num_workers=task_cfg["num_workers"],
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=task_cfg["batch_size"],
            shuffle=not task_cfg["use_oversampling"],
            num_workers=task_cfg["num_workers"],
        )

    val_loader = DataLoader(
        val_ds, batch_size=task_cfg["batch_size"],
        shuffle=False, num_workers=task_cfg["num_workers"],
    )

    # ---- Model ----
    model = BertClassifier(model_cfg["pretrained_bert"], num_labels).to(device)

    if task_cfg.get("use_focal"):
        criterion = FocalLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=task_cfg["lr"])
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=task_cfg["epochs"], eta_min=task_cfg["lr"] / 10,
    )

    # ---- Training Loop ----
    best_val_auc = -1.0
    log_path = os.path.join(save_dir, "training_log.txt")

    for epoch in range(1, task_cfg["epochs"] + 1):
        model.train()
        total_loss = total_correct = total_samples = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{task_cfg['epochs']}", leave=False):
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items() if k not in ("label", "pid")}
            labels = batch["label"].to(device)

            logits = model(**inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)

        train_loss = total_loss / max(total_samples, 1)
        train_acc = total_correct / max(total_samples, 1)

        # Validate
        val_xlsx = os.path.join(save_dir, f"epoch{epoch}_val_preds.xlsx")
        result = evaluate(model, val_loader, device, num_labels, class_names, save_csv=val_xlsx)

        # Compute val_loss
        model.eval()
        val_loss_sum = val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k not in ("label", "pid")}
                labels = batch["label"].to(device)
                logits = model(**inputs)
                val_loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
                val_count += labels.size(0)
        val_loss = val_loss_sum / max(val_count, 1)

        print(
            f"Epoch {epoch}/{task_cfg['epochs']}  "
            f"Train Loss={train_loss:.4f}  Train Acc={train_acc:.4f} | "
            f"Val Loss={val_loss:.4f}  Val Acc={result['acc']:.4f}  AUC={result['auc']:.4f}  F1={result['f1']:.4f}"
        )
        print(f"Confusion Matrix:\n{result['cm']}")

        # Classification report
        from sklearn.metrics import classification_report
        report = classification_report(
            result["labels"], result["preds"],
            digits=4, zero_division=0, labels=list(range(num_labels)),
        )
        print(report)

        # Write to unified log file
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{'='*60}\n")
            f.write(f"Epoch {epoch}/{task_cfg['epochs']}\n")
            f.write(f"Train Loss={train_loss:.4f}  Train Acc={train_acc:.4f} | "
                    f"Val Loss={val_loss:.4f}  Val Acc={result['acc']:.4f}  AUC={result['auc']:.4f}  F1={result['f1']:.4f}\n")
            f.write(f"Confusion Matrix:\n{result['cm']}\n")
            f.write(f"{report}\n")

        # ROC curve
        roc_path = os.path.join(save_dir, f"roc_epoch{epoch}_val.png")
        try:
            plot_multiclass_roc(result["labels"], result["probs"], class_names, roc_path)
        except Exception as e:
            print(f"ROC plotting failed: {e}")

        # Confusion Matrix
        cm_path = os.path.join(save_dir, f"cm_epoch{epoch}_val.png")
        try:
            plot_confusion_matrix(result["labels"], result["preds"], class_names, cm_path)
        except Exception as e:
            print(f"CM plotting failed: {e}")

        # Save best model
        if not (result["auc"] != result["auc"]):  # not nan
            if result["auc"] > best_val_auc:
                best_val_auc = result["auc"]
                torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))
                print(f"  >> Saved best model (val_auc={best_val_auc:.4f})")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"  >> Best model saved (val_auc={best_val_auc:.4f})\n")

        torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch}.pt"))
        scheduler.step()

    # ---- Final summary.xlsx with bootstrap CI on best validation results ----
    print(f"\nTraining finished. Best Val AUC = {best_val_auc:.4f}")
    print("Generating final summary.xlsx (bootstrap 1000)...")
    best_result = evaluate(model, val_loader, device, num_labels, class_names)
    save_summary_xlsx(
        os.path.join(save_dir, "summary.xlsx"),
        class_names, best_result["labels"], best_result["preds"],
        np.array(best_result["probs"]),
        title_text="BERT Six-Class Training Results",
    )


def main():
    parser = argparse.ArgumentParser(description="BERT text classification training")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
