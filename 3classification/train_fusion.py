"""
Fusion MLP training entry (BERT text + GAAN image -> MLP classification)
Usage: python -m renal_cls.train_fusion [--config path/to/config.yaml]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

from .config import load_config, get_device
from .data import make_split, build_text_maps, FusionDataset
from .models import BertClassifier, GaanClassifier, MLPFeatureFusion
from .utils import set_seed, plot_multiclass_roc


def train(cfg: dict) -> None:
    common = cfg["common"]
    data_cfg = cfg["data"]
    model_cfg = cfg["models"]
    task_cfg = cfg["fusion"]

    set_seed(common["seed"])
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    device = get_device(cfg)
    num_classes = common["num_labels"]
    class_names = common["class_names"]

    save_dir = task_cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # ---- Data ----
    df_part = pd.read_csv(data_cfg["partition_csv"])
    train_ids, train_labels = make_split(df_part, "train", "train_label")
    val_ids, val_labels = make_split(df_part, "val", "val_label")
    text_map, age_map, sex_map = build_text_maps(data_cfg["text_excel"])

    tokenizer = BertTokenizer.from_pretrained(model_cfg["pretrained_bert"])

    train_ds = FusionDataset(
        train_ids, train_labels, text_map, age_map, sex_map,
        data_cfg["image_feat_dir"], tokenizer,
        max_len=task_cfg["max_len"], balance=True,
    )
    val_ds = FusionDataset(
        val_ids, val_labels, text_map, age_map, sex_map,
        data_cfg["image_feat_dir"], tokenizer,
        max_len=task_cfg["max_len"], balance=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=task_cfg["batch_size"],
        shuffle=True, num_workers=task_cfg["num_workers"],
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=task_cfg["batch_size"],
        shuffle=False, num_workers=task_cfg["num_workers"],
        pin_memory=False,
    )

    # ---- Load pretrained sub-models ----
    text_net = BertClassifier(
        model_cfg["pretrained_bert"], num_labels=num_classes,
    ).to(device)
    img_net = GaanClassifier(num_classes=num_classes).to(device)

    text_state = torch.load(model_cfg["text_model_path"], map_location=device)
    text_net.load_state_dict(text_state)

    img_state = torch.load(model_cfg["img_model_path"], map_location=device)
    img_net.load_state_dict(img_state)

    # Freeze sub-models
    text_net.eval()
    img_net.eval()
    for p in text_net.parameters():
        p.requires_grad = False
    for p in img_net.parameters():
        p.requires_grad = False

    # ---- Fusion model ----
    fusion = MLPFeatureFusion(
        img_dim=task_cfg["img_dim"],
        text_dim=task_cfg["text_dim"],
        projection_dim=task_cfg["projection_dim"],
        num_classes=num_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(fusion.parameters(), lr=task_cfg["lr"])
    criterion = nn.CrossEntropyLoss()

    # ---- Training Loop ----
    best_val_acc = 0.0
    log_path = os.path.join(save_dir, "training_log.txt")

    for epoch in range(1, task_cfg["epochs"] + 1):
        # Train
        fusion.train()
        total_loss = total_correct = total_samples = 0

        for text_inputs, img_feat_batch, labels_batch in tqdm(
            train_loader, desc=f"Epoch {epoch}/{task_cfg['epochs']}", leave=False,
        ):
            input_ids, masks, age, sex = text_inputs
            input_ids = input_ids.to(device)
            masks = masks.to(device)
            age = age.to(device)
            sex = sex.to(device)
            img_feat = img_feat_batch.to(device)
            labels = labels_batch.to(device)

            optimizer.zero_grad()

            # Extract features (no gradient)
            with torch.no_grad():
                img_feat_vec = img_net([img_feat.squeeze(0)], return_feat=True)
                text_feat_vec = text_net(input_ids, masks, age, sex, return_feat=True)

            # Fusion classification
            output = fusion(img_feat_vec, text_feat_vec)
            if labels.dim() > 1:
                labels = labels.squeeze()

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                total_correct += (output.argmax(1) == labels).sum().item()
                total_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)

        train_loss = total_loss / max(total_samples, 1)
        train_acc = total_correct / max(total_samples, 1)

        # Validate
        fusion.eval()
        correct = total = 0
        all_preds, all_labels, all_logits = [], [], []
        results_records = []
        val_loss_sum = 0.0

        with torch.no_grad():
            for batch_idx, (text_inputs, img_feat_batch, labels_batch) in enumerate(val_loader):
                input_ids, masks, age, sex = text_inputs
                input_ids = input_ids.to(device)
                masks = masks.to(device)
                age = age.to(device)
                sex = sex.to(device)
                img_feat = img_feat_batch.to(device)
                labels = labels_batch.to(device)

                img_feat_vec = img_net([img_feat.squeeze(0)], return_feat=True)
                text_feat_vec = text_net(input_ids, masks, age, sex, return_feat=True)
                output = fusion(img_feat_vec, text_feat_vec)

                loss = criterion(output, labels)
                val_loss_sum += loss.item() * labels.size(0)

                probs = F.softmax(output, dim=1).cpu().numpy().squeeze(0)
                pred = int(np.argmax(probs))
                label = int(labels.cpu().numpy().squeeze(0))

                all_preds.append(pred)
                all_labels.append(label)
                all_logits.append(probs)

                pid = val_ids[batch_idx] if batch_idx < len(val_ids) else f"val_{batch_idx:04d}"
                results_records.append({
                    "ID": pid,
                    "true_label": label,
                    "predicted_label": pred,
                    **{f"prob_class_{i}": float(probs[i]) for i in range(len(probs))},
                })

                correct += 1 if pred == label else 0
                total += 1

        val_acc = correct / max(total, 1)
        val_loss = val_loss_sum / max(total, 1)

        # Compute F1
        from sklearn.metrics import f1_score, confusion_matrix, classification_report
        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
        report = classification_report(all_labels, all_preds, zero_division=0,
                                       labels=list(range(num_classes)), digits=4)

        print(
            f"Epoch {epoch}/{task_cfg['epochs']}  "
            f"Train Loss={train_loss:.4f}  Train Acc={train_acc:.4f} | "
            f"Val Loss={val_loss:.4f}  Val Acc={val_acc:.4f}  F1={val_f1:.4f}"
        )
        print(f"Confusion Matrix:\n{cm}")
        print(report)

        # Write to unified log file
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{'='*60}\n")
            f.write(f"Epoch {epoch}/{task_cfg['epochs']}\n")
            f.write(f"Train Loss={train_loss:.4f}  Train Acc={train_acc:.4f} | "
                    f"Val Loss={val_loss:.4f}  Val Acc={val_acc:.4f}  F1={val_f1:.4f}\n")
            f.write(f"Confusion Matrix:\n{cm}\n")
            f.write(f"{report}\n")

        # Save CSV
        csv_path = os.path.join(save_dir, f"epoch_{epoch}_results.csv")
        pd.DataFrame(results_records).to_csv(csv_path, index=False, encoding="utf-8-sig")

        # Save model
        torch.save(fusion.state_dict(), os.path.join(save_dir, f"fusion_epoch_{epoch}.pt"))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(fusion.state_dict(), os.path.join(save_dir, "best_model.pt"))
            print(f"  >> Saved best model (val_acc={best_val_acc:.4f})")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"  >> Best model saved (val_acc={best_val_acc:.4f})\n")

        # ROC
        roc_path = os.path.join(save_dir, f"roc_epoch{epoch}.png")
        plot_multiclass_roc(
            all_labels, all_logits, class_names, roc_path,
            n_bootstrap=task_cfg.get("n_bootstrap", 500),
        )

    print(f"\nTraining finished. Best Val Acc = {best_val_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Fusion MLP training")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
