"""
BERT text-only prediction script
Usage: python -m renal_cls.predict_bert --config config.yaml --input_excel path.xlsx --output_excel path.xlsx
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, auc, roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from .config import load_config, get_device
from .models import BertClassifier
from .utils import set_seed


# ============================================================
#  Dataset
# ============================================================
class BertPredictionDataset(Dataset):
    """Load text + clinical data from Excel for prediction"""

    def __init__(self, excel_path: str, tokenizer, pid_col: str = "ID",
                 label_col: str = "Class", max_len: int = 256):
        df = pd.read_excel(excel_path)

        if pid_col not in df.columns:
            raise ValueError(f"Column '{pid_col}' not found. Available: {list(df.columns)}")

        self.df = df
        self.pid_col = pid_col
        self.has_label = label_col in df.columns
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.ids = df[pid_col].astype(str).tolist()
        self.text_map = dict(zip(df[pid_col].astype(str), df["immunofluorescence_report"].astype(str)))
        self.age_map = dict(zip(df[pid_col].astype(str), df["age"].astype(float)))
        self.sex_map = dict(zip(df[pid_col].astype(str),
                                df["sex"].map({"male": -1, "female": 1}).astype(int)))

        if self.has_label:
            self.labels = df[label_col].tolist()
        else:
            self.labels = [-1] * len(df)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        text = self.text_map[pid]
        age = self.age_map[pid]
        sex = self.sex_map[pid]

        tokenized = self.tokenizer(
            text, max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )

        return {
            "pid": pid,
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "age": torch.tensor([float(age)], dtype=torch.float),
            "sex": torch.tensor([float(sex)], dtype=torch.float),
        }


# ============================================================
#  Predict
# ============================================================
def predict(
    input_excel, output_excel, model_path, pretrained_bert, device,
    pid_col="ID", label_col="Class", num_classes=6,
    class_names=None, max_len=256, batch_size=32, n_bootstrap=1000, seed=42,
):
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    save_dir = os.path.dirname(output_excel) or "."
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(pretrained_bert)
    dataset = BertPredictionDataset(input_excel, tokenizer, pid_col, label_col, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = BertClassifier(pretrained_bert, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[OK] Loaded BERT model: {model_path}")

    results = []
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Predicting")):
            pids = batch["pid"]
            inputs = {k: v.to(device) for k, v in batch.items() if k != "pid"}
            logits = model(**inputs)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            probs_np = probs.cpu().numpy()
            preds_np = preds.cpu().numpy()

            if dataset.has_label:
                batch_start = batch_idx * batch_size
                batch_labels = dataset.labels[batch_start:batch_start + len(preds)]
                all_labels.extend(batch_labels)

            all_preds.extend(preds_np.tolist())
            all_probs.extend(probs_np.tolist())

            for i in range(len(pids)):
                row = {pid_col: pids[i], "predicted_label": int(preds_np[i]),
                       "predicted_disease": class_names[int(preds_np[i])]}
                for j in range(num_classes):
                    row[f"class_{j}_prob"] = float(probs_np[i][j])
                results.append(row)

    result_df = pd.DataFrame(results)
    original_df = pd.read_excel(input_excel)
    result_df = pd.merge(original_df, result_df, on=pid_col)
    result_df.to_excel(output_excel, index=False)
    print(f"\n[OK] Predictions saved to: {output_excel}")

    # Evaluate if labels exist
    if len(all_labels) > 0:
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0))
        print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")

        # Save report
        report = classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0)
        report_path = output_excel.replace(".xlsx", "_report.txt")
        with open(report_path, "w") as f:
            f.write(report)

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(output_excel.replace(".xlsx", "_cm.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # ROC with bootstrap CI
        y_onehot = label_binarize(all_labels, classes=list(range(num_classes)))
        y_prob = np.array(all_probs)
        fpr, tpr, roc_auc_val = {}, {}, {}
        for i in range(num_classes):
            try:
                fpr[i], tpr[i], _ = roc_curve(y_onehot[:, i], y_prob[:, i])
                roc_auc_val[i] = auc(fpr[i], tpr[i])
            except Exception:
                roc_auc_val[i] = np.nan

        # Bootstrap CI
        rng = np.random.RandomState(seed)
        boot_aucs = {i: [] for i in range(num_classes)}
        boot_aucs["micro"] = []
        boot_aucs["macro"] = []
        n = len(all_labels)
        for _ in range(n_bootstrap):
            idx = rng.randint(0, n, n)
            y_b, s_b = y_onehot[idx], y_prob[idx]
            for i in range(num_classes):
                try:
                    boot_aucs[i].append(roc_auc_score(y_b[:, i], s_b[:, i]))
                except Exception:
                    boot_aucs[i].append(np.nan)
            try:
                boot_aucs["micro"].append(roc_auc_score(y_b.ravel(), s_b.ravel()))
            except Exception:
                boot_aucs["micro"].append(np.nan)
            per_class = []
            for i in range(num_classes):
                try:
                    per_class.append(roc_auc_score(y_b[:, i], s_b[:, i]))
                except Exception:
                    per_class.append(np.nan)
            arr = np.array(per_class, dtype=float)
            boot_aucs["macro"].append(np.nanmean(arr) if not np.all(np.isnan(arr)) else np.nan)

        ci_dict = {}
        for k, vals in boot_aucs.items():
            arr = np.array(vals, dtype=float)
            arr = arr[~np.isnan(arr)]
            ci_dict[k] = (np.percentile(arr, 2.5), np.percentile(arr, 97.5)) if arr.size > 0 else (np.nan, np.nan)

        print("\nAUC per class (95% CI):")
        for i in range(num_classes):
            ci = ci_dict.get(i, (np.nan, np.nan))
            print(f"  {class_names[i]:<8} AUC={roc_auc_val.get(i, np.nan):.4f} 95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")

        # ROC plot
        plt.figure(figsize=(8, 6))
        for i in range(num_classes):
            if not np.isnan(roc_auc_val.get(i, np.nan)):
                ci = ci_dict[i]
                plt.plot(fpr[i], tpr[i], lw=1,
                         label=f"{class_names[i]} (AUC={roc_auc_val[i]:.3f} [{ci[0]:.3f}-{ci[1]:.3f}])")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(fontsize="small")
        plt.tight_layout()
        plt.savefig(output_excel.replace(".xlsx", "_roc.png"), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\n[OK] ROC curve saved to: {output_excel.replace('.xlsx', '_roc.png')}")
    else:
        print("No true labels found, skipping evaluation.")


# ============================================================
#  CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT text prediction")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--input_excel", type=str, required=True, help="Input Excel file")
    parser.add_argument("--output_excel", type=str, required=True, help="Output prediction Excel")
    parser.add_argument("--model_path", type=str, required=True, help="BERT model checkpoint")
    parser.add_argument("--pretrained_bert", type=str, default=None, help="Med-BERT pretrained path")
    parser.add_argument("--pid_col", type=str, default="ID")
    parser.add_argument("--label_col", type=str, default="Class")
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}
    class_names = cfg.get("common", {}).get("class_names", ["IgAN", "MCD", "MN", "DN", "MSPGN", "LN"])
    pretrained_bert = args.pretrained_bert or cfg.get("models", {}).get("pretrained_bert", "")

    if args.device.isdigit():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device(args.device)

    predict(
        input_excel=args.input_excel, output_excel=args.output_excel,
        model_path=args.model_path, pretrained_bert=pretrained_bert,
        device=device, pid_col=args.pid_col, label_col=args.label_col,
        num_classes=args.num_classes, class_names=class_names,
        max_len=args.max_len, batch_size=args.batch_size,
        n_bootstrap=args.n_bootstrap, seed=args.seed,
    )
