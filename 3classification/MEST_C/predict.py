"""
MEST-C component prediction script
Usage: python -m renal_cls.mestc.predict --task M --input_excel path.xlsx --output_csv path.csv
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from ..config import load_config, get_device
from ..models import GaanClassifier
from .data import TASK_CONFIG


# ============================================================
#  Prediction Dataset
# ============================================================
class MESTCPredictionDataset(Dataset):
    """Load features from Excel + .pt files for prediction"""

    def __init__(self, excel_path: str, pt_dir: str, id_column: str = "ID"):
        df = pd.read_excel(excel_path)
        if id_column not in df.columns:
            raise ValueError(f"Column '{id_column}' not found. Available: {list(df.columns)}")

        self.df = df
        self.id_column = id_column
        self.pt_dir = pt_dir
        self.ids = df[id_column].astype(str).tolist()

        print(f"[OK] Predictable samples: {len(self.ids)}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        pt_path = os.path.join(self.pt_dir, f"{sample_id}.pt")

        bag = torch.load(pt_path, weights_only=True)
        if isinstance(bag, dict) and "tokens" in bag:
            bag = bag["tokens"]

        return bag, sample_id


def collate_fn(batch):
    bags = [item[0] for item in batch]
    ids = [item[1] for item in batch]
    return bags, ids


# ============================================================
#  Model Loading
# ============================================================
def clean_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k.replace("module.", "", 1)
        new_state_dict[k] = v
    return new_state_dict


def detect_num_classes(model_path):
    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        if "fc2.weight" in ckpt:
            return ckpt["fc2.weight"].shape[0]
        return 2
    except Exception:
        return 2


# ============================================================
#  Predict
# ============================================================
def predict(
    input_excel: str,
    pt_dir: str,
    model_path: str,
    output_csv: str,
    task: str,
    id_column: str = "ID",
    device: str = "0",
):
    _, _, num_classes = TASK_CONFIG[task]
    class_names = [f"{task}{i}" for i in range(num_classes)]

    if device.isdigit():
        dev = torch.device(f"cuda:{device}")
    else:
        dev = torch.device(device)

    # Load model
    model = GaanClassifier(num_classes=num_classes).to(dev)
    ckpt = torch.load(model_path, map_location=dev, weights_only=False)
    model.load_state_dict(clean_state_dict(ckpt))
    model.eval()
    print(f"[OK] Loaded {task} model: {model_path} (classes={num_classes})")

    # Dataset
    dataset = MESTCPredictionDataset(input_excel, pt_dir, id_column)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Predict
    results = []
    with torch.no_grad():
        for bags, sample_ids in tqdm(loader, desc=f"Predicting {task}"):
            bags = [b.to(dev) for b in bags]
            logits = model(bags)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))

            row = {id_column: sample_ids[0], f"{task}_pred": pred}
            for c in range(num_classes):
                row[f"{task}{c}_prob"] = float(probs[c])
            results.append(row)

    result_df = pd.DataFrame(results)

    # Merge with original Excel
    original_df = pd.read_excel(input_excel)
    result_df = pd.merge(original_df, result_df, on=id_column)

    # Save
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] {task} predictions saved to: {output_csv}")

    # Evaluate if true labels exist
    label_col = f"{task}" if task in result_df.columns else None
    if label_col and result_df[label_col].notna().any():
        true_vals = pd.to_numeric(result_df[label_col], errors="coerce")
        valid = true_vals.notna()
        if valid.any():
            y_true = true_vals[valid].astype(int).tolist()
            y_pred = result_df.loc[valid, f"{task}_pred"].tolist()

            print(f"\n[{task}] Classification Report:")
            print(classification_report(y_true, y_pred, labels=list(range(num_classes)),
                                        target_names=class_names, digits=4, zero_division=0))

            cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

            # Print confusion matrix to terminal
            print(f"\n[{task}] Confusion Matrix:")
            header = "        " + "  ".join(f"{name:>8}" for name in class_names)
            print(header)
            for i, row_name in enumerate(class_names):
                row_str = "  ".join(f"{cm[i, j]:>8d}" for j in range(num_classes))
                print(f"  {row_name:>6}  {row_str}")

            # Save confusion matrix image
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=class_names, yticklabels=class_names)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"{task} Component - Confusion Matrix")
            plt.tight_layout()
            cm_path = output_csv.replace(".csv", f"_{task}_cm.png")
            plt.savefig(cm_path, dpi=300)
            plt.close()
            print(f"\n[OK] Confusion matrix saved to: {cm_path}")

    return result_df


# ============================================================
#  CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MEST-C component prediction")
    parser.add_argument("--task", type=str, required=True, choices=["M", "E", "S", "T", "C"])
    parser.add_argument("--input_excel", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--pt_dir", type=str, required=True, help="Directory of .pt feature files")
    parser.add_argument("--model_path", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--id_column", type=str, default="ID")
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    predict(
        input_excel=args.input_excel, pt_dir=args.pt_dir,
        model_path=args.model_path, output_csv=args.output_csv,
        task=args.task, id_column=args.id_column, device=args.device,
    )
