"""
Multimodal fusion prediction script (BERT text + GAAN image -> MLP)
Usage: python -m renal_cls.predict_fusion --config config.yaml --input_excel path.xlsx --output_excel path.xlsx
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
from .models import BertClassifier, GaanClassifier, MLPFeatureFusion
from .utils import set_seed


# ============================================================
#  Dataset
# ============================================================
class FusionPredictionDataset(Dataset):
    """Load text + image features from Excel + .pt files"""

    def __init__(self, excel_path: str, image_feat_dir: str, tokenizer,
                 pid_col: str = "ID", label_col: str = "Class", max_len: int = 256):
        df = pd.read_excel(excel_path)
        if pid_col not in df.columns:
            raise ValueError(f"Column '{pid_col}' not found. Available: {list(df.columns)}")

        self.df = df
        self.pid_col = pid_col
        self.label_col = label_col
        self.image_feat_dir = image_feat_dir
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.ids = df[pid_col].astype(str).tolist()
        self.text_map = dict(zip(df[pid_col].astype(str), df["immunofluorescence_report"].astype(str)))
        self.age_map = dict(zip(df[pid_col].astype(str), df["age"].astype(float)))
        self.sex_map = dict(zip(df[pid_col].astype(str),
                                df["sex"].map({"male": -1, "female": 1}).astype(int)))
        self.labels = df[label_col].tolist() if label_col in df.columns else [-1] * len(df)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        text = self.text_map[pid]
        age = self.age_map[pid]
        sex = self.sex_map[pid]

        enc = self.tokenizer(
            text, padding="max_length", truncation=True,
            max_length=self.max_len, return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        age_t = torch.tensor(float(age), dtype=torch.float)
        sex_t = torch.tensor(float(sex), dtype=torch.float)

        img_path = os.path.join(self.image_feat_dir, f"{pid}.pt")
        img_feat = torch.load(img_path, map_location="cpu")
        if isinstance(img_feat, dict) and "tokens" in img_feat:
            img_feat = img_feat["tokens"]

        label = self.labels[idx]
        return (input_ids, attention_mask, age_t, sex_t), img_feat, pid, label


# ============================================================
#  Predict
# ============================================================
def predict(
    input_excel, output_excel, image_feat_dir,
    text_model_path, img_model_path, fusion_model_path,
    pretrained_bert, device,
    pid_col="ID", label_col="Class", num_classes=6,
    class_names=None, max_len=256, n_bootstrap=1000, seed=42,
):
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    save_dir = os.path.dirname(output_excel) or "."
    os.makedirs(save_dir, exist_ok=True)

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(pretrained_bert)

    # Load models
    text_net = BertClassifier(pretrained_bert, num_classes).to(device)
    img_net = GaanClassifier(num_classes=num_classes).to(device)
    fusion = MLPFeatureFusion(num_classes=num_classes).to(device)

    text_net.load_state_dict(torch.load(text_model_path, map_location=device))
    img_net.load_state_dict(torch.load(img_model_path, map_location=device))
    fusion.load_state_dict(torch.load(fusion_model_path, map_location=device))
    text_net.eval()
    img_net.eval()
    fusion.eval()
    print(f"[OK] Loaded text model: {text_model_path}")
    print(f"[OK] Loaded image model: {img_model_path}")
    print(f"[OK] Loaded fusion model: {fusion_model_path}")

    # Dataset & loader
    df = pd.read_excel(input_excel)
    dataset = FusionPredictionDataset(input_excel, image_feat_dir, tokenizer, pid_col, label_col, max_len)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    predictions, true_labels, all_logits, all_probs = [], [], [], []

    with torch.no_grad():
        for text_inputs, img_feat, pid, label in tqdm(loader, desc="Predicting"):
            input_ids, masks, age, sex = [t.to(device) for t in text_inputs]
            img_feat = img_feat.to(device)

            true_labels.append(label.item())

            img_out = img_net([img_feat.squeeze(0)])
            txt_out = text_net(input_ids, masks, age, sex)
            combined = torch.cat((img_out, txt_out), dim=1)
            logits = fusion(combined)

            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            all_logits.append(logits.cpu().numpy()[0])
            all_probs.append(probs)
            pred_class = int(torch.argmax(logits, dim=1).item())

            row = {pid_col: pid[0], "true_label": label.item(), "predicted_label": pred_class}
            for i in range(num_classes):
                row[f"class_{i}_prob"] = float(probs[i])
            predictions.append(row)

    # Save results
    result_df = pd.DataFrame(predictions)
    original_cols = [c for c in df.columns if c not in ["predicted_label", "true_label"]]
    result_df = pd.merge(df[original_cols], result_df, on=pid_col)
    col_order = [pid_col, "true_label", "predicted_label"] + \
                [f"class_{i}_prob" for i in range(num_classes)] + \
                [c for c in original_cols if c != pid_col]
    result_df = result_df[col_order]
    result_df.to_excel(output_excel, index=False)
    print(f"\n[OK] Predictions saved to: {output_excel}")

    # Evaluate if labels exist
    if any(l != -1 for l in true_labels):
        valid = [i for i, l in enumerate(true_labels) if l != -1]
        valid_true = [true_labels[i] for i in valid]
        valid_logits = np.array([all_logits[i] for i in valid])
        valid_pred = [result_df.loc[i, "predicted_label"] for i in valid]

        present = sorted(set(valid_true + valid_pred))
        print(f"\nPresent classes: {present}")

        cm = confusion_matrix(valid_true, valid_pred, labels=present)
        print("\nClassification Report:")
        print(classification_report(valid_true, valid_pred, labels=present, digits=4, zero_division=0))
        print(f"Accuracy: {accuracy_score(valid_true, valid_pred):.4f}")

        # ROC with bootstrap CI
        y_onehot = label_binarize(valid_true, classes=list(range(num_classes)))
        fpr, tpr, roc_auc_val = {}, {}, {}
        for i in present:
            try:
                fpr[i], tpr[i], _ = roc_curve(y_onehot[:, i], valid_logits[:, i])
                roc_auc_val[i] = auc(fpr[i], tpr[i])
            except Exception:
                roc_auc_val[i] = np.nan

        if len(present) > 1:
            try:
                fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot[:, present].ravel(), valid_logits[:, present].ravel())
                roc_auc_val["micro"] = auc(fpr["micro"], tpr["micro"])
            except Exception:
                roc_auc_val["micro"] = np.nan
            valid_fprs = [i for i in present if i in fpr]
            if valid_fprs:
                all_fpr = np.unique(np.concatenate([fpr[i] for i in valid_fprs]))
                mean_tpr = np.zeros_like(all_fpr, dtype=float)
                for i in valid_fprs:
                    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
                mean_tpr /= len(valid_fprs)
                fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
                roc_auc_val["macro"] = auc(fpr["macro"], tpr["macro"])

        # Bootstrap CI
        rng = np.random.RandomState(seed)
        boot_aucs = {i: [] for i in present}
        boot_aucs["micro"] = []
        boot_aucs["macro"] = []
        n = len(valid_true)
        for _ in range(n_bootstrap):
            idx = rng.randint(0, n, n)
            y_b, s_b = y_onehot[idx], valid_logits[idx]
            for i in present:
                try:
                    boot_aucs[i].append(roc_auc_score(y_b[:, i], s_b[:, i]))
                except Exception:
                    boot_aucs[i].append(np.nan)
            try:
                boot_aucs["micro"].append(roc_auc_score(y_b[:, present].ravel(), s_b[:, present].ravel()))
            except Exception:
                boot_aucs["micro"].append(np.nan)
            per_class = []
            for i in present:
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
            if i in present:
                ci = ci_dict.get(i, (np.nan, np.nan))
                print(f"  {class_names[i]:<8} AUC={roc_auc_val.get(i, np.nan):.4f} 95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")
        print(f"  Micro    AUC={roc_auc_val.get('micro', np.nan):.4f} 95% CI [{ci_dict['micro'][0]:.4f}, {ci_dict['micro'][1]:.4f}]")
        print(f"  Macro    AUC={roc_auc_val.get('macro', np.nan):.4f} 95% CI [{ci_dict['macro'][0]:.4f}, {ci_dict['macro'][1]:.4f}]")

        # ROC plot
        plt.figure(figsize=(7, 6))
        for key in ["micro", "macro"]:
            if not np.isnan(roc_auc_val.get(key, np.nan)):
                ci = ci_dict[key]
                plt.plot(fpr[key], tpr[key], lw=2,
                         label=f"{key}-average (AUC={roc_auc_val[key]:.3f} [{ci[0]:.3f}-{ci[1]:.3f}])")
        for i in present:
            if i in fpr:
                ci = ci_dict.get(i, (np.nan, np.nan))
                plt.plot(fpr[i], tpr[i], lw=1,
                         label=f"{class_names[i]} (AUC={roc_auc_val[i]:.3f} [{ci[0]:.3f}-{ci[1]:.3f}])")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Fusion ROC Curve")
        plt.legend(loc="lower right", fontsize="small")
        plt.tight_layout()
        roc_path = os.path.join(save_dir, "fusion_roc_auc.png")
        plt.savefig(roc_path, dpi=300)
        plt.close()
        print(f"\n[OK] ROC curve saved to: {roc_path}")

        # Confusion matrix
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=[class_names[i] for i in present],
                    yticklabels=[class_names[i] for i in present])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        cm_path = os.path.join(save_dir, "fusion_confusion_matrix.png")
        plt.savefig(cm_path, dpi=300)
        plt.close()
        print(f"[OK] Confusion matrix saved to: {cm_path}")
    else:
        print("No true labels found, skipping evaluation.")


# ============================================================
#  CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal fusion prediction")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--input_excel", type=str, required=True, help="Input Excel file")
    parser.add_argument("--output_excel", type=str, required=True, help="Output prediction Excel")
    parser.add_argument("--image_feat_dir", type=str, required=True, help="Directory of .pt feature files")
    parser.add_argument("--text_model_path", type=str, default=None, help="BERT model checkpoint")
    parser.add_argument("--img_model_path", type=str, default=None, help="GAAN model checkpoint")
    parser.add_argument("--fusion_model_path", type=str, required=True, help="Fusion MLP checkpoint")
    parser.add_argument("--pretrained_bert", type=str, default=None, help="Med-BERT pretrained path")
    parser.add_argument("--pid_col", type=str, default="ID")
    parser.add_argument("--label_col", type=str, default="Class")
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}
    class_names = cfg.get("common", {}).get("class_names", ["IgAN", "MCD", "MN", "DN", "MSPGN", "LN"])
    model_cfg = cfg.get("models", {})
    pretrained_bert = args.pretrained_bert or model_cfg.get("pretrained_bert", "")
    text_model_path = args.text_model_path or model_cfg.get("text_model_path", "")
    img_model_path = args.img_model_path or model_cfg.get("img_model_path", "")

    if args.device.isdigit():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device(args.device)

    predict(
        input_excel=args.input_excel, output_excel=args.output_excel,
        image_feat_dir=args.image_feat_dir,
        text_model_path=text_model_path, img_model_path=img_model_path,
        fusion_model_path=args.fusion_model_path, pretrained_bert=pretrained_bert,
        device=device, pid_col=args.pid_col, label_col=args.label_col,
        num_classes=args.num_classes, class_names=class_names,
        max_len=args.max_len, n_bootstrap=args.n_bootstrap, seed=args.seed,
    )
