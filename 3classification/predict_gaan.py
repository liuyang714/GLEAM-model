"""
GAAN image-only prediction script
Usage: python -m renal_cls.predict_gaan --config config.yaml --input_excel path.xlsx --output_excel path.xlsx
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
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, auc, roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from .config import load_config, get_device
from .models import GaanClassifier
from .utils import set_seed, save_summary_xlsx


# ============================================================
#  Dataset
# ============================================================
class GaanPredictionDataset(Dataset):
    """Load image features from Excel + .pt files"""

    def __init__(self, excel_path: str, image_feat_dir: str, pid_col: str = "ID", label_col: str = "Class"):
        self.pid_col = pid_col
        self.label_col = label_col
        self.image_feat_dir = image_feat_dir

        df = pd.read_excel(excel_path)
        if pid_col not in df.columns:
            raise ValueError(f"Column '{pid_col}' not found. Available: {list(df.columns)}")

        self.has_label = label_col in df.columns
        self.df = df.reset_index(drop=True)
        print(f"[OK] Predictable samples: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = str(row[self.pid_col])
        bag = torch.load(os.path.join(self.image_feat_dir, f"{pid}.pt"), map_location="cpu")

        if isinstance(bag, dict) and "tokens" in bag:
            bag = bag["tokens"]
        if isinstance(bag, torch.Tensor) and bag.dim() == 4 and bag.size(0) == 1:
            bag = bag.squeeze(0)

        bag = bag.float()
        label = int(row[self.label_col]) if self.has_label else -1
        return bag, pid, label


def collate_fn(batch):
    bags, pids, labels = zip(*batch)
    return list(bags), list(pids), torch.tensor(labels, dtype=torch.long)


# ============================================================
#  Model loading
# ============================================================
def clean_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k.replace("module.", "", 1)
        new_state_dict[k] = v
    return new_state_dict


def load_gaan_model(model_path, device, num_classes=6):
    model = GaanClassifier(num_classes=num_classes).to(device)
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    else:
        raise TypeError(f"Unknown checkpoint format: {type(ckpt)}")
    model.load_state_dict(clean_state_dict(state_dict))
    model.eval()
    print(f"[OK] Loaded GAAN model: {model_path}")
    return model


# ============================================================
#  Evaluation
# ============================================================
def evaluate_and_plot(y_true, y_pred, y_score, save_dir, class_names, n_bootstrap=1000, seed=42):
    os.makedirs(save_dir, exist_ok=True)
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    y_score = np.array(y_score, dtype=float)
    num_classes = len(class_names)
    present = sorted(set(y_true.tolist() + y_pred.tolist()))

    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred, labels=present,
        target_names=[class_names[i] for i in present], digits=4, zero_division=0,
    ))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    cm = confusion_matrix(y_true, y_pred, labels=present)
    print(f"Confusion Matrix:\n{cm}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=present)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[class_names[i] for i in present],
                yticklabels=[class_names[i] for i in present])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    # ROC with bootstrap CI
    y_onehot = label_binarize(y_true, classes=list(range(num_classes)))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in present:
        try:
            fpr[i], tpr[i], _ = roc_curve(y_onehot[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        except Exception:
            roc_auc[i] = np.nan

    if len(present) > 1:
        try:
            fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot[:, present].ravel(), y_score[:, present].ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        except Exception:
            roc_auc["micro"] = np.nan
        valid_fprs = [i for i in present if i in fpr]
        if valid_fprs:
            all_fpr = np.unique(np.concatenate([fpr[i] for i in valid_fprs]))
            mean_tpr = np.zeros_like(all_fpr, dtype=float)
            for i in valid_fprs:
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= len(valid_fprs)
            fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Bootstrap CI
    rng = np.random.RandomState(seed)
    boot_aucs = {i: [] for i in present}
    boot_aucs["micro"] = []
    boot_aucs["macro"] = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, len(y_true), len(y_true))
        y_b, s_b = y_onehot[idx], y_score[idx]
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
            print(f"  {class_names[i]:<8} AUC={roc_auc.get(i, np.nan):.4f} 95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"  Micro    AUC={roc_auc.get('micro', np.nan):.4f} 95% CI [{ci_dict['micro'][0]:.4f}, {ci_dict['micro'][1]:.4f}]")
    print(f"  Macro    AUC={roc_auc.get('macro', np.nan):.4f} 95% CI [{ci_dict['macro'][0]:.4f}, {ci_dict['macro'][1]:.4f}]")

    # ROC plot
    plt.figure(figsize=(7, 6))
    for key in ["micro", "macro"]:
        if not np.isnan(roc_auc.get(key, np.nan)):
            ci = ci_dict[key]
            plt.plot(fpr[key], tpr[key], lw=2,
                     label=f"{key}-average (AUC={roc_auc[key]:.3f} [{ci[0]:.3f}-{ci[1]:.3f}])")
    for i in present:
        if i in fpr:
            ci = ci_dict.get(i, (np.nan, np.nan))
            plt.plot(fpr[i], tpr[i], lw=1,
                     label=f"{class_names[i]} (AUC={roc_auc[i]:.3f} [{ci[0]:.3f}-{ci[1]:.3f}])")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right", fontsize="small")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_auc.png"), dpi=300)
    plt.close()

    # AUC Boxplot (bootstrap)
    from matplotlib.collections import PolyCollection
    rng_box = np.random.RandomState(42)
    boot_aucs_box = {i: [] for i in present}
    for _ in range(1000):
        idx_b = rng_box.randint(0, n, n)
        for i in present:
            try:
                boot_aucs_box[i].append(roc_auc_score(y_onehot[idx_b][:, i], y_score[idx_b][:, i]))
            except Exception:
                pass

    plt.figure(figsize=(7, 5))
    for idx_i, i in enumerate(present):
        vals = np.array(boot_aucs_box[i])
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        bp = plt.gca().boxplot(vals, positions=[idx_i], widths=0.4, showcaps=True, showfliers=False, patch_artist=True)
        for box in bp["boxes"]:
            box.set_facecolor(["#FFB7B2", "#FFDAC1", "#E2F0CB", "#B5EAD7", "#C7CEEA", "#E8C4EC"][i % 6])
            box.set_alpha(0.7)
            box.set_edgecolor(["#E07A7A", "#E5A97E", "#B0C986", "#79C2A5", "#8FA0D2", "#B387B8"][i % 6])
        for w in bp["whiskers"]:
            w.set_color(["#E07A7A", "#E5A97E", "#B0C986", "#79C2A5", "#8FA0D2", "#B387B8"][i % 6])
        for c in bp["caps"]:
            c.set_color(["#E07A7A", "#E5A97E", "#B0C986", "#79C2A5", "#8FA0D2", "#B387B8"][i % 6])
        for m in bp["medians"]:
            m.set_color(["#E07A7A", "#E5A97E", "#B0C986", "#79C2A5", "#8FA0D2", "#B387B8"][i % 6])
    ax_box = plt.gca()
    ax_box.set_xticks(range(len(present)))
    ax_box.set_xticklabels([class_names[i] for i in present])
    ax_box.set_ylabel("AUC")
    ax_box.set_title("AUC Distribution (Bootstrap)")
    ax_box.grid(axis="y", linestyle=":", alpha=0.4)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "auc_boxplot.png"), dpi=300)
    plt.close()

    # F1 Violin Plot (bootstrap)
    boot_f1s = {i: [] for i in present}
    for _ in range(1000):
        idx_b = rng_box.randint(0, n, n)
        boot_true_b = y_true[idx_b]
        boot_pred_b = y_pred[idx_b]
        for i in present:
            true_c = (boot_true_b == i).astype(int)
            pred_c = (boot_pred_b == i).astype(int)
            boot_f1s[i].append(f1_score(true_c, pred_c, zero_division=0))

    rows_f1 = []
    for i in present:
        for v in boot_f1s[i]:
            rows_f1.append({"Class": class_names[i], "F1": v})
    df_f1 = pd.DataFrame(rows_f1)

    plt.figure(figsize=(7, 5))
    palette = {class_names[i]: ["#FFB7B2", "#FFDAC1", "#E2F0CB", "#B5EAD7", "#C7CEEA", "#E8C4EC"][i % 6] for i in present}
    sns.violinplot(data=df_f1, x="Class", y="F1", order=[class_names[i] for i in present],
                   inner=None, cut=0, linewidth=0.8, palette=palette, saturation=1.0)
    violin_bodies = [c for c in plt.gca().collections if isinstance(c, PolyCollection)]
    for idx_b, body in enumerate(violin_bodies):
        if idx_b < len(present):
            body.set_alpha(0.65)
            body.set_edgecolor(["#E07A7A", "#E5A97E", "#B0C986", "#79C2A5", "#8FA0D2", "#B387B8"][present[idx_b] % 6])
    plt.xlabel("True labels")
    plt.ylabel("F1-Score")
    plt.title("F1-Score Distribution (Bootstrap)")
    plt.grid(axis="y", linestyle=":", alpha=0.4)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "f1_violin.png"), dpi=300)
    plt.close()


# ============================================================
#  Predict
# ============================================================
def predict(
    input_excel, output_excel, image_feat_dir, model_path, device,
    pid_col="ID", label_col="Class", num_classes=6,
    class_names=None, batch_size=1, n_bootstrap=1000, seed=42,
):
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    save_dir = os.path.dirname(output_excel) or "."
    os.makedirs(save_dir, exist_ok=True)

    dataset = GaanPredictionDataset(input_excel, image_feat_dir, pid_col, label_col)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    model = load_gaan_model(model_path, device, num_classes)

    predictions, all_true, all_pred, all_probs = [], [], [], []

    with torch.no_grad():
        for bags, pids, labels in tqdm(loader, desc="Predicting"):
            bags = [b.to(device) for b in bags]
            labels = labels.to(device)
            logits = model(bags)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            probs_np, pred_np, labels_np = probs.cpu().numpy(), preds.cpu().numpy(), labels.cpu().numpy()
            for i, pid in enumerate(pids):
                row = {pid_col: pid, "true_label": int(labels_np[i]), "predicted_label": int(pred_np[i])}
                for c in range(num_classes):
                    row[f"class_{c}_prob"] = float(probs_np[i, c])
                row["predicted_disease"] = class_names[int(pred_np[i])]
                predictions.append(row)
                all_true.append(int(labels_np[i]))
                all_pred.append(int(pred_np[i]))
                all_probs.append(probs_np[i])

    pred_df = pd.DataFrame(predictions)
    original_df = dataset.df.copy()
    original_cols = [c for c in original_df.columns if c not in ["true_label", "predicted_label"]]
    result_df = pd.merge(original_df[original_cols], pred_df, on=pid_col, how="inner")

    prob_cols = [f"class_{i}_prob" for i in range(num_classes)]
    front_cols = [pid_col, "true_label", "predicted_label", "predicted_disease"] + prob_cols
    other_cols = [c for c in result_df.columns if c not in front_cols]
    result_df = result_df[front_cols + other_cols]
    result_df.to_excel(output_excel, index=False)
    print(f"\n[OK] Predictions saved to: {output_excel}")

    # Evaluate if labels exist
    if any(l != -1 for l in all_true):
        valid = [i for i, l in enumerate(all_true) if l != -1]
        evaluate_and_plot(
            [all_true[i] for i in valid], [all_pred[i] for i in valid],
            np.array([all_probs[i] for i in valid]), save_dir, class_names, n_bootstrap, seed,
        )

        # Summary Excel with bootstrap CI
        summary_path = os.path.join(save_dir, "summary.xlsx")
        save_summary_xlsx(
            summary_path, class_names,
            [all_true[i] for i in valid], [all_pred[i] for i in valid],
            np.array([all_probs[i] for i in valid]),
            title_text="GAAN Six-Class Results",
        )
    else:
        print("No true labels found, skipping evaluation.")


# ============================================================
#  CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAAN image prediction")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--input_excel", type=str, required=True, help="Input Excel (must contain ID column)")
    parser.add_argument("--output_excel", type=str, required=True, help="Output prediction Excel")
    parser.add_argument("--image_feat_dir", type=str, required=True, help="Directory of .pt feature files")
    parser.add_argument("--model_path", type=str, required=True, help="GAAN model checkpoint path")
    parser.add_argument("--pid_col", type=str, default="ID", help="Patient ID column name")
    parser.add_argument("--label_col", type=str, default="Class", help="True label column name")
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}
    class_names = cfg.get("common", {}).get("class_names", ["IgAN", "MCD", "MN", "DN", "MSPGN", "LN"])

    if args.device.isdigit():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device(args.device)

    predict(
        input_excel=args.input_excel, output_excel=args.output_excel,
        image_feat_dir=args.image_feat_dir, model_path=args.model_path,
        device=device, pid_col=args.pid_col, label_col=args.label_col,
        num_classes=args.num_classes, class_names=class_names,
        batch_size=args.batch_size, n_bootstrap=args.n_bootstrap, seed=args.seed,
    )
