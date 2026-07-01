"""
Six-class evaluation script: training (2 plots) and prediction (4 plots + document)
Usage:
  Training:  python -m 3classification.evaluate --mode train --input /path/to/val_predictions.csv
  Prediction: python -m 3classification.evaluate --mode predict --input /path/to/test_predictions.xlsx
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    roc_auc_score, f1_score, accuracy_score,
    classification_report,
)
from sklearn.preprocessing import label_binarize


# ============================================================
#  Visual settings
# ============================================================
sns.set_theme(style="ticks")
plt.rcParams["font.sans-serif"] = ["Arial", "Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 600

MACARON_LIGHT = ["#FFB7B2", "#FFDAC1", "#E2F0CB", "#B5EAD7", "#C7CEEA", "#E8C4EC"]
MACARON_DARK = ["#E07A7A", "#E5A97E", "#B0C986", "#79C2A5", "#8FA0D2", "#B387B8"]

DEFAULT_CLASS_NAMES = ["IgAN", "MCD", "MN", "DN", "MsPGN", "LN"]


# ============================================================
#  Plot helpers
# ============================================================
def _save_fig(plt_or_fig, path, dpi=600):
    plt_or_fig.savefig(path, dpi=dpi, bbox_inches="tight")
    svg_path = os.path.splitext(path)[0] + ".svg"
    plt_or_fig.savefig(svg_path, format="svg", bbox_inches="tight")


def plot_roc(y_true, y_score, class_names, output_dir, n_bootstrap=1000):
    """Plot ROC curve with bootstrap 95% CI."""
    num_classes = len(class_names)
    y_onehot = label_binarize(y_true, classes=list(range(num_classes)))

    fpr, tpr, roc_auc_val = {}, {}, {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot[:, i], y_score[:, i])
        roc_auc_val[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot.ravel(), y_score.ravel())
    roc_auc_val["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc_val["macro"] = auc(fpr["macro"], tpr["macro"])

    # Bootstrap CI
    rng = np.random.RandomState(42)
    boot_micro, boot_macro = [], []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        try:
            boot_micro.append(roc_auc_score(y_onehot[idx].ravel(), y_score[idx].ravel()))
        except Exception:
            boot_micro.append(np.nan)
        try:
            per_class = [roc_auc_score(y_onehot[idx][:, i], y_score[idx][:, i]) for i in range(num_classes)]
            boot_macro.append(np.nanmean(per_class))
        except Exception:
            boot_macro.append(np.nan)

    boot_micro = [x for x in boot_micro if not np.isnan(x)]
    boot_macro = [x for x in boot_macro if not np.isnan(x)]
    ci_micro = (np.percentile(boot_micro, 2.5), np.percentile(boot_micro, 97.5)) if boot_micro else (np.nan, np.nan)
    ci_macro = (np.percentile(boot_macro, 2.5), np.percentile(boot_macro, 97.5)) if boot_macro else (np.nan, np.nan)

    # Plot
    plt.figure(figsize=(7, 6))
    plt.plot(fpr["micro"], tpr["micro"], lw=2.5,
             label=f"micro-average (AUC={roc_auc_val['micro']:.3f} [{ci_micro[0]:.3f}-{ci_micro[1]:.3f}])")
    plt.plot(fpr["macro"], tpr["macro"], lw=2.0,
             label=f"macro-average (AUC={roc_auc_val['macro']:.3f} [{ci_macro[0]:.3f}-{ci_macro[1]:.3f}])")
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], lw=1.2,
                 label=f"{class_names[i]} (AUC={roc_auc_val[i]:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right", fontsize="small")
    plt.tight_layout()
    _save_fig(plt, os.path.join(output_dir, "roc_curve.png"))
    plt.close()

    print(f"  Micro-AUC = {roc_auc_val['micro']:.4f} [{ci_micro[0]:.4f}-{ci_micro[1]:.4f}]")
    print(f"  Macro-AUC = {roc_auc_val['macro']:.4f} [{ci_macro[0]:.4f}-{ci_macro[1]:.4f}]")
    for i in range(num_classes):
        print(f"    {class_names[i]:10s} AUC = {roc_auc_val[i]:.4f}")

    return roc_auc_val, ci_micro, ci_macro


def plot_confusion_matrix(y_true, y_pred, class_names, output_dir):
    """Plot confusion matrix."""
    num_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    _save_fig(plt, os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=list(range(num_classes)),
                                target_names=class_names, digits=4, zero_division=0))
    return cm


def plot_auc_boxplot(y_true, y_score, class_names, output_dir):
    """Plot AUC boxplot (bootstrap distribution per class)."""
    num_classes = len(class_names)
    y_onehot = label_binarize(y_true, classes=list(range(num_classes)))
    n = len(y_true)
    rng = np.random.RandomState(42)

    boot_aucs = {i: [] for i in range(num_classes)}
    for _ in range(1000):
        idx = rng.randint(0, n, n)
        for i in range(num_classes):
            try:
                boot_aucs[i].append(roc_auc_score(y_onehot[idx][:, i], y_score[idx][:, i]))
            except Exception:
                pass

    # Build dataframe
    rows = []
    for i in range(num_classes):
        for v in boot_aucs[i]:
            rows.append({"Class": class_names[i], "AUC": v})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 5))
    positions = range(num_classes)
    for i, cls_name in enumerate(class_names):
        vals = df[df["Class"] == cls_name]["AUC"].values
        if len(vals) == 0:
            continue
        bp = ax.boxplot(vals, positions=[i], widths=0.4, showcaps=True, showfliers=False, patch_artist=True)
        for box in bp["boxes"]:
            box.set_facecolor(MACARON_LIGHT[i % len(MACARON_LIGHT)])
            box.set_alpha(0.7)
            box.set_edgecolor(MACARON_DARK[i % len(MACARON_DARK)])
        for whisker in bp["whiskers"]:
            whisker.set_color(MACARON_DARK[i % len(MACARON_DARK)])
        for cap in bp["caps"]:
            cap.set_color(MACARON_DARK[i % len(MACARON_DARK)])
        for median in bp["medians"]:
            median.set_color(MACARON_DARK[i % len(MACARON_DARK)])

    ax.set_xticks(list(positions))
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_ylabel("AUC")
    ax.set_title("AUC Distribution (Bootstrap)")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    sns.despine()
    plt.tight_layout()
    _save_fig(plt, os.path.join(output_dir, "auc_boxplot.png"))
    plt.close()


def plot_f1_violin(y_true, y_pred, class_names, output_dir):
    """Plot F1 violin plot (bootstrap distribution per class)."""
    num_classes = len(class_names)
    n = len(y_true)
    rng = np.random.RandomState(42)

    boot_f1s = {i: [] for i in range(num_classes)}
    for _ in range(1000):
        idx = rng.randint(0, n, n)
        boot_true = y_true[idx]
        boot_pred = y_pred[idx]
        for i in range(num_classes):
            true_c = (boot_true == i).astype(int)
            pred_c = (boot_pred == i).astype(int)
            boot_f1s[i].append(f1_score(true_c, pred_c, zero_division=0))

    # Build dataframe
    rows = []
    for i in range(num_classes):
        for v in boot_f1s[i]:
            rows.append({"Class": class_names[i], "F1": v})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 5))
    palette = {cls: MACARON_LIGHT[i % len(MACARON_LIGHT)] for i, cls in enumerate(class_names)}
    edge_colors = {cls: MACARON_DARK[i % len(MACARON_DARK)] for i, cls in enumerate(class_names)}

    sns.violinplot(data=df, x="Class", y="F1", order=class_names, inner=None, cut=0,
                   linewidth=0.8, palette=palette, saturation=1.0, ax=ax)

    # Color edges
    violin_bodies = [c for c in ax.collections if isinstance(c, PolyCollection)]
    for idx, body in enumerate(violin_bodies[:num_classes]):
        body.set_alpha(0.65)
        body.set_edgecolor(edge_colors[class_names[idx]])
        body.set_linewidth(0.9)

    # Mark median and mean
    means = df.groupby("Class")["F1"].mean().reindex(class_names).values
    medians = df.groupby("Class")["F1"].median().reindex(class_names).values
    for i, cls_name in enumerate(class_names):
        c_dark = edge_colors[cls_name]
        ax.plot(i, medians[i], marker="_", color=c_dark, markeredgewidth=1.8, markersize=10, zorder=3)
        ax.plot(i, means[i], marker="D", color=c_dark, markerfacecolor=c_dark, markersize=4, alpha=0.4, zorder=4)

    ax.set_xlabel("True labels")
    ax.set_ylabel("F1-Score")
    ax.set_title("F1-Score Distribution (Bootstrap)")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    sns.despine()

    legend_elements = [
        Line2D([0], [0], marker="_", color="w", markeredgecolor="#555", markeredgewidth=1.8, markersize=10, label="Median"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#555", markeredgecolor="#555", alpha=0.6, markersize=5, label="Mean"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9)
    plt.tight_layout()
    _save_fig(plt, os.path.join(output_dir, "f1_violin.png"))
    plt.close()


# ============================================================
#  Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Six-class evaluation")
    parser.add_argument("--mode", choices=["train", "predict"], required=True,
                        help="train: ROC + CM (2 plots); predict: ROC + CM + AUC box + F1 violin + Excel (4 plots + doc)")
    parser.add_argument("--input", "-i", required=True, help="Prediction CSV/Excel file")
    parser.add_argument("--output_dir", "-o", default=None, help="Output directory")
    parser.add_argument("--n_bootstrap", type=int, default=1000, help="Bootstrap iterations")
    parser.add_argument("--class_names", nargs="+", default=None, help="Custom class names")
    args = parser.parse_args()

    class_names = args.class_names or DEFAULT_CLASS_NAMES

    # Read data
    input_path = args.input
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(input_path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(input_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    output_dir = args.output_dir or os.path.join(os.path.dirname(input_path), f"six_class_{args.mode}")
    os.makedirs(output_dir, exist_ok=True)

    # Detect probability columns
    prob_cols = sorted([c for c in df.columns if c.startswith("class_") and c.endswith("_prob")],
                       key=lambda x: int(x.split("_")[1]))
    if not prob_cols:
        prob_cols = sorted([c for c in df.columns if c.startswith("prob_class_")],
                           key=lambda x: int(x.split("_")[2]))
    if not prob_cols:
        raise ValueError("No probability columns found")

    y_true = df["true_label"].astype(int).values
    y_pred = df["predicted_label"].astype(int).values
    y_score = df[prob_cols].values

    print(f"\n[Mode: {args.mode}] Evaluating {len(y_true)} samples, {len(class_names)} classes")

    # ---- Always: ROC + Confusion Matrix ----
    print("\n1. ROC Curve:")
    roc_auc_val, ci_micro, ci_macro = plot_roc(y_true, y_score, class_names, output_dir, args.n_bootstrap)

    print("\n2. Confusion Matrix:")
    cm = plot_confusion_matrix(y_true, y_pred, class_names, output_dir)

    # ---- Predict mode only: AUC boxplot + F1 violin + Excel ----
    if args.mode == "predict":
        print("\n3. AUC Boxplot:")
        plot_auc_boxplot(y_true, y_score, class_names, output_dir)

        print("\n4. F1 Violin Plot:")
        plot_f1_violin(y_true, y_pred, class_names, output_dir)

        # Save result document
        result_df = df.copy()
        result_df.to_excel(os.path.join(output_dir, "prediction_results.xlsx"), index=False)
        print(f"\n[OK] Result document saved")

    print(f"\n[OK] All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
