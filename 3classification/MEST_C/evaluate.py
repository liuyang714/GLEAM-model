"""
MEST-C evaluation script: ROC + confusion matrix only
Usage: python -m 3classification.MEST_C.evaluate --input /path/to/predictions.xlsx --output_dir /path/to/results
"""
from __future__ import annotations

import argparse
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from sklearn.metrics import classification_report

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import plot_multiclass_roc, plot_confusion_matrix


TASK_CONFIG = {
    "M": 2, "E": 2, "S": 2,  # binary
    "T": 3, "C": 3,           # multiclass
}


def evaluate_task(df: pd.DataFrame, task: str, output_dir: str, n_bootstrap: int = 1000):
    """Evaluate a single MEST-C task: ROC + confusion matrix only."""

    num_classes = TASK_CONFIG[task]
    true_col = f"{task}"
    pred_col = f"{task}_pred"
    prob_cols = sorted([c for c in df.columns if re.match(rf"^{task}\d+$", c)])

    if true_col not in df.columns or pred_col not in df.columns:
        print(f"[SKIP] Task {task}: columns not found")
        return

    if not prob_cols:
        print(f"[SKIP] Task {task}: no probability columns found")
        return

    df_task = df[[true_col, pred_col] + prob_cols].dropna().copy()
    y_true = df_task[true_col].astype(int).values
    y_pred = df_task[pred_col].astype(int).values
    y_score = df_task[prob_cols].values

    os.makedirs(output_dir, exist_ok=True)

    class_names = [f"{task}{i}" for i in range(num_classes)]

    # Classification Report
    print(f"\n[{task}] Classification Report:")
    print(classification_report(y_true, y_pred, labels=list(range(num_classes)),
                                target_names=class_names, digits=4, zero_division=0))

    # 1. ROC Curve
    plot_multiclass_roc(y_true, y_score, class_names,
                        os.path.join(output_dir, f"{task}_roc.png"),
                        n_bootstrap=n_bootstrap)

    # 2. Confusion Matrix
    plot_confusion_matrix(y_true, y_pred, class_names,
                          os.path.join(output_dir, f"{task}_confusion_matrix.png"))

    print(f"[{task}] Done. Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="MEST-C evaluation (ROC + confusion matrix)")
    parser.add_argument("--input", "-i", required=True, help="Prediction CSV/Excel file")
    parser.add_argument("--output_dir", "-o", default=None, help="Output directory")
    parser.add_argument("--n_bootstrap", type=int, default=1000, help="Bootstrap iterations")
    args = parser.parse_args()

    input_path = args.input
    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(input_path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(input_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    output_dir = args.output_dir or os.path.join(os.path.dirname(input_path), "mestc_results")
    os.makedirs(output_dir, exist_ok=True)

    for task in ["M", "E", "S", "T", "C"]:
        task_dir = os.path.join(output_dir, f"task_{task}")
        evaluate_task(df, task, task_dir, n_bootstrap=args.n_bootstrap)

    print(f"\n[OK] All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
