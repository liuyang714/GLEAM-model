"""
Shared utilities: random seed, loss functions, ROC plotting, evaluation metrics
"""
from __future__ import annotations

import os
import random
import warnings
from typing import Any, Sequence

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
#  Random Seed
# ============================================================
def set_seed(seed: int = 1208) -> None:
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader worker seed setting"""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
#  Loss Functions
# ============================================================
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        elif self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss


# ============================================================
#  ROC Plotting (with Bootstrap 95% CI)
# ============================================================
def plot_multiclass_roc(
    all_labels: Sequence[int],
    all_logits: np.ndarray,
    class_names: list[str],
    save_path: str,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> None:
    """
    Plot multiclass ROC curves with micro/macro averages and 95% CI.

    Parameters
    ----------
    all_labels : ground truth labels
    all_logits : model output probabilities/scores, shape [N, C]
    class_names : list of class names
    save_path : PNG save path
    n_bootstrap : number of bootstrap iterations (default 1000)
    seed : random seed
    """
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)
    C = all_logits.shape[1]
    N = len(all_labels)
    assert C == len(class_names), "class_names length must match logits columns"

    y_onehot = label_binarize(all_labels, classes=list(range(C)))
    if y_onehot.shape[1] != C:
        tmp = np.zeros((y_onehot.shape[0], C))
        tmp[:, : y_onehot.shape[1]] = y_onehot
        y_onehot = tmp

    # ---- Compute ROC ----
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(C):
        try:
            fpr[i], tpr[i], _ = roc_curve(y_onehot[:, i], all_logits[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        except Exception:
            fpr[i], tpr[i] = np.array([0, 1]), np.array([0, 1])
            roc_auc[i] = float("nan")

    try:
        fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot.ravel(), all_logits.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    except Exception:
        fpr["micro"], tpr["micro"] = np.array([0, 1]), np.array([0, 1])
        roc_auc["micro"] = float("nan")

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(C)]))
    mean_tpr = np.concatenate(
        [np.interp(all_fpr, fpr[i], tpr[i]).reshape(-1, 1) for i in range(C)], axis=1
    ).mean(1)
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    try:
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    except Exception:
        roc_auc["macro"] = float("nan")

    # ---- Bootstrap 95% CI ----
    rng = np.random.RandomState(seed)
    boot_aucs: dict[str, list[float]] = {i: [] for i in range(C)}
    boot_aucs["micro"] = []
    boot_aucs["macro"] = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, N, N)
        y_b = y_onehot[idx]
        logits_b = all_logits[idx]

        for i in range(C):
            try:
                boot_aucs[i].append(roc_auc_score(y_b[:, i], logits_b[:, i]))
            except Exception:
                boot_aucs[i].append(np.nan)

        try:
            boot_aucs["micro"].append(roc_auc_score(y_b.ravel(), logits_b.ravel()))
        except Exception:
            boot_aucs["micro"].append(np.nan)

        per_class = []
        for i in range(C):
            try:
                per_class.append(roc_auc_score(y_b[:, i], logits_b[:, i]))
            except Exception:
                per_class.append(np.nan)
        per_class_arr = np.array(per_class)
        boot_aucs["macro"].append(
            np.nanmean(per_class_arr) if not np.all(np.isnan(per_class_arr)) else np.nan
        )

    ci_dict = {}
    for k, vals in boot_aucs.items():
        vals_arr = np.array(vals)
        vals_arr = vals_arr[~np.isnan(vals_arr)]
        if len(vals_arr) == 0:
            ci_dict[k] = (float("nan"), float("nan"))
        else:
            ci_dict[k] = (np.percentile(vals_arr, 2.5), np.percentile(vals_arr, 97.5))

    # ---- Plot ----
    plt.figure(figsize=(8, 7))
    if not np.isnan(roc_auc.get("micro", np.nan)):
        ci = ci_dict["micro"]
        label = f"micro-average (AUC={roc_auc['micro']:.3f} [{ci[0]:.3f}-{ci[1]:.3f}])"
        plt.plot(fpr["micro"], tpr["micro"], lw=2, label=label)
    if not np.isnan(roc_auc.get("macro", np.nan)):
        ci = ci_dict["macro"]
        label = f"macro-average (AUC={roc_auc['macro']:.3f} [{ci[0]:.3f}-{ci[1]:.3f}])"
        plt.plot(fpr["macro"], tpr["macro"], lw=2, label=label)

    for i in range(C):
        ci_low, ci_high = ci_dict[i]
        if np.isnan(roc_auc[i]):
            label = f"{class_names[i]} (AUC=nan)"
        else:
            label = f"{class_names[i]} (AUC={roc_auc[i]:.3f} [{ci_low:.3f}-{ci_high:.3f}])"
        plt.plot(fpr[i], tpr[i], lw=1, label=label)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve with 95% CI (bootstrap)")
    plt.legend(loc="lower right", fontsize="small")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ============================================================
#  Evaluation
# ============================================================
def evaluate(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    num_classes: int,
    class_names: list[str] | None = None,
    save_csv: str | None = None,
    dataset: Any = None,
) -> dict[str, Any]:
    """
    Unified evaluation function, returns dict with all metrics and raw predictions.

    Returns
    -------
    dict with keys: acc, auc, f1, cm, labels, preds, probs, pids
    """
    model.eval()
    all_labels, all_preds, all_probs, all_pids = [], [], [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # Support different DataLoader output formats
            if isinstance(batch, dict):
                pids = batch.get("pid", None)
                inputs = {k: v.to(device) for k, v in batch.items() if k not in ("label", "pid")}
                labels = batch["label"].to(device)
                logits = model(**inputs)
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                Xs, ys = batch
                if ys.dim() == 0:
                    ys = ys.unsqueeze(0)
                ys = ys.to(device)
                logits = model(Xs)
                labels = ys
                pids = None
            else:
                raise ValueError(f"Unsupported batch format: {type(batch)}")

            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

            if pids is not None:
                all_pids.extend(pids)
            elif dataset is not None and hasattr(dataset, "ids"):
                start = batch_idx * (loader.batch_size or 1)
                end = start + len(labels)
                all_pids.extend(dataset.ids[start:end])
            else:
                all_pids.extend([None] * len(labels))

    # ---- Metrics ----
    acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    try:
        auc_score = roc_auc_score(all_labels, np.array(all_probs), multi_class="ovr")
    except Exception:
        auc_score = float("nan")
    _, _, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    # ---- Save CSV ----
    if save_csv and class_names:
        rows = []
        for pid, lab, pred, prob in zip(all_pids, all_labels, all_preds, all_probs):
            row = {"ID": pid, "true_label": int(lab), "predicted_label": int(pred)}
            for i in range(num_classes):
                row[f"prob_class_{i}"] = float(prob[i])
            rows.append(row)
        pd.DataFrame(rows).to_csv(save_csv, index=False, encoding="utf-8-sig")

    return {
        "acc": acc,
        "auc": auc_score,
        "f1": f1,
        "cm": cm,
        "labels": all_labels,
        "preds": all_preds,
        "probs": all_probs,
        "pids": all_pids,
    }
