"""Evaluation metrics: AUC, precision, recall, F1, IoU."""
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, average_precision_score


def compute_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    return roc_auc_score(y_true, scores)


def compute_ap(y_true: np.ndarray, scores: np.ndarray) -> float:
    return average_precision_score(y_true, scores)


def compute_prf(y_true: np.ndarray, y_pred: np.ndarray):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    return {"precision": p, "recall": r, "f1": f}


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return float(intersection / union) if union > 0 else 0.0
