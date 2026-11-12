"""
Fraud Detection System (FDS) - Model Evaluation Metrics
Author: Alpha
Description: Imbalanced classification metrics suite (PR-AUC, ROC-AUC, Recall,
             Precision, F1, Confusion Matrix) tailored for anomaly detection.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_fraud_metrics(
    y_true: np.ndarray | list,
    risk_scores: np.ndarray | list,
    threshold: float = 65.0,
) -> Dict[str, Any]:
    """
    Computes production evaluation metrics on imbalanced fraud datasets.
    Accuracy is omitted from primary evaluation as it is deceptive on 99.5% negative data.
    """
    y_t = np.asarray(y_true).astype(int)
    scores = np.asarray(risk_scores).astype(float)
    y_pred = (scores >= threshold).astype(int)

    # Confusion matrix extraction
    cm = confusion_matrix(y_t, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Scores normalized to [0, 1] for AUC calculations
    norm_scores = scores / 100.0

    # Area Under Precision-Recall Curve (PR-AUC is the gold standard for rare fraud detection)
    try:
        pr_auc = float(average_precision_score(y_t, norm_scores))
    except Exception:
        pr_auc = 0.0

    # Area Under ROC Curve
    try:
        roc_auc = float(roc_auc_score(y_t, norm_scores))
    except Exception:
        roc_auc = 0.0

    precision = float(precision_score(y_t, y_pred, zero_division=0))
    recall = float(recall_score(y_t, y_pred, zero_division=0))
    f1 = float(f1_score(y_t, y_pred, zero_division=0))

    false_alarm_rate = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    missed_fraud_rate = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    metrics = {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "threshold_used": round(threshold, 2),
        "total_samples": int(len(y_t)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "false_alarm_rate": round(false_alarm_rate, 4),
        "missed_fraud_rate": round(missed_fraud_rate, 4),
    }
    return metrics


def print_metrics_report(metrics: Dict[str, Any], model_name: str = "Isolation Forest") -> None:
    """Prints a structured performance scorecard in console."""
    print(f"\n==================== {model_name} Performance ====================")
    print(f" PR-AUC:               {metrics['pr_auc']:.4f}  (Gold standard for rare fraud)")
    print(f" ROC-AUC:              {metrics['roc_auc']:.4f}")
    print(f" Precision:            {metrics['precision']:.4f}")
    print(f" Recall:               {metrics['recall']:.4f}")
    print(f" F1-Score:             {metrics['f1']:.4f}")
    print(f" Decision Threshold:   {metrics['threshold_used']}")
    print("---------------------------------------------------------------")
    print(f" Confusion Matrix:     TP={metrics['true_positives']} | FP={metrics['false_positives']}")
    print(f"                       FN={metrics['false_negatives']} | TN={metrics['true_negatives']}")
    print(f" False Alarm Rate:     {metrics['false_alarm_rate']*100:.2f}%")
    print(f" Missed Fraud Rate:    {metrics['missed_fraud_rate']*100:.2f}%")
    print("===============================================================\n")
