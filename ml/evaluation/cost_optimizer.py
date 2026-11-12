"""
Fraud Detection System (FDS) - Cost-Based Threshold Optimizer
Author: Alpha
Description: Tunes anomaly decision threshold by minimizing asymmetric financial
             misclassification loss (Cost of Missed Fraud >> Cost of False Alarm).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


class CostBasedThresholdOptimizer:
    """
    Optimizes risk score threshold to minimize total economic impact.

    Mathematical Basis:
    ------------------
    Cost(T) = (False Negatives * Cost_FN) + (False Positives * Cost_FP)
    Where Cost_FN represents average direct fraud chargeback + operational fine (~$450),
    and Cost_FP represents customer friction / analyst review cost (~$15).
    """

    def __init__(
        self,
        cost_fn: float = 450.0,  # Missed fraud loss
        cost_fp: float = 15.0,   # False alert investigation cost
        cost_tp: float = 2.0,    # Prevention & notification overhead
    ) -> None:
        self.cost_fn = cost_fn
        self.cost_fp = cost_fp
        self.cost_tp = cost_tp

    def optimize(
        self,
        y_true: np.ndarray | list,
        risk_scores: np.ndarray | list,
        search_range: Tuple[float, float] = (30.0, 90.0),
        step: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Scans candidate thresholds and returns the economically optimal decision point.
        """
        y_t = np.asarray(y_true).astype(int)
        scores = np.asarray(risk_scores).astype(float)

        best_threshold = 65.0
        min_total_cost = float("inf")
        best_stats = {}

        candidate_thresholds = np.arange(search_range[0], search_range[1] + step, step)
        total_frauds = int(np.sum(y_t == 1))
        unmitigated_cost = total_frauds * self.cost_fn

        for thresh in candidate_thresholds:
            y_pred = (scores >= thresh).astype(int)

            tp = int(np.sum((y_t == 1) & (y_pred == 1)))
            fp = int(np.sum((y_t == 0) & (y_pred == 1)))
            fn = int(np.sum((y_t == 1) & (y_pred == 0)))
            tn = int(np.sum((y_t == 0) & (y_pred == 0)))

            total_cost = (fn * self.cost_fn) + (fp * self.cost_fp) + (tp * self.cost_tp)

            if total_cost < min_total_cost:
                min_total_cost = total_cost
                best_threshold = round(float(thresh), 2)
                best_stats = {
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "precision": round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0,
                    "recall": round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0,
                }

        cost_saved = unmitigated_cost - min_total_cost
        roi_percentage = (cost_saved / unmitigated_cost * 100.0) if unmitigated_cost > 0 else 0.0

        return {
            "optimal_threshold": best_threshold,
            "min_total_cost": round(min_total_cost, 2),
            "unmitigated_cost_no_model": round(unmitigated_cost, 2),
            "net_cost_savings": round(cost_saved, 2),
            "roi_percentage": round(roi_percentage, 2),
            "cost_assumptions": {
                "cost_fn": self.cost_fn,
                "cost_fp": self.cost_fp,
                "cost_tp": self.cost_tp,
            },
            "confusion_at_optimal": best_stats,
        }
