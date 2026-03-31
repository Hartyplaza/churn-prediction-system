from __future__ import annotations

from typing import Any


def compute_selection_score(metrics: dict[str, Any]) -> float:
    if "f1_weighted" in metrics:
        score = float(metrics.get("f1_weighted", 0.0))
        score += 0.25 * float(metrics.get("quadratic_weighted_kappa", 0.0))
        score += 0.10 * float(metrics.get("f1_macro", 0.0))
        score -= 0.05 * float(metrics.get("ordinal_mae", 0.0))
        return score

    score = -float(metrics.get("rmse", 0.0))
    score -= 0.25 * float(metrics.get("mae", 0.0))
    score += 0.50 * float(metrics.get("r2", 0.0))
    return score
