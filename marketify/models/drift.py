from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DriftResult:
    level: str  # "none" | "warning" | "critical"
    mae_delta: float
    feature_shift_z: float
    reason: str


def check_prediction_drift(
    recent_errors: pd.Series,
    baseline_errors: pd.Series,
    warning_threshold: float = 0.5,
    critical_threshold: float = 1.0,
) -> DriftResult:
    """Compare recent prediction errors against baseline window.

    Uses MAE delta normalised by baseline std as shift proxy.
    """
    if recent_errors.empty or baseline_errors.empty:
        return DriftResult(level="none", mae_delta=0.0, feature_shift_z=0.0, reason="insufficient_data")

    baseline_mae = float(np.mean(np.abs(baseline_errors)))
    recent_mae = float(np.mean(np.abs(recent_errors)))
    mae_delta = recent_mae - baseline_mae

    baseline_std = float(np.std(baseline_errors)) or 1e-9
    z_score = mae_delta / baseline_std

    if abs(z_score) >= critical_threshold:
        level = "critical"
        reason = f"MAE drift z={z_score:.2f} exceeds critical threshold {critical_threshold}"
    elif abs(z_score) >= warning_threshold:
        level = "warning"
        reason = f"MAE drift z={z_score:.2f} exceeds warning threshold {warning_threshold}"
    else:
        level = "none"
        reason = f"MAE drift z={z_score:.2f} within normal range"

    return DriftResult(level=level, mae_delta=mae_delta, feature_shift_z=z_score, reason=reason)


def check_feature_drift(
    recent_features: pd.DataFrame,
    baseline_features: pd.DataFrame,
    warning_threshold: float = 0.5,
    critical_threshold: float = 1.0,
) -> DriftResult:
    """Simple feature distribution shift via mean z-score across columns."""
    if recent_features.empty or baseline_features.empty:
        return DriftResult(level="none", mae_delta=0.0, feature_shift_z=0.0, reason="insufficient_data")

    z_scores = []
    for col in recent_features.columns:
        if col not in baseline_features.columns:
            continue
        baseline_mean = float(baseline_features[col].mean())
        baseline_std = float(baseline_features[col].std()) or 1e-9
        recent_mean = float(recent_features[col].mean())
        z_scores.append(abs((recent_mean - baseline_mean) / baseline_std))

    if not z_scores:
        return DriftResult(level="none", mae_delta=0.0, feature_shift_z=0.0, reason="no_common_features")

    avg_z = float(np.mean(z_scores))

    if avg_z >= critical_threshold:
        level = "critical"
        reason = f"Feature drift avg_z={avg_z:.2f} exceeds critical threshold {critical_threshold}"
    elif avg_z >= warning_threshold:
        level = "warning"
        reason = f"Feature drift avg_z={avg_z:.2f} exceeds warning threshold {warning_threshold}"
    else:
        level = "none"
        reason = f"Feature drift avg_z={avg_z:.2f} within normal range"

    return DriftResult(level=level, mae_delta=0.0, feature_shift_z=avg_z, reason=reason)
