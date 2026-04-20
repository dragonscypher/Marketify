"""Tests for model drift detection."""
import numpy as np
import pandas as pd

from marketify.models.drift import (
    DriftResult,
    check_feature_drift,
    check_prediction_drift,
)


def test_no_drift():
    baseline = pd.Series(np.random.normal(0, 0.01, 100))
    recent = pd.Series(np.random.normal(0, 0.01, 30))
    result = check_prediction_drift(recent, baseline)
    assert result.level == "none"


def test_warning_drift():
    baseline = pd.Series(np.random.normal(0, 0.01, 200))
    recent = pd.Series(np.random.normal(0.05, 0.01, 30))  # large shift
    result = check_prediction_drift(recent, baseline, warning_threshold=1.0, critical_threshold=5.0)
    assert result.level in ("warning", "critical")


def test_critical_drift():
    baseline = pd.Series(np.random.normal(0, 0.001, 200))
    recent = pd.Series(np.random.normal(0.05, 0.001, 30))
    result = check_prediction_drift(recent, baseline, warning_threshold=1.0, critical_threshold=2.0)
    assert result.level == "critical"


def test_feature_drift_none():
    cols = ["rsi_14", "macd", "bb_pct"]
    baseline = pd.DataFrame(np.random.normal(0, 1, (100, 3)), columns=cols)
    recent = pd.DataFrame(np.random.normal(0, 1, (30, 3)), columns=cols)
    result = check_feature_drift(recent, baseline)
    assert result.level == "none"


def test_feature_drift_shifted():
    cols = ["rsi_14", "macd", "bb_pct"]
    baseline = pd.DataFrame(np.random.normal(0, 0.01, (200, 3)), columns=cols)
    recent = pd.DataFrame(np.random.normal(5, 0.01, (30, 3)), columns=cols)
    result = check_feature_drift(recent, baseline, warning_threshold=1.0, critical_threshold=2.0)
    assert result.level == "critical"


def test_drift_result_dataclass():
    r = DriftResult(level="warning", mae_delta=0.005, feature_shift_z=1.5, reason="test")
    assert r.level == "warning"
    assert r.mae_delta == 0.005
