from __future__ import annotations

import pandas as pd
import pytest

from marketify.models.rnn_model import (TECHNICAL_FEATURE_COLUMNS,
                                        build_aligned_labels,
                                        make_recurrent_model)
from scripts.compare_models import (build_recurrent_leaderboard_markdown,
                                    prepare_validation_only_data)
from scripts.run_exit_rule_experiment import ValidationSplitMeta


def _make_feature_frame(rows: int = 1000) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=rows, freq="5min")
    base = pd.DataFrame(index=idx)
    base["Open"] = 100.0 + (pd.Series(range(rows), index=idx) * 0.01)
    base["High"] = base["Open"] + 0.2
    base["Low"] = base["Open"] - 0.2
    base["Close"] = base["Open"] + 0.05
    base["Volume"] = 1_000 + pd.Series(range(rows), index=idx)
    base["vol_20"] = 0.01
    for pos, col in enumerate(TECHNICAL_FEATURE_COLUMNS):
        if col in base.columns:
            continue
        base[col] = (pos + 1) * 0.001
    return base


def _make_split_meta() -> ValidationSplitMeta:
    return ValidationSplitMeta(
        train_end=700,
        val_end=850,
        test_start=850,
        validation_start_ts="2026-01-01 00:00:00",
        validation_end_ts="2026-01-02 00:00:00",
        test_start_ts="2026-01-03 00:00:00",
        test_split_touched=False,
    )


def test_label_horizon_matches_hold_horizon():
    idx = pd.date_range("2026-01-01", periods=5, freq="5min")
    frame = pd.DataFrame({"Close": [100.0, 105.0, 110.0, 115.0, 120.0]}, index=idx)

    labels = build_aligned_labels(frame, hold_horizon_bars=2, mode="return")

    assert labels.name == "target_h2_return"
    assert labels.iloc[0] == pytest.approx(0.10)
    assert labels.iloc[1] == pytest.approx(115.0 / 105.0 - 1.0)


def test_gru_and_lstm_train_pipeline_import_cleanly():
    from marketify.models import train_pipeline
    from scripts import train_recurrent_baseline

    gru = make_recurrent_model("gru")
    lstm = make_recurrent_model("lstm")

    assert hasattr(train_pipeline, "train_recurrent_baselines")
    assert hasattr(train_recurrent_baseline, "main")
    assert gru.architecture == "gru"
    assert lstm.architecture == "lstm"


def test_compare_models_does_not_touch_final_test_split():
    frame = _make_feature_frame()

    train_val, validation_index, meta, _feature_cols, target_col = prepare_validation_only_data(
        frame,
        hold_horizon_bars=48,
        label_mode="return",
    )

    test_start_ts = pd.Timestamp(meta.test_start_ts)
    assert target_col == "target_h48_return"
    assert meta.test_split_touched is False
    assert train_val.index.max() < test_start_ts
    assert validation_index.max() < test_start_ts


def test_recurrent_report_never_claims_guaranteed_profit():
    df = pd.DataFrame(
        [
            {
                "model_name": "gru",
                "hold_horizon_bars": 48,
                "label_mode": "return",
                "regime_filter": "skip_high_vol",
                "high_vol_confidence_threshold": 0.8,
                "weekly_return_pct": 0.42,
                "sharpe": 0.2,
                "max_drawdown_pct": 0.5,
                "trade_count": 20,
                "win_rate_pct": 45.0,
                "profit_factor": 0.95,
                "expectancy": -0.1,
                "baseline_pass": False,
                "weekly_target_pass": False,
                "test_split_touched": False,
                "rejected": False,
                "reject_reason": "",
            }
        ]
    )

    md = build_recurrent_leaderboard_markdown(df, _make_split_meta(), "skip_high_vol", "return")

    assert "guaranteed profit" not in md.lower()
    assert "No profit promise. Outcomes remain uncertain." in md
    assert "baseline comparison: FAIL" in md
    assert "Paper mode only. No leverage increase used." in md