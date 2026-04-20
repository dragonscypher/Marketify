"""Tests for benchmark labelling — synthetic vs real."""
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from marketify.backtest.benchmark import compute_benchmark

ROOT = Path(__file__).resolve().parent.parent


def test_synthetic_smoke_benchmark_labelled_synthetic():
    """step_benchmark() must label its report as synthetic."""
    sys.path.insert(0, str(ROOT))
    from scripts.colab_run_all import step_benchmark, REPORTS

    step_benchmark()
    report_path = REPORTS / "smoke_benchmark_report.json"
    assert report_path.exists(), "smoke_benchmark_report.json not created"
    with open(report_path) as f:
        report = json.load(f)
    assert report["synthetic"] is True
    assert report["source"] == "synthetic_smoke_test"


def test_no_trades_fails_honestly():
    """run_backtest_and_benchmark must FAIL_NO_TRADES when no trades occur."""
    sys.path.insert(0, str(ROOT))
    import importlib
    rb_mod = importlib.import_module("scripts.run_backtest_and_benchmark")

    report = rb_mod._no_trade_report()
    assert report["synthetic"] is False
    assert report["status"] == "FAIL_NO_TRADES"
    assert report["trade_count"] == 0
    assert report["weekly_pass"] is False


def test_real_benchmark_requires_real_equity():
    """compute_benchmark on rising synthetic data should not be labelled real."""
    history = [
        {"ts": f"2026-01-0{i+1}T00:00:00", "equity": 10000 + i * 50}
        for i in range(7)
    ]
    result = compute_benchmark(history, weekly_goal=0.01)
    # compute_benchmark itself does not label source.
    # The caller is responsible. This test validates the contract:
    # result contains no "source" or "synthetic" field — callers must add them.
    assert not hasattr(result, "source")
    assert not hasattr(result, "synthetic")
    assert result.weekly_pass is True  # sanity
