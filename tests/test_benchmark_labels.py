"""Tests for benchmark labelling — synthetic vs real."""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from marketify.backtest.benchmark import compute_benchmark

ROOT = Path(__file__).resolve().parent.parent


def test_synthetic_smoke_benchmark_labelled_synthetic():
    """step_benchmark() must label its report as synthetic."""
    sys.path.insert(0, str(ROOT))
    from scripts import colab_run_all as cra

    with tempfile.TemporaryDirectory() as tmpdir:
        reports = Path(tmpdir)
        with patch.object(cra, "REPORTS", reports):
            cra.step_benchmark()
        report = json.loads((reports / "smoke_benchmark_report.json").read_text())

    assert report["synthetic"] is True
    assert report["source"] == "synthetic_smoke_test"
    assert report["status"] == "PASS"


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
    assert "No synthetic replacement" in report["reason"]


def test_real_benchmark_requires_real_equity():
    """Real benchmark report must reject insufficient real equity history."""
    sys.path.insert(0, str(ROOT))
    import importlib

    rb_mod = importlib.import_module("scripts.run_backtest_and_benchmark")
    equity = pd.Series([10_000.0], index=pd.to_datetime(["2026-01-01T00:00:00"]))

    report = rb_mod._build_real_benchmark_report(
        equity=equity,
        trade_count=1,
        weekly_goal=0.01,
        compute_benchmark=compute_benchmark,
    )

    assert report["synthetic"] is False
    assert report["status"] == "FAIL"
    assert "2 real equity points" in report["reason"]


def test_benchmark_fail_does_not_make_runtime_fail():
    """_read_benchmark_status returns FAIL string; main() still exits 0 if runtime steps pass."""
    sys.path.insert(0, str(ROOT))
    from scripts import colab_run_all as cra

    # Write a FAIL benchmark report
    with tempfile.TemporaryDirectory() as tmpdir:
        reports = Path(tmpdir)
        rb = {
            "source": "real_backtest",
            "synthetic": False,
            "status": "FAIL",
            "weekly_return_pct": 0.5851,
            "daily_return_pct": 0.08,
            "max_drawdown_pct": 0.4065,
            "sharpe_ratio": 0.397,
            "cvar_95_pct": 0.1,
            "weekly_pass": False,
            "drawdown_pass": True,
            "sharpe_pass": False,
            "trade_count": 95,
        }
        (reports / "real_benchmark.json").write_text(json.dumps(rb))

        with patch.object(cra, "REPORTS", reports):
            status = cra._read_benchmark_status()

    assert status == "FAIL", f"expected FAIL got {status!r}"
    # Verify generate_next_status accepts FAIL benchmark without crashing
    with tempfile.TemporaryDirectory() as tmpdir2:
        reports2 = Path(tmpdir2)
        (reports2 / "real_benchmark.json").write_text(json.dumps(rb))
        (reports2 / "smoke_benchmark_report.json").write_text(json.dumps({
            "source": "synthetic_smoke_test",
            "synthetic": True,
            "status": "PASS",
            "weekly_return_pct": 3.0,
            "daily_return_pct": 0.4,
            "max_drawdown_pct": 0.1,
            "sharpe_ratio": 2.0,
            "cvar_95_pct": 0.05,
        }))
        with patch.object(cra, "REPORTS", reports2):
            path = cra.generate_next_status(
                {"requirements": 0, "pytest": 0},
                benchmark_status="FAIL",
                tuning_status="PASS",
            )
        content = path.read_text()
    assert "Runtime Status: PASS" in content
    assert "Smoke Benchmark Status: PASS" in content
    assert "Real Benchmark Status: FAIL" in content


def test_runtime_crash_exits_nonzero():
    """generate_next_status with a failed runtime step marks Runtime Status FAIL."""
    sys.path.insert(0, str(ROOT))
    from scripts import colab_run_all as cra

    with tempfile.TemporaryDirectory() as tmpdir:
        reports = Path(tmpdir)
        with patch.object(cra, "REPORTS", reports):
            path = cra.generate_next_status(
                {"requirements": 0, "pytest": 1},  # pytest failed = runtime crash
                benchmark_status="UNKNOWN",
                tuning_status="UNKNOWN",
            )
        content = path.read_text()
    assert "Runtime Status: FAIL" in content


def test_next_status_contains_both_statuses():
    """NEXT_STATUS.md must separate smoke and real benchmark sections."""
    sys.path.insert(0, str(ROOT))
    from scripts import colab_run_all as cra

    rb = {
        "source": "real_backtest", "synthetic": False, "status": "FAIL",
        "weekly_return_pct": 0.5851, "daily_return_pct": 0.08,
        "max_drawdown_pct": 0.4065, "sharpe_ratio": 0.397,
        "cvar_95_pct": 0.1, "weekly_pass": False, "drawdown_pass": True,
        "sharpe_pass": False, "trade_count": 95,
    }
    sb = {
        "source": "synthetic_smoke_test", "synthetic": True, "status": "PASS",
        "weekly_return_pct": 3.0, "daily_return_pct": 0.4,
        "max_drawdown_pct": 0.1, "sharpe_ratio": 2.0,
        "cvar_95_pct": 0.05,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        reports = Path(tmpdir)
        (reports / "real_benchmark.json").write_text(json.dumps(rb))
        (reports / "smoke_benchmark_report.json").write_text(json.dumps(sb))
        with patch.object(cra, "REPORTS", reports):
            path = cra.generate_next_status(
                {"requirements": 0, "pytest": 0},
                benchmark_status="FAIL",
                tuning_status="PASS",
            )
        content = path.read_text()

    assert "Runtime Status:" in content
    assert "Smoke Benchmark Status:" in content
    assert "Real Benchmark Status:" in content
    assert "## Smoke Benchmark" in content
    assert "## Real Benchmark" in content
    assert "Synthetic: True" in content
    assert "Synthetic: False" in content
    assert "Trade count: 95" in content
    assert "1.0% target" in content
