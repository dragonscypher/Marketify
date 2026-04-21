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


def test_benchmark_fail_does_not_make_runtime_fail():
    """_read_benchmark_status returns FAIL string; main() still exits 0 if runtime steps pass."""
    import json
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, MagicMock
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
        with patch.object(cra, "REPORTS", reports2):
            path = cra.generate_next_status(
                {"requirements": 0, "pytest": 0},
                benchmark_status="FAIL",
                tuning_status="PASS",
            )
        content = path.read_text()
    assert "Runtime Status: PASS" in content
    assert "Real Benchmark Status: FAIL" in content


def test_runtime_crash_exits_nonzero():
    """generate_next_status with a failed runtime step marks Runtime Status FAIL."""
    import tempfile
    from pathlib import Path
    from unittest.mock import patch
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
    """NEXT_STATUS.md must contain Runtime Status and Real Benchmark Status lines."""
    import json
    import tempfile
    from pathlib import Path
    from unittest.mock import patch
    sys.path.insert(0, str(ROOT))
    from scripts import colab_run_all as cra

    rb = {
        "source": "real_backtest", "synthetic": False, "status": "FAIL",
        "weekly_return_pct": 0.5851, "daily_return_pct": 0.08,
        "max_drawdown_pct": 0.4065, "sharpe_ratio": 0.397,
        "cvar_95_pct": 0.1, "weekly_pass": False, "drawdown_pass": True,
        "sharpe_pass": False, "trade_count": 95,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        reports = Path(tmpdir)
        (reports / "real_benchmark.json").write_text(json.dumps(rb))
        with patch.object(cra, "REPORTS", reports):
            path = cra.generate_next_status(
                {"requirements": 0, "pytest": 0},
                benchmark_status="FAIL",
                tuning_status="PASS",
            )
        content = path.read_text()

    assert "Runtime Status:" in content
    assert "Real Benchmark Status:" in content
    assert "FAIL" in content
    assert "1%" in content
