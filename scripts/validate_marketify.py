"""Unified validation runner for Marketify paper-trading engine.

Runs import smoke, pytest, audit/risk log checks, drift checks.
Outputs reports/validation_summary.json and reports/validation_summary.md
"""
from __future__ import annotations

import json
import subprocess
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPORTS_DIR = Path("reports")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_import_smoke() -> dict[str, Any]:
    """Verify all core modules import without error."""
    errors: list[str] = []
    modules = [
        "marketify.config",
        "marketify.broker.paper",
        "marketify.broker.ibkr",
        "marketify.risk.risk_engine",
        "marketify.state.store",
        "marketify.models.xgb_model",
        "marketify.models.ensemble",
        "marketify.models.drift",
        "marketify.models.train_pipeline",
        "marketify.backtest.benchmark",
        "marketify.backtest.simulator",
        "marketify.onboarding",
        "marketify.data.market_data",
        "marketify.features.technical",
    ]
    for mod in modules:
        try:
            __import__(mod)
        except Exception as e:
            errors.append(f"{mod}: {e}")
    return {"name": "import_smoke", "passed": len(errors) == 0, "errors": errors}


def run_pytest() -> dict[str, Any]:
    """Run pytest and capture result."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=short"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    return {
        "name": "pytest",
        "passed": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout[-2000:] if result.stdout else "",
        "stderr": result.stderr[-1000:] if result.stderr else "",
    }


def run_audit_smoke() -> dict[str, Any]:
    """Create a temp PaperBroker and verify audit/risk tables exist."""
    import tempfile
    from marketify.broker.paper import PaperBroker
    from marketify.config import BrokerConfig

    td = tempfile.mkdtemp()
    try:
        cfg = BrokerConfig(db_path=str(Path(td) / "test.db"))
        broker = PaperBroker(cfg)
        broker.append_audit("smoke_test", "ok")
        broker.append_risk_event("smoke_test", "info", "validation smoke")
        audit = broker.get_audit_log(limit=1)
        risk = broker.get_risk_events(limit=1)
        res = {
            "name": "audit_smoke",
            "passed": len(audit) == 1 and len(risk) == 1,
            "audit_count": len(audit),
            "risk_count": len(risk),
        }
        broker.conn.close()
        return res
    finally:
        import shutil
        shutil.rmtree(td, ignore_errors=True)


def run_drift_smoke() -> dict[str, Any]:
    """Run drift checker on synthetic data."""
    import numpy as np
    import pandas as pd
    from marketify.models.drift import check_prediction_drift

    baseline = pd.Series(np.random.normal(0, 0.01, 100))
    recent = pd.Series(np.random.normal(0.02, 0.01, 30))
    result = check_prediction_drift(recent, baseline)
    return {
        "name": "drift_smoke",
        "passed": result.level in ("none", "warning", "critical"),
        "drift_level": result.level,
    }

def run_benchmark_smoke() -> dict[str, Any]:
    """Verify benchmark calculator runs."""
    from marketify.backtest.benchmark import compute_benchmark as compare_to_benchmark
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range("2023-01-01", periods=10)
    equities = (1000 * (1 + pd.Series(np.random.normal(0.001, 0.01, 10))).cumprod()).tolist()
    equity_history = [{"ts": str(d), "equity": e} for d, e in zip(dates, equities)]
    
    try:
        compare_to_benchmark(equity_history)
        return {"name": "benchmark_smoke", "passed": True}
    except Exception as e:
        return {"name": "benchmark_smoke", "passed": False, "error": str(e)}

def run_all() -> dict[str, Any]:
    """Run all validations and return combined report."""
    results: list[dict[str, Any]] = []
    results.append(run_import_smoke())
    results.append(run_pytest())
    results.append(run_audit_smoke())
    results.append(run_drift_smoke())
    results.append(run_benchmark_smoke())

    all_passed = all(r["passed"] for r in results)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "all_passed": all_passed,
        "results": results,
    }
    return report


def save_reports(report: dict[str, Any]) -> tuple[Path, Path]:
    """Save validation report as JSON and Markdown."""
    _ensure_dir(REPORTS_DIR)
    json_path = REPORTS_DIR / "validation_summary.json"
    md_path = REPORTS_DIR / "validation_summary.md"

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    with open(md_path, "w") as f:
        f.write("# Marketify Validation Report\n\n")
        f.write(f"**Timestamp:** {report['timestamp']}\n")
        f.write(f"**Overall Status:** {'? PASSED' if report['all_passed'] else '? FAILED'}\n\n")
        f.write("## Check Details\n\n")
        for res in report["results"]:
            status = "?" if res["passed"] else "?"
            f.write(f"### {status} {res['name']}\n")
            if not res["passed"]:
                for k, v in res.items():
                    if k not in ["name", "passed"]:
                        f.write(f"- **{k}:** {v}\n")
            f.write("\n")

    return json_path, md_path


if __name__ == "__main__":
    print("Starting Marketify Validation...")
    report = run_all()
    j, m = save_reports(report)
    print(f"Validation complete. Overall: {'PASSED' if report['all_passed'] else 'FAILED'}")
    print(f"Reports saved to {j} and {m}")
    if not report["all_passed"]:
        # Print failures to stderr for visibility
        for res in report["results"]:
            if not res["passed"]:
                print(f"FAILED: {res['name']}", file=sys.stderr)
    sys.exit(0 if report["all_passed"] else 1)
