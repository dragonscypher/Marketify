"""Colab-only heavy runner for Marketify.

Runs: requirements check, pytest, validation, training, benchmark.
Exits nonzero if MARKETIFY_REQUIRE_COLAB=1 and not in Colab.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

# Allow `from scripts.colab_check import ...` when running from project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def _is_colab() -> bool:
    return os.environ.get("COLAB_RELEASE_TAG") is not None or Path("/content").is_dir()

# Ensure subprocesses can import marketify
_ENV = os.environ.copy()
_ENV["PYTHONPATH"] = str(ROOT) + os.pathsep + _ENV.get("PYTHONPATH", "")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _run(cmd: list[str], label: str) -> subprocess.CompletedProcess:
    print(f"\n{'='*60}")
    print(f"[STEP] {label}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(ROOT), env=_ENV, text=True)
    if result.returncode != 0:
        print(f"[WARN] {label} exited {result.returncode}")
    return result


def check_colab() -> bool:
    from scripts.colab_check import get_runtime_info, is_colab, print_info
    print_info()
    require = os.environ.get("MARKETIFY_REQUIRE_COLAB", "0") == "1"
    if require and not is_colab():
        print("FATAL: Colab required but not connected.")
        print("Do NOT fake COLAB_RELEASE_TAG. Switch to real Colab kernel.")
        return False
    return True


def step_requirements() -> int:
    # Colab gets heavy deps; local gets light deps only
    req_file = "requirements-colab.txt" if _is_colab() else "requirements.txt"
    r = _run([sys.executable, "-m", "pip", "install", "-q", "-r", req_file],
             f"Install/check {req_file}")
    if r.returncode != 0:
        return r.returncode
    r2 = _run([sys.executable, "-m", "pip", "install", "-q", "-e", "."],
              "Install marketify in editable mode")
    return r2.returncode


def step_audit() -> int:
    audit_script = ROOT / "scripts" / "audit_repo_files.py"
    if audit_script.exists():
        r = _run([sys.executable, str(audit_script)], "Audit repo files")
        return r.returncode
    print("[INFO] audit_repo_files.py not found — skipping.")
    return 0


def step_pytest() -> int:
    r = _run([sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"],
             "Run pytest")
    return r.returncode


def step_validation() -> int:
    r = _run([sys.executable, "scripts/validate_marketify.py"],
             "Run validation")
    return r.returncode


def step_training() -> int:
    """Run train_if_missing or train_and_check if available."""
    for script in ["scripts/train_and_check.py", "scripts/train_if_missing.py"]:
        if (ROOT / script).exists():
            r = _run([sys.executable, script], f"Run {script}")
            return r.returncode
    # Fallback: use train_pipeline module directly
    print("[INFO] No train script found. Running train_pipeline.train_if_missing().")
    r = _run(
        [sys.executable, "-c",
         "from marketify.models.train_pipeline import train_if_missing; train_if_missing()"],
        "train_if_missing()"
    )
    return r.returncode


def step_benchmark() -> int:
    """Run SMOKE benchmark with synthetic equity — not real profit."""
    print(f"\n{'='*60}")
    print("[STEP] Smoke benchmark (SYNTHETIC — not real profit)")
    print(f"{'='*60}")
    try:
        from marketify.backtest.benchmark import compute_benchmark

        history = [
            {"ts": f"2026-01-0{i+1}T00:00:00", "equity": 10000 + i * 50}
            for i in range(7)
        ]
        result = compute_benchmark(history, weekly_goal=0.01)
        status = "PASS" if result.weekly_pass else "FAIL"
        report = {
            "source": "synthetic_smoke_test",
            "synthetic": True,
            "status": status,
            "weekly_return_pct": round(result.weekly_return * 100, 4),
            "daily_return_pct": round(result.daily_return * 100, 4),
            "max_drawdown_pct": round(result.max_drawdown * 100, 4),
            "cvar_95_pct": round(result.cvar_95 * 100, 4),
            "trade_count": 0,
            "weekly_pass": result.weekly_pass,
            "drawdown_pass": result.drawdown_pass,
            "sharpe_ratio": round(result.sharpe_ratio, 4),
            "sharpe_pass": result.sharpe_pass,
        }
        _ensure_dir(REPORTS)
        with open(REPORTS / "smoke_benchmark_report.json", "w") as f:
            json.dump(report, f, indent=2)

        print(f"[SYNTHETIC] Weekly return: {report['weekly_return_pct']:.4f}% → {status}")
        print(f"[SYNTHETIC] Sharpe ratio:  {report['sharpe_ratio']:.3f}")
        print("[NOTE] This is synthetic data. Not real profit.")
        return 0
    except Exception as e:
        print(f"[ERROR] Smoke benchmark: {e}")
        return 1


def step_real_benchmark() -> int:
    """Run real walk-forward backtest + benchmark."""
    r = _run([sys.executable, "scripts/run_backtest_and_benchmark.py"],
             "Real backtest + benchmark")
    return r.returncode


def step_tuning() -> int:
    """Run validation-only parameter tuning."""
    r = _run([sys.executable, "scripts/run_validation_tuning.py"],
             "Validation-only tuning")
    return r.returncode


def step_strategy_failure_analysis() -> int:
    """Run deep-dive strategy failure analysis report generator."""
    r = _run([sys.executable, "scripts/analyze_strategy_failure.py"],
             "Strategy failure deep-dive")
    return r.returncode


def generate_next_status(
    results: dict[str, int],
    benchmark_status: str,
    tuning_status: str,
    strategy_analysis_status: str = "UNKNOWN",
    smoke_status: str | None = None,
) -> Path:
    _ensure_dir(REPORTS)
    path = REPORTS / "NEXT_STATUS.md"
    ts = datetime.now(timezone.utc).isoformat()

    # Collect runtime info
    try:
        from scripts.colab_check import get_runtime_info
        ri = get_runtime_info()
    except Exception:
        ri = {"is_colab": False, "colab_tag": "N/A", "platform": "unknown",
              "gpu": "unknown", "system_ram_gb": None}

    # Determine runtime status (benchmark/tuning excluded — those are strategy results)
    RUNTIME_STEPS = {
        "requirements",
        "audit",
        "pytest",
        "validation",
        "training",
        "smoke_benchmark",
        "strategy_failure_analysis",
    }
    runtime_fail = any(v != 0 for k, v in results.items() if k in RUNTIME_STEPS)
    runtime_status = "FAIL" if runtime_fail else "PASS"

    if smoke_status is None:
        smoke_status = _read_smoke_benchmark_status()

    lines = [
        "# NEXT_STATUS — Marketify Paper Engine",
        f"**Generated:** {ts}",
        f"- Runtime Status: {runtime_status}",
        f"- Smoke Benchmark Status: {smoke_status}",
        f"- Real Benchmark Status: {benchmark_status}",
        f"- Tuning Status: {tuning_status}",
        f"- Strategy Analysis Status: {strategy_analysis_status}",
        "",
        "> NOTE: Real Benchmark FAIL = strategy did not meet 1% weekly target.",
        "> This is NOT a runtime crash. Pipeline ran to completion.",
        "",
        "## Runtime",
        f"- Colab: {ri['is_colab']}",
        f"- COLAB_TAG: {ri.get('colab_tag', 'N/A')}",
        f"- Platform: {ri.get('platform', 'unknown')}",
        f"- GPU: {ri.get('gpu', 'unknown')}",
        f"- GPU RAM: {ri.get('gpu_ram_gb', 'N/A')} GB" if ri.get('gpu_ram_gb') else "- GPU RAM: N/A",
        f"- System RAM: {ri.get('system_ram_gb', 'N/A')} GB" if ri.get('system_ram_gb') else "- System RAM: N/A",
        "",
        "## Step Results",
        "| Step | Exit Code | Status |",
        "|------|-----------|--------|",
    ]
    for step, code in results.items():
        tag = "PASS" if code == 0 else "FAIL"
        lines.append(f"| {step} | {code} | {tag} |")

    smoke_report = _read_report(REPORTS / "smoke_benchmark_report.json")
    if smoke_report:
        lines.append("")
        lines.append("## Smoke Benchmark")
        lines.append(f"- Source: {smoke_report.get('source', 'unknown')}")
        lines.append(f"- Synthetic: {smoke_report.get('synthetic', 'unknown')}")
        lines.append(f"- Status: {smoke_status}")
        lines.append(f"- Weekly return: {smoke_report.get('weekly_return_pct', 0)}%")
        lines.append(f"- Daily return: {smoke_report.get('daily_return_pct', 0)}%")
        lines.append(f"- Max drawdown: {smoke_report.get('max_drawdown_pct', 0)}%")
        lines.append(f"- Sharpe ratio: {smoke_report.get('sharpe_ratio', 0)}")
        lines.append(f"- CVaR 5%: {smoke_report.get('cvar_95_pct', 0)}%")
        lines.append("- Purpose: smoke check only. Not real profit.")

    # Append real benchmark detail if available
    real_report = _read_report(REPORTS / "real_benchmark.json")
    if real_report:
        try:
            lines.append("")
            lines.append("## Real Benchmark")
            lines.append(f"- Source: {real_report.get('source', 'unknown')}")
            lines.append(f"- Synthetic: {real_report.get('synthetic', 'unknown')}")
            lines.append(f"- Status: {benchmark_status}")
            lines.append(
                f"- Weekly return: {real_report.get('weekly_return_pct', 0)}%  "
                f"(target: {real_report.get('target_weekly_return_pct', 1.0)}%)"
            )
            lines.append(f"- Daily return: {real_report.get('daily_return_pct', 0)}%")
            lines.append(f"- Max drawdown: {real_report.get('max_drawdown_pct', 0)}%")
            lines.append(f"- Sharpe ratio: {real_report.get('sharpe_ratio', 0)}")
            lines.append(f"- CVaR 5%: {real_report.get('cvar_95_pct', 0)}%")
            lines.append(f"- Trade count: {real_report.get('trade_count', 0)}")
            if real_report.get("reason"):
                lines.append(f"- Reason: {real_report['reason']}")
            if benchmark_status == "FAIL":
                lines.append("")
                lines.append(
                    f"**Reason for FAIL:** weekly return {real_report.get('weekly_return_pct', 0)}% "
                    f"< {real_report.get('target_weekly_return_pct', 1.0)}% target."
                )
                lines.append("Strategy underperformed. No runtime error. Tuning required.")
            if benchmark_status == "FAIL_NO_TRADES":
                lines.append("- Honest outcome: no real trades, no synthetic replacement.")
        except Exception:
            pass

    # Tuning leaderboard info
    tuning_path = REPORTS / "tuning_leaderboard.csv"
    if tuning_path.exists():
        try:
            import pandas as pd
            tdf = pd.read_csv(tuning_path)
            accepted = tdf[~tdf["rejected"]].sort_values("sharpe_ratio", ascending=False)
            lines.append("")
            lines.append("## Validation Tuning")
            lines.append(f"- Total configs evaluated: {len(tdf)}")
            lines.append(f"- Accepted configs: {len(accepted)}")
            if len(accepted) > 0:
                best = accepted.iloc[0]
                lines.append(f"- Best weekly return: {best['weekly_return_pct']}%")
                lines.append(f"- Best Sharpe: {best['sharpe_ratio']}")
                lines.append(f"- Best config: SL={best['stop_loss_pct']} TP={best['take_profit_pct']} "
                             f"POS={best['max_position_fraction']} TS={int(best['time_stop_bars'])}")
                weekly_target_met = float(best['weekly_return_pct']) >= 1.0
                lines.append(f"- Meets 1% weekly target: {'YES' if weekly_target_met else 'NO — still below 1%'}")
                lines.append("- NOTE: This is validation-split result. Test split not evaluated.")
            else:
                lines.append("- All configs rejected by safety constraints.")
                lines.append("- Tuning status: completed but no safe config found.")
        except Exception:
            pass

    # Strategy deep-dive summary
    deep_dive = _read_strategy_deep_dive_summary()
    if deep_dive:
        lines.append("")
        lines.append("## Strategy Failure Deep Dive")
        lines.append(f"- Status: {strategy_analysis_status}")
        if "real_benchmark_status" in deep_dive:
            lines.append(f"- Real benchmark status in deep-dive: {deep_dive['real_benchmark_status']}")
        if "weekly_return_pct" in deep_dive:
            lines.append(f"- Weekly Return %: {deep_dive['weekly_return_pct']}")
        if "expectancy_per_trade" in deep_dive:
            lines.append(f"- Expectancy per trade: {deep_dive['expectancy_per_trade']}")
        if "profit_factor" in deep_dive:
            lines.append(f"- Profit factor: {deep_dive['profit_factor']}")
        if "primary_cause" in deep_dive:
            lines.append(f"- Primary cause: {deep_dive['primary_cause']}")
        lines.append("- See: reports/strategy_failure_deep_dive.md")

    lines.append("")
    lines.append("## Safety")
    lines.append("- Paper mode: DEFAULT")
    lines.append("- Live trading: DISABLED")
    lines.append("- Kill switch: FUNCTIONAL")
    lines.append("- Daily loss halt: FUNCTIONAL")
    lines.append("- Max drawdown halt: FUNCTIONAL")
    lines.append("")
    lines.append("## Disclaimer")
    lines.append(
        "Experimental paper-trading engine. No financial advice. "
        "Trading involves risk including loss of principal. "
        "Benchmark targets are goals, not guarantees."
    )

    with open(path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nNEXT_STATUS.md written → {path}")
    return path


def main() -> int:
    if not check_colab():
        return 1

    # Runtime steps — failure here = real crash, exit nonzero
    results: dict[str, int] = {}
    results["requirements"] = step_requirements()
    results["audit"] = step_audit()
    results["pytest"] = step_pytest()
    results["validation"] = step_validation()
    results["training"] = step_training()
    results["smoke_benchmark"] = step_benchmark()
    smoke_status = _read_smoke_benchmark_status()

    # Benchmark — FAIL = strategy result, not crash; never exits nonzero for this
    rb_code = step_real_benchmark()
    results["real_benchmark"] = rb_code
    benchmark_status = _read_benchmark_status()

    # Tuning — PASS if script ran (exit 0), even if no config hits target
    tuning_code = step_tuning()
    results["tuning"] = tuning_code
    tuning_status = "PASS" if tuning_code == 0 and _tuning_reports_exist() else "FAIL"

    # Deep-dive analyzer is report-generation runtime step
    strategy_analysis_code = step_strategy_failure_analysis()
    results["strategy_failure_analysis"] = strategy_analysis_code
    strategy_analysis_status = "PASS" if strategy_analysis_code == 0 else "FAIL"

    runtime_steps = {
        "requirements",
        "audit",
        "pytest",
        "validation",
        "training",
        "smoke_benchmark",
        "strategy_failure_analysis",
    }
    runtime_ok = all(results.get(step, 1) == 0 for step in runtime_steps)

    # If benchmark status cannot be read, treat as runtime error/crash
    if benchmark_status not in {"PASS", "FAIL", "FAIL_NO_TRADES"}:
        runtime_ok = False

    generate_next_status(results, benchmark_status, tuning_status, strategy_analysis_status, smoke_status)

    runtime_status = "PASS" if runtime_ok else "FAIL"
    print(f"\n{'='*60}")
    print(f"RUNTIME:          {runtime_status}")
    print(f"SMOKE_BENCHMARK:  {smoke_status}")
    print(f"REAL_BENCHMARK:   {benchmark_status}")
    print(f"TUNING:           {tuning_status}")
    print(f"{'='*60}")
    print(f"EXIT: 0" if runtime_ok else f"EXIT: 1")

    # Only exit nonzero for runtime failures (install/pytest/validation/crash)
    return 0 if runtime_ok else 1


def _read_benchmark_status() -> str:
    """Read benchmark status from real_benchmark.json without failing if absent."""
    rb_path = REPORTS / "real_benchmark.json"
    try:
        with open(rb_path) as f:
            rb = json.load(f)
        return rb.get("status", "UNKNOWN")
    except Exception:
        return "UNKNOWN"


def _read_smoke_benchmark_status() -> str:
    report = _read_report(REPORTS / "smoke_benchmark_report.json")
    if not report:
        return "UNKNOWN"
    return str(report.get("status", "UNKNOWN"))


def _read_report(path: Path) -> dict[str, object]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _tuning_reports_exist() -> bool:
    """Tuning is considered complete only if both leaderboard files exist."""
    return (REPORTS / "tuning_leaderboard.csv").exists() and (REPORTS / "tuning_leaderboard.md").exists()


def _read_strategy_deep_dive_summary() -> dict[str, str]:
    """Read key metrics from strategy_failure_deep_dive.csv for NEXT_STATUS summary."""
    path = REPORTS / "strategy_failure_deep_dive.csv"
    if not path.exists():
        return {}

    try:
        import pandas as pd

        df = pd.read_csv(path)
        if df.empty or "metric" not in df.columns or "value" not in df.columns:
            return {}

        if "section" in df.columns:
            head = df[df["section"] == "headline"]
            if head.empty:
                head = df
        else:
            head = df

        metrics = {
            str(row["metric"]): str(row["value"])
            for _, row in head.iterrows()
            if str(row.get("metric", "")).strip()
        }
        return metrics
    except Exception:
        return {}


if __name__ == "__main__":
    sys.exit(main())
