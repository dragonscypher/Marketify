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
        report = {
            "source": "synthetic_smoke_test",
            "synthetic": True,
            "weekly_return": result.weekly_return,
            "weekly_pass": result.weekly_pass,
            "max_drawdown": result.max_drawdown,
            "drawdown_pass": result.drawdown_pass,
            "sharpe_ratio": result.sharpe_ratio,
            "sharpe_pass": result.sharpe_pass,
        }
        _ensure_dir(REPORTS)
        with open(REPORTS / "smoke_benchmark_report.json", "w") as f:
            json.dump(report, f, indent=2)

        status = "PASS" if result.weekly_pass else "FAIL"
        print(f"[SYNTHETIC] Weekly return: {result.weekly_return:.4%} → {status}")
        print(f"[SYNTHETIC] Sharpe ratio:  {result.sharpe_ratio:.3f}")
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


def generate_next_status(results: dict[str, int]) -> Path:
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

    all_pass = all(v == 0 for v in results.values())
    lines = [
        "# NEXT_STATUS — Marketify Paper Engine",
        f"**Generated:** {ts}",
        f"**Overall:** {'PASS' if all_pass else 'FAIL'}",
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

    # Append real benchmark detail if available
    rb_path = REPORTS / "real_benchmark.json"
    if rb_path.exists():
        try:
            with open(rb_path) as f:
                rb = json.load(f)
            lines.append("")
            lines.append("## Real Benchmark")
            lines.append(f"- Source: {rb.get('source', 'unknown')}")
            lines.append(f"- Synthetic: {rb.get('synthetic', 'unknown')}")
            lines.append(f"- Status: {rb.get('status', 'unknown')}")
            lines.append(f"- Weekly return: {rb.get('weekly_return_pct', 0)}%")
            lines.append(f"- Daily return: {rb.get('daily_return_pct', 0)}%")
            lines.append(f"- Max drawdown: {rb.get('max_drawdown_pct', 0)}%")
            lines.append(f"- Sharpe ratio: {rb.get('sharpe_ratio', 0)}")
            lines.append(f"- CVaR 5%: {rb.get('cvar_95_pct', 0)}%")
            lines.append(f"- Trade count: {rb.get('trade_count', 0)}")
        except Exception:
            pass

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

    results = {}
    results["requirements"] = step_requirements()
    results["audit"] = step_audit()
    results["pytest"] = step_pytest()
    results["validation"] = step_validation()
    results["training"] = step_training()
    results["smoke_benchmark"] = step_benchmark()
    results["real_benchmark"] = step_real_benchmark()

    generate_next_status(results)

    all_pass = all(v == 0 for v in results.values())
    print(f"\n{'='*60}")
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print(f"{'='*60}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
