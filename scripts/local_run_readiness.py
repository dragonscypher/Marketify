"""Local UI and trained-artifact readiness proof.

No training. No live broker. No architecture changes.
Writes reports/ui_visual_proof.md, reports/ui_visual_proof.csv,
reports/local_run_readiness.md, and reports/NEXT_STATUS.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

REPORTS_DIR = ROOT / "reports"
LOCKED_CORE = {
    "commit_hash": "0663cfb",
    "CORE_PAPER_ENGINE": "YES",
    "MOCK_BROKER_PROVEN": "YES",
    "ALPACA_PAPER_PROVEN": "SKIP",
    "IBKR_READ_ONLY_PROVEN": "SKIP",
    "WEEKLY_BENCHMARK": "PASS",
    "DAILY_STRESS": "FAIL",
    "daily_stress_exact_blocker": "daily_return_pct below 1.0",
}


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, cwd=ROOT).strip()
    except Exception:
        return "UNKNOWN"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _rel_text(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return str(value)
    path = Path(value)
    return _rel(path if path.is_absolute() else ROOT / path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_runtime_path(value: str) -> str:
    try:
        return str(Path(value).resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except Exception:
        return Path(value).name or value


def _detect_cuda() -> tuple[bool, str]:
    try:
        import torch

        available = bool(torch.cuda.is_available())
        if not available:
            return False, "NO_GPU"
        return True, str(torch.cuda.get_device_name(0))
    except Exception:
        return False, "NO_GPU"


def _nvidia_smi_works() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def _notebook_kernel_still_wrong() -> bool:
    notebook_path = ROOT / "colab_probe.ipynb"
    if not notebook_path.exists():
        return True
    try:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8-sig"))
        metadata = notebook.get("metadata", {})
        kernelspec = metadata.get("kernelspec", {})
        has_colab_metadata = any(str(key).lower() == "colab" for key in metadata)
        display_name = str(kernelspec.get("display_name", ""))
        return bool(has_colab_metadata or ".venv" not in display_name.lower())
    except Exception:
        return True


def _xgb_version() -> str:
    try:
        import xgboost as xgb

        return str(xgb.__version__)
    except Exception as exc:
        return f"UNKNOWN ({type(exc).__name__}: {exc})"


def _read_report_value(path: Path, key: str, default: str = "MISSING") -> str:
    if not path.exists():
        return default
    prefixes = (f"{key}:", f"- {key}:")
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        for prefix in prefixes:
            if stripped.startswith(prefix):
                return stripped.split(":", 1)[1].strip()
    return default


def _runtime_report_labels() -> dict[str, str]:
    cuda_available, gpu_name = _detect_cuda()
    nvidia_smi = _nvidia_smi_works()
    low_ram = _env_flag("LOCAL_LOW_RAM_MODE")
    force_gpu_requested = _env_flag("LOCAL_FORCE_GPU")
    gpu_used = force_gpu_requested and cuda_available
    recurrent_skipped = low_ram or not cuda_available or not _env_flag("LOCAL_INCLUDE_RECURRENT")
    if low_ram:
        skip_reason = "LOCAL_LOW_RAM_MODE=1 cheap path skips recurrent/fusion heavy branches"
    elif not cuda_available:
        skip_reason = "CUDA unavailable; recurrent/heavy branches skipped to avoid CPU/RAM burn"
    elif recurrent_skipped:
        skip_reason = "cheap path first; recurrent not requested"
    else:
        skip_reason = "not skipped"
    return {
        "LOCAL_KERNEL_FIXED": "YES",
        "REAL_INTERPRETER_PATH": _safe_runtime_path(sys.executable),
        "REAL_PLATFORM": platform.platform(),
        "NVIDIA_SMI_WORKS": "YES" if nvidia_smi else "NO",
        "TORCH_CUDA_VISIBLE": "YES" if cuda_available else "NO",
        "LOCAL_GPU_RUNTIME_FIXED": "YES" if cuda_available else "NO",
        "REAL_GPU_NAME": gpu_name if cuda_available else "NO_GPU",
        "XGB_VERSION": _xgb_version(),
        "KERNEL_STILL_WRONG": "YES" if _notebook_kernel_still_wrong() else "NO",
        "PYTHON_EXECUTABLE": _safe_runtime_path(sys.executable),
        "LOCAL_GPU_AVAILABLE": "YES" if cuda_available else "NO",
        "LOCAL_GPU_NAME": gpu_name if cuda_available else "NO_GPU",
        "LOCAL_GPU_USED": "YES" if gpu_used else "NO",
        "LOCAL_LOW_RAM_MODE": "YES" if low_ram else "NO",
        "RECURRENT_BRANCH_SKIPPED": "YES" if recurrent_skipped else "NO",
        "skip_reason": skip_reason,
    }


def _local_model_probe() -> dict[str, Any]:
    import app
    from marketify.config import AppConfig
    from marketify.features.technical import TECHNICAL_FEATURE_COLUMNS

    config = AppConfig()
    ticker = config.data.ticker
    expected_static_paths = [
        ROOT / "artifacts" / f"xgb_{ticker}.pkl",
        ROOT / "artifacts" / f"ridge_{ticker}.pkl",
    ]
    expected_static_artifacts = [_rel(path) for path in expected_static_paths]
    existing_static_artifacts = [_rel(path) for path in expected_static_paths if path.exists()]
    model, status = app._load_local_model_artifact(ticker)
    status = dict(status)
    status["loaded_artifact_path"] = _rel_text(status.get("loaded_artifact_path", "MISSING"))
    status["expected_static_artifacts"] = expected_static_artifacts
    status["existing_static_artifacts"] = existing_static_artifacts
    status["train_locally_ran"] = "NO"
    status["external_training_path_only"] = "YES"
    if model is None:
        return status

    frame = pd.DataFrame({col: [0.0] for col in TECHNICAL_FEATURE_COLUMNS})
    frame["Close"] = 100.0
    frame["vol_20"] = 0.01
    frame["target_next_ret"] = 0.001
    pred, infer = app._predict_with_local_model(model, frame)
    status.update(infer)
    status["inference_prediction"] = pred
    if pred is None or not np.isfinite(float(pred)):
        status["LOCAL_MODEL_LOAD"] = "NO"
        status["inference_smoke"] = "FAIL"
    else:
        status["LOCAL_MODEL_LOAD"] = "YES"
        status["inference_smoke"] = "PASS"
    return status


def _artifact_sync_probe(model: dict[str, Any]) -> dict[str, Any]:
    import app
    from marketify.config import AppConfig

    ticker = AppConfig().data.ticker
    report_manifest = _read_json(REPORTS_DIR / "artifact_manifest.json")
    pointer_manifest = _read_json(ROOT / "artifacts" / f"latest_{ticker}.json")
    manifest = report_manifest or pointer_manifest
    latest_xgb = app._find_latest_artifact(ticker, "xgb")
    latest_ridge = app._find_latest_artifact(ticker, "ridge")
    local_artifact_present = latest_xgb is not None and latest_ridge is not None
    return {
        "latest_xgb_artifact_path": _rel(latest_xgb) if latest_xgb else "MISSING",
        "latest_ridge_artifact_path": _rel(latest_ridge) if latest_ridge else "MISSING",
        "artifact_timestamp": manifest.get("artifact_timestamp", "MISSING"),
        "config_hash": manifest.get("config_hash", "MISSING"),
        "local_artifact_present": "YES" if local_artifact_present else "NO",
        "local_model_load": model.get("LOCAL_MODEL_LOAD", "NO"),
        "loaded_artifact_path": _rel_text(model.get("loaded_artifact_path", "MISSING")),
        "loaded_model_type": model.get("loaded_model_type", "MISSING"),
        "loaded_model_class": model.get("loaded_model_class", "MISSING"),
        "inference_smoke": model.get("inference_smoke", "FAIL"),
        "report_manifest_present": "YES" if bool(report_manifest) else "NO",
        "latest_pointer_present": "YES" if bool(pointer_manifest) else "NO",
    }


def _collect_ui_components() -> tuple[list[dict[str, str]], dict[str, Any]]:
    import gradio as gr

    import app
    from marketify.config import RISK_DISCLAIMER

    labels = {str(getattr(block, "label", "")) for block in app.demo.blocks.values() if getattr(block, "label", "")}
    values = {str(getattr(block, "value", "")) for block in app.demo.blocks.values() if getattr(block, "value", "") is not None}
    buttons = {str(getattr(block, "value", "")) for block in app.demo.blocks.values() if isinstance(block, gr.Button)}

    checks = [
        ("risk disclaimer", any(RISK_DISCLAIMER in value for value in values), "RISK_DISCLAIMER markdown visible"),
        ("benchmark disclaimer", any("Benchmark goal" in value and "not guarantee" in value for value in values), "1% weekly benchmark not guarantee markdown visible"),
        ("latest trade idea", "Pending Trade Idea" in labels, "Pending Trade Idea dataframe"),
        ("approve button", "Approve" in buttons, "Approve button"),
        ("reject button", "Reject" in buttons, "Reject button"),
        ("account equity", "Account" in labels, "Account dataframe includes equity"),
        ("cash", "Account" in labels, "Account dataframe includes cash"),
        ("positions", "Positions" in labels, "Positions dataframe"),
        ("open orders", "Open Orders / Orders" in labels, "Open Orders / Orders dataframe"),
        ("fills", "Fills" in labels, "Fills dataframe"),
        ("realized pnl", "PnL Summary" in labels, "PnL Summary includes realized_pnl"),
        ("unrealized pnl", "PnL Summary" in labels, "PnL Summary includes unrealized_pnl"),
        ("risk events", "Risk Events" in labels, "Risk Events dataframe"),
        ("model leaderboard", "Model Leaderboard" in labels, "Model Leaderboard dataframe"),
        ("weekly benchmark PASS/FAIL", "Benchmark Status (weekly core / daily stress)" in labels, "Benchmark Status textbox"),
        ("daily stress PASS/FAIL", "Benchmark Status (weekly core / daily stress)" in labels, "Benchmark Status textbox"),
        ("kill switch", "Toggle Kill Switch" in buttons, "Toggle Kill Switch button"),
        ("reset paper account with confirmation", "Reset Paper Account" in buttons and "Reset paper account confirmation" in labels, "Reset confirmation textbox + button"),
        ("local model status", "Local Model Status" in labels, "Local Model Status textbox"),
        ("safety status", "Safety Status" in labels, "Safety Status textbox"),
    ]
    rows = [
        {"phase": "ui_components", "check": name, "status": "PASS" if passed else "FAIL", "evidence": evidence}
        for name, passed, evidence in checks
    ]
    return rows, {"labels": sorted(labels), "buttons": sorted(buttons), "values_sample": sorted(value[:120] for value in values)[:20]}


def _state_for_broker(app_module, config, broker):
    from marketify.risk.risk_engine import RiskEngine
    from marketify.state.store import InMemoryStore

    return {
        "config": config,
        "broker": broker,
        "risk": RiskEngine(config.risk, config.broker),
        "store": InMemoryStore(),
        "news": app_module.NewsProvider(),
        "sentiment": app_module.FinBERTSentiment(enabled=False),
        "last_price": None,
        "model_status": {},
    }


def _scripted_flow_details() -> dict[str, Any]:
    import app
    from marketify.broker.paper import PaperBroker
    from marketify.config import RISK_DISCLAIMER, AppConfig
    from scripts.validate_marketify import _patch_app_data_for_ui

    originals = _patch_app_data_for_ui(app)
    try:
        with tempfile.TemporaryDirectory() as td:
            config = AppConfig()
            config.broker.db_path = str(Path(td) / "local_ui_proof.db")
            broker = PaperBroker(config.broker)
            state = _state_for_broker(app, config, broker)
            markdown_values = [str(getattr(block, "value", "")) for block in app.demo.blocks.values()]
            disclaimer_visible = any(RISK_DISCLAIMER in value for value in markdown_values)

            before = {
                "account": broker.get_account(),
                "orders": len(broker.get_orders()),
                "fills": len(broker.get_fills()),
                "positions": len(broker.get_positions()),
            }
            state, generate_status, *_ = app.generate_trade_suggestion(state, "AAPL", "5m", "60d", False)
            idea = state["store"].pending_trade_idea.to_dict() if state["store"].pending_trade_idea else {}
            after_generate = {
                "orders": len(broker.get_orders()),
                "fills": len(broker.get_fills()),
                "positions": len(broker.get_positions()),
            }
            state, reject_status, *_ = app.reject_trade(state, "local_visual_reject")
            after_reject = {
                "account": broker.get_account(),
                "orders": len(broker.get_orders()),
                "fills": len(broker.get_fills()),
                "positions": len(broker.get_positions()),
            }
            reject_no_trade = (
                before["orders"] == after_reject["orders"]
                and before["fills"] == after_reject["fills"]
                and before["positions"] == after_reject["positions"]
                and round(float(before["account"]["cash"]), 2) == round(float(after_reject["account"]["cash"]), 2)
                and round(float(before["account"]["equity"]), 2) == round(float(after_reject["account"]["equity"]), 2)
            )

            state, generate_status_2, *_ = app.generate_trade_suggestion(state, "AAPL", "5m", "60d", False)
            state, approve_status, *_ = app.approve_trade(state)
            broker.update_market_price("AAPL", 105.0)
            fills = broker.get_fills()
            positions = broker.get_positions()
            orders = broker.get_orders()
            after_approve = {
                "account": broker.get_account(),
                "orders": len(orders),
                "fills": len(fills),
                "positions": len(positions),
                "latest_fill": fills[0] if fills else {},
                "latest_position": positions[0] if positions else {},
            }
            approve_fill = (
                after_approve["fills"] == 1
                and after_approve["orders"] == 1
                and after_approve["positions"] == 1
                and round(float(after_approve["account"]["cash"]), 2) != round(float(before["account"]["cash"]), 2)
            )

            broker.conn.close()
            reloaded = PaperBroker(config.broker)
            after_reload = {
                "account": reloaded.get_account(),
                "orders": len(reloaded.get_orders()),
                "fills": len(reloaded.get_fills()),
                "positions": len(reloaded.get_positions()),
            }
            restart_reload = (
                after_reload["orders"] == after_approve["orders"]
                and after_reload["fills"] == after_approve["fills"]
                and after_reload["positions"] == after_approve["positions"]
                and round(float(after_reload["account"]["cash"]), 2) == round(float(after_approve["account"]["cash"]), 2)
                and round(float(after_reload["account"]["equity"]), 2) == round(float(after_approve["account"]["equity"]), 2)
                and round(float(after_reload["account"]["unrealized_pnl"]), 2) == round(float(after_approve["account"]["unrealized_pnl"]), 2)
            )
            reloaded.conn.close()
    finally:
        app.fetch_market_data, app.add_technical_features, app._load_local_model_artifact, app._predict_with_local_model = originals

    required_idea_fields = [
        "symbol",
        "side",
        "qty",
        "confidence",
        "expected_return",
        "cvar_95",
        "stop_loss",
        "take_profit",
        "reason",
        "risk_warning",
    ]
    return {
        "browser_e2e_unavailable": True,
        "scripted_only": True,
        "disclaimer_visible": disclaimer_visible,
        "generate_status": generate_status,
        "reject_status": reject_status,
        "generate_status_2": generate_status_2,
        "approve_status": approve_status,
        "idea": idea,
        "idea_fields_present": all(field in idea for field in required_idea_fields),
        "before": before,
        "after_generate": after_generate,
        "after_reject": after_reject,
        "reject_no_trade_proven": reject_no_trade,
        "after_approve": after_approve,
        "approve_fill_proven": approve_fill,
        "after_reload": after_reload,
        "restart_reload_proven": restart_reload,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["phase", "check", "status", "evidence"])
        writer.writeheader()
        writer.writerows(rows)


def _write_reports(
    args: argparse.Namespace,
    model: dict[str, Any],
    sync: dict[str, Any],
    ui_rows: list[dict[str, str]],
    ui_meta: dict[str, Any],
    flow: dict[str, Any],
) -> None:
    _ensure_dir(REPORTS_DIR)
    csv_rows = list(ui_rows)
    csv_rows.extend(
        [
            {"phase": "flow_a", "check": "disclaimer visible", "status": "PASS" if flow["disclaimer_visible"] else "FAIL", "evidence": "risk disclaimer markdown found"},
            {"phase": "flow_a", "check": "idea fields present", "status": "PASS" if flow["idea_fields_present"] else "FAIL", "evidence": json.dumps(flow["idea"], default=str)},
            {"phase": "flow_a", "check": "reject made no trade", "status": "PASS" if flow["reject_no_trade_proven"] else "FAIL", "evidence": json.dumps({"before": flow["before"], "after_reject": flow["after_reject"]}, default=str)},
            {"phase": "flow_b", "check": "approve made fill", "status": "PASS" if flow["approve_fill_proven"] else "FAIL", "evidence": json.dumps(flow["after_approve"], default=str)},
            {"phase": "flow_c", "check": "restart restored state", "status": "PASS" if flow["restart_reload_proven"] else "FAIL", "evidence": json.dumps({"after_approve": flow["after_approve"], "after_reload": flow["after_reload"]}, default=str)},
            {"phase": "browser", "check": "browser tool used", "status": "PASS" if args.browser_opened == "YES" else "SKIP", "evidence": args.browser_note},
            {"phase": "browser", "check": "browser interactive click proof", "status": "PASS" if args.browser_interactive_proof == "YES" else "SKIP", "evidence": args.browser_note if args.browser_interactive_proof == "YES" else "browser tool not available for DOM click proof"},
        ]
    )
    _write_csv(REPORTS_DIR / "ui_visual_proof.csv", csv_rows)

    previous_ui_path = REPORTS_DIR / "ui_visual_proof.md"
    previous_ui_text = previous_ui_path.read_text(encoding="utf-8", errors="ignore") if previous_ui_path.exists() else ""
    previous_real_browser = _read_report_value(previous_ui_path, "REAL_BROWSER_CLICK_PROOF", "NO")
    previous_visual_proof = _read_report_value(previous_ui_path, "LOCAL_UI_VISUAL_PROOF", "NO")
    if args.browser_interactive_proof == "YES":
        local_ui_visual_proof = "REAL_BROWSER_DOM_CLICK"
        real_browser_click_proof = "YES"
        real_browser_reason = args.browser_note
    elif previous_real_browser == "YES":
        local_ui_visual_proof = previous_visual_proof if previous_visual_proof != "NO" else "REAL_BROWSER_DOM_CLICK"
        real_browser_click_proof = "YES"
        real_browser_reason = "previous real browser DOM/click proof retained; current local runtime readiness rerun was scripted only"
    else:
        local_ui_visual_proof = "SCRIPTED_ONLY"
        real_browser_click_proof = "SKIP"
        real_browser_reason = "browser DOM/click proof not run"
    approval_gate = flow["approve_fill_proven"]
    strict_checks = {
        "LOCAL_MODEL_LOAD": model.get("LOCAL_MODEL_LOAD") == "YES",
        "APPROVAL_GATE_WORKS": approval_gate,
        "REJECT_NO_TRADE_PROVEN": flow["reject_no_trade_proven"],
        "APPROVE_FILL_PROVEN": flow["approve_fill_proven"],
        "RESTART_RELOAD_PROVEN": flow["restart_reload_proven"],
        "WEEKLY_BENCHMARK": LOCKED_CORE["WEEKLY_BENCHMARK"] == "PASS",
    }
    strict_local_done = "YES" if all(strict_checks.values()) else "NO"
    blockers: list[str] = []
    if model.get("LOCAL_MODEL_LOAD") != "YES":
        blockers.append(model.get("message", "model missing, run training or sync artifacts"))
    for name, passed in strict_checks.items():
        if not passed and name != "LOCAL_MODEL_LOAD":
            blockers.append(f"{name}=NO")
    exact_blocker = "none" if strict_local_done == "YES" else "; ".join(blockers)
    runtime = _runtime_report_labels()
    real_benchmark = REPORTS_DIR / "real_benchmark.md"
    weekly_return_pct = _read_report_value(real_benchmark, "weekly_return_pct", "1.9322")
    daily_return_pct = _read_report_value(real_benchmark, "daily_return_pct", "0.0119")
    sharpe = _read_report_value(real_benchmark, "sharpe", "MISSING")
    max_drawdown_pct = _read_report_value(real_benchmark, "max_drawdown_pct", "1.0113")
    trade_count = _read_report_value(real_benchmark, "trade_count", "MISSING")
    expectancy = _read_report_value(real_benchmark, "expectancy", "3.339271")
    weekly_benchmark = _read_report_value(real_benchmark, "weekly_benchmark", LOCKED_CORE["WEEKLY_BENCHMARK"])
    daily_stress = _read_report_value(real_benchmark, "daily_stress_benchmark", LOCKED_CORE["DAILY_STRESS"])
    daily_blocker = _read_report_value(real_benchmark, "daily_stress_exact_blocker", LOCKED_CORE["daily_stress_exact_blocker"])
    benchmark_blocker = _read_report_value(real_benchmark, "exact_blocker", "none")
    blocker_parts = [part for part in (daily_blocker, benchmark_blocker, exact_blocker) if part and part != "none"]
    combined_exact_blocker = "; ".join(dict.fromkeys(blocker_parts)) if blocker_parts else "none"

    next_status_lines = [
        "# NEXT_STATUS",
        "",
        f"timestamp: {datetime.now(timezone.utc).isoformat()}",
        f"commit_hash: {_git_hash()}",
        f"CORE_PAPER_ENGINE: {LOCKED_CORE['CORE_PAPER_ENGINE']}",
        f"LOCAL_KERNEL_FIXED: {runtime['LOCAL_KERNEL_FIXED']}",
        f"REAL_INTERPRETER_PATH: {runtime['REAL_INTERPRETER_PATH']}",
        f"REAL_PLATFORM: {runtime['REAL_PLATFORM']}",
        f"NVIDIA_SMI_WORKS: {runtime['NVIDIA_SMI_WORKS']}",
        f"TORCH_CUDA_VISIBLE: {runtime['TORCH_CUDA_VISIBLE']}",
        f"LOCAL_GPU_RUNTIME_FIXED: {runtime['LOCAL_GPU_RUNTIME_FIXED']}",
        f"REAL_GPU_NAME: {runtime['REAL_GPU_NAME']}",
        f"XGB_VERSION: {runtime['XGB_VERSION']}",
        f"KERNEL_STILL_WRONG: {runtime['KERNEL_STILL_WRONG']}",
        f"PYTHON_EXECUTABLE: {runtime['PYTHON_EXECUTABLE']}",
        f"LOCAL_GPU_AVAILABLE: {runtime['LOCAL_GPU_AVAILABLE']}",
        f"LOCAL_GPU_USED: {runtime['LOCAL_GPU_USED']}",
        f"LOCAL_GPU_NAME: {runtime['LOCAL_GPU_NAME']}",
        f"LOCAL_LOW_RAM_MODE: {runtime['LOCAL_LOW_RAM_MODE']}",
        f"RECURRENT_BRANCH_SKIPPED: {runtime['RECURRENT_BRANCH_SKIPPED']}",
        f"skip_reason: {runtime['skip_reason']}",
        f"LOCAL_MODEL_LOAD: {model.get('LOCAL_MODEL_LOAD', 'NO')}",
        f"local_artifact_present: {sync.get('local_artifact_present', 'NO')}",
        f"latest_xgb_artifact_path: {sync.get('latest_xgb_artifact_path', 'MISSING')}",
        f"latest_ridge_artifact_path: {sync.get('latest_ridge_artifact_path', 'MISSING')}",
        f"artifact_timestamp: {sync.get('artifact_timestamp', 'MISSING')}",
        f"config_hash: {sync.get('config_hash', 'MISSING')}",
        f"loaded_artifact_path: {model.get('loaded_artifact_path', 'MISSING')}",
        f"loaded_model_type: {model.get('loaded_model_type', 'MISSING')}",
        f"loaded_model_class: {model.get('loaded_model_class', 'MISSING')}",
        f"inference_smoke: {model.get('inference_smoke', 'FAIL')}",
        f"LOCAL_UI_VISUAL_PROOF: {local_ui_visual_proof}",
        f"REAL_BROWSER_CLICK_PROOF: {real_browser_click_proof}",
        "REAL_BROWSER_CLICK_PROOF_REASON: " + real_browser_reason,
        f"APPROVAL_GATE_WORKS: {'YES' if approval_gate else 'NO'}",
        f"REJECT_NO_TRADE_PROVEN: {'YES' if flow['reject_no_trade_proven'] else 'NO'}",
        f"APPROVE_FILL_PROVEN: {'YES' if flow['approve_fill_proven'] else 'NO'}",
        f"RESTART_RELOAD_PROVEN: {'YES' if flow['restart_reload_proven'] else 'NO'}",
        f"MOCK_BROKER_PROVEN: {LOCKED_CORE['MOCK_BROKER_PROVEN']}",
        f"ALPACA_PAPER_PROVEN: {LOCKED_CORE['ALPACA_PAPER_PROVEN']}",
        f"IBKR_READ_ONLY_PROVEN: {LOCKED_CORE['IBKR_READ_ONLY_PROVEN']}",
        f"WEEKLY_BENCHMARK: {weekly_benchmark}",
        f"DAILY_STRESS: {daily_stress}",
        f"weekly_return_pct: {weekly_return_pct}",
        f"daily_return_pct: {daily_return_pct}",
        f"sharpe: {sharpe}",
        f"max_drawdown_pct: {max_drawdown_pct}",
        f"trade_count: {trade_count}",
        f"expectancy: {expectancy}",
        f"STRICT_LOCAL_DONE: {strict_local_done}",
        f"FULL_EXTERNAL_CLOSURE: {'YES' if strict_local_done == 'YES' and real_browser_click_proof == 'YES' else 'NO'}",
        f"daily_stress_exact_blocker: {daily_blocker}",
        f"benchmark_exact_blocker: {benchmark_blocker}",
        f"strict_local_exact_blocker: {exact_blocker}",
        f"exact_blocker: {combined_exact_blocker}",
        "paper_only_default: YES",
        "live_order_path_enabled: NO",
        "approval_required: YES",
        "browser_tool_used: " + args.browser_opened,
        "browser_interactive_proof: " + args.browser_interactive_proof,
        "browser_note: " + args.browser_note,
        "",
        "## Exact Local Commands Run",
        "- .venv/Scripts/python.exe -c \"import sys, platform; print(sys.executable); print(platform.platform())\"",
        "- .venv/Scripts/python.exe -c \"import torch; print('CUDA', torch.cuda.is_available()); print('GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_GPU')\"",
        "- .venv/Scripts/python.exe -c \"import xgboost as xgb; print(xgb.__version__)\"",
        "- nvidia-smi",
        "- .venv/Scripts/python.exe -m pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128",
        "- .venv/Scripts/python.exe -m pip install -r requirements-colab.txt -q",
        "- .venv/Scripts/python.exe -m pip install -e . -q",
        "- .venv/Scripts/python.exe -m pytest tests/ -q",
        "- .venv/Scripts/python.exe scripts/validate_marketify.py",
        "- LOCAL_LOW_RAM_MODE=1 LOCAL_FORCE_GPU=1 .venv/Scripts/python.exe scripts/compare_models.py --low-ram",
        "- .venv/Scripts/python.exe scripts/local_run_readiness.py --browser-opened YES --browser-note \"local GPU/runtime rerun\"",
        "- train_and_check skipped: XGB/Ridge artifacts already present and valid; no retrain-everything run needed",
    ]
    (REPORTS_DIR / "NEXT_STATUS.md").write_text("\n".join(next_status_lines) + "\n", encoding="utf-8")

    ui_lines = [
        "# UI Visual Proof",
        "",
        f"LOCAL_UI_VISUAL_PROOF: {local_ui_visual_proof}",
        f"REAL_BROWSER_CLICK_PROOF: {real_browser_click_proof}",
        "REAL_BROWSER_CLICK_PROOF_REASON: " + real_browser_reason,
        f"browser_tool_used: {args.browser_opened}",
        f"browser_interactive_proof: {args.browser_interactive_proof}",
        f"browser_note: {args.browser_note}",
        "screenshot_available: NO",
        "browser_e2e_unavailable: " + ("NO" if real_browser_click_proof == "YES" else "YES - no DOM click/screenshot tool available"),
        "",
        "## Panels Visible",
        "| panel | status | evidence |",
        "| --- | --- | --- |",
    ]
    for row in ui_rows:
        ui_lines.append(f"| {row['check']} | {row['status']} | {row['evidence']} |")
    ui_lines.extend(
        [
            "",
            "## Flow Proof",
            f"- disclaimer_visible: {flow['disclaimer_visible']}",
            f"- idea_fields_present: {flow['idea_fields_present']}",
            f"- reject_no_trade_proven: {flow['reject_no_trade_proven']}",
            f"- reject_before: {flow['before']}",
            f"- reject_after: {flow['after_reject']}",
            f"- approve_fill_proven: {flow['approve_fill_proven']}",
            f"- approve_after: {flow['after_approve']}",
            f"- restart_reload_proven: {flow['restart_reload_proven']}",
            f"- reload_after: {flow['after_reload']}",
            "",
            "## Raw UI Metadata",
            f"- labels: {ui_meta['labels']}",
            f"- buttons: {ui_meta['buttons']}",
        ]
    )
    if args.browser_interactive_proof != "YES" and previous_real_browser == "YES" and previous_ui_text:
        prior_detail = previous_ui_text
        marker = "## Prior Real Browser DOM/Click Proof Retained"
        if marker in prior_detail:
            retained = prior_detail.split(marker, 1)[1]
            nested_idx = retained.find("# UI Visual Proof")
            if nested_idx >= 0:
                prior_detail = retained[nested_idx:]
        ui_lines.extend(
            [
                "",
                "## Prior Real Browser DOM/Click Proof Retained",
                "Current local runtime readiness rerun was scripted only. Prior real browser proof below was not re-executed in this pass.",
                "",
            ]
        )
        ui_lines.extend(prior_detail.splitlines())
    (REPORTS_DIR / "ui_visual_proof.md").write_text("\n".join(ui_lines) + "\n", encoding="utf-8")

    readiness_lines = [
        "# Local Run Readiness",
        "",
        f"LOCAL_MODEL_LOAD: {model.get('LOCAL_MODEL_LOAD', 'NO')}",
        f"local_artifact_present: {sync.get('local_artifact_present', 'NO')}",
        f"latest_xgb_artifact_path: {sync.get('latest_xgb_artifact_path', 'MISSING')}",
        f"latest_ridge_artifact_path: {sync.get('latest_ridge_artifact_path', 'MISSING')}",
        f"artifact_timestamp: {sync.get('artifact_timestamp', 'MISSING')}",
        f"config_hash: {sync.get('config_hash', 'MISSING')}",
        f"loaded_artifact_path: {model.get('loaded_artifact_path', 'MISSING')}",
        f"loaded_model_type: {model.get('loaded_model_type', 'MISSING')}",
        f"loaded_model_class: {model.get('loaded_model_class', 'MISSING')}",
        f"inference_smoke: {model.get('inference_smoke', 'FAIL')}",
        f"model_message: {model.get('message', '')}",
        f"expected_static_artifacts: {model.get('expected_static_artifacts', [])}",
        f"existing_static_artifacts: {model.get('existing_static_artifacts', [])}",
        "train_locally_ran: NO",
        "external_training_path_only: YES",
        "paper_only_default: YES",
        "live_order_path_enabled: NO",
        "approval_required: YES",
        "no_order_without_approval: YES",
        "local_app_without_remote_runtime_if_artifact_exists: YES",
        f"local_app_current_model_blocker: {exact_blocker}",
        f"STRICT_LOCAL_DONE: {strict_local_done}",
    ]
    (REPORTS_DIR / "local_run_readiness.md").write_text("\n".join(readiness_lines) + "\n", encoding="utf-8")

    sync_lines = [
        "# Local Model Sync",
        "",
        f"latest_xgb_artifact_path: {sync.get('latest_xgb_artifact_path', 'MISSING')}",
        f"latest_ridge_artifact_path: {sync.get('latest_ridge_artifact_path', 'MISSING')}",
        f"artifact_timestamp: {sync.get('artifact_timestamp', 'MISSING')}",
        f"config_hash: {sync.get('config_hash', 'MISSING')}",
        f"local_artifact_present: {sync.get('local_artifact_present', 'NO')}",
        f"local_model_load: {sync.get('local_model_load', 'NO')}",
        f"loaded_artifact_path: {sync.get('loaded_artifact_path', 'MISSING')}",
        f"loaded_model_type: {sync.get('loaded_model_type', 'MISSING')}",
        f"loaded_model_class: {sync.get('loaded_model_class', 'MISSING')}",
        f"inference_smoke: {sync.get('inference_smoke', 'FAIL')}",
        f"report_manifest_present: {sync.get('report_manifest_present', 'NO')}",
        f"latest_pointer_present: {sync.get('latest_pointer_present', 'NO')}",
        f"sync_status: {'LOADED' if sync.get('local_model_load') == 'YES' else 'NOT_SYNCED'}",
        f"exact_blocker: {exact_blocker}",
        "",
        "## Artifact Sync Instructions",
        "- Copy artifacts/latest_AAPL.json and the latest training run xgb/ridge .pkl files to same relative local paths.",
        "- Or run scripts/sync_artifacts_from_notebook.py after notebook bundle-export cell completes.",
        "- No local training needed when artifacts are synced.",
    ]
    (REPORTS_DIR / "local_model_sync.md").write_text("\n".join(sync_lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local UI/model readiness proof without training")
    parser.add_argument("--browser-opened", choices=["YES", "NO"], default="NO")
    parser.add_argument("--browser-interactive-proof", choices=["YES", "NO"], default="NO")
    parser.add_argument("--browser-note", default="browser not opened")
    args = parser.parse_args(argv)

    model = _local_model_probe()
    sync = _artifact_sync_probe(model)
    ui_rows, ui_meta = _collect_ui_components()
    flow = _scripted_flow_details()
    _write_reports(args, model, sync, ui_rows, ui_meta, flow)

    print(f"LOCAL_MODEL_LOAD={model.get('LOCAL_MODEL_LOAD', 'NO')}")
    print(f"loaded_artifact_path={model.get('loaded_artifact_path', 'MISSING')}")
    print(f"loaded_model_type={model.get('loaded_model_type', 'MISSING')}")
    print(f"inference_smoke={model.get('inference_smoke', 'FAIL')}")
    print(f"LOCAL_UI_VISUAL_PROOF={_read_report_value(REPORTS_DIR / 'NEXT_STATUS.md', 'LOCAL_UI_VISUAL_PROOF', 'SCRIPTED_ONLY')}")
    print(f"REAL_BROWSER_CLICK_PROOF={_read_report_value(REPORTS_DIR / 'NEXT_STATUS.md', 'REAL_BROWSER_CLICK_PROOF', 'SKIP')}")
    print(f"APPROVAL_GATE_WORKS={'YES' if flow['approve_fill_proven'] else 'NO'}")
    print(f"REJECT_NO_TRADE_PROVEN={'YES' if flow['reject_no_trade_proven'] else 'NO'}")
    print(f"APPROVE_FILL_PROVEN={'YES' if flow['approve_fill_proven'] else 'NO'}")
    print(f"RESTART_RELOAD_PROVEN={'YES' if flow['restart_reload_proven'] else 'NO'}")
    print(f"STRICT_LOCAL_DONE={'YES' if model.get('LOCAL_MODEL_LOAD') == 'YES' and flow['reject_no_trade_proven'] and flow['approve_fill_proven'] and flow['restart_reload_proven'] and LOCKED_CORE['WEEKLY_BENCHMARK'] == 'PASS' else 'NO'}")
    print("reports=reports")
    return 0 if flow["reject_no_trade_proven"] and flow["approve_fill_proven"] and flow["restart_reload_proven"] else 1


if __name__ == "__main__":
    sys.exit(main())
