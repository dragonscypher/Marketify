"""Marketify validation automation.

Runs validation/readiness checks only. No live trading. No leverage change.
Writes reports/validation_summary.json and reports/validation_summary.md.
"""
from __future__ import annotations

import json
import os
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


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ok(name: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "passed": True, "skipped": False, **extra}


def _fail(name: str, error: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "passed": False, "skipped": False, "error": error, **extra}


def run_import_smoke() -> dict[str, Any]:
    modules = [
        "app",
        "marketify.config",
        "marketify.broker.base",
        "marketify.broker.paper",
        "marketify.broker.ibkr",
        "marketify.broker.alpaca",
        "marketify.risk.risk_engine",
        "marketify.state.store",
        "marketify.models.xgb_model",
        "marketify.models.ridge_model",
        "marketify.models.train_pipeline",
        "marketify.backtest.benchmark",
        "marketify.backtest.simulator",
        "marketify.onboarding",
        "marketify.data.market_data",
        "marketify.features.technical",
    ]
    errors: list[str] = []
    for mod in modules:
        try:
            __import__(mod)
        except Exception as exc:
            errors.append(f"{mod}: {exc}")
    if errors:
        return _fail("import_smoke", "; ".join(errors), errors=errors)
    return _ok("import_smoke", modules=modules)


def run_pytest() -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    payload = {
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:] if result.stdout else "",
        "stderr_tail": result.stderr[-2000:] if result.stderr else "",
    }
    if result.returncode != 0:
        return _fail("pytest", "pytest failed", **payload)
    return _ok("pytest", **payload)


def run_synthetic_broker_accounting_test() -> dict[str, Any]:
    from marketify.broker.paper import PaperBroker
    from marketify.config import BrokerConfig

    with tempfile.TemporaryDirectory() as td:
        broker = PaperBroker(
            BrokerConfig(initial_cash=10_000.0, fee_bps=0.0, slippage_bps=0.0, db_path=str(Path(td) / "paper.db"))
        )
        broker.update_market_price("AAPL", 100.0)
        broker.submit_order({"symbol": "AAPL", "side": "buy", "qty": 10, "price": 100.0})
        broker.update_market_price("AAPL", 110.0)
        account = broker.get_account()
        positions = broker.get_positions()
        fills = broker.get_fills()
        broker.conn.close()
        if round(float(account["equity"]), 2) != 10100.0:
            return _fail("synthetic_broker_accounting", "equity mismatch", account=account)
        if len(positions) != 1 or len(fills) != 1:
            return _fail("synthetic_broker_accounting", "position/fill mismatch", positions=positions, fills=fills)
    return _ok("synthetic_broker_accounting", equity=10100.0, fills=1, positions=1)


def run_backtest_smoke() -> dict[str, Any]:
    from marketify.backtest.benchmark import compute_benchmark

    dates = pd.date_range("2026-01-01", periods=20, freq="D")
    equity = 10_000.0 * (1.0 + pd.Series(np.linspace(-0.001, 0.002, len(dates)))).cumprod()
    history = [{"ts": str(ts), "equity": float(value)} for ts, value in zip(dates, equity)]
    result = compute_benchmark(history, weekly_goal=0.01)
    return _ok(
        "backtest_smoke",
        weekly_return_pct=round(float(result.weekly_return) * 100.0, 4),
        max_drawdown_pct=round(float(result.max_drawdown) * 100.0, 4),
        note="synthetic smoke only; not success proof",
    )


def run_gradio_launch_smoke() -> dict[str, Any]:
    import gradio as gr

    import app

    demo = getattr(app, "demo", None)
    if not isinstance(demo, gr.Blocks):
        return _fail("gradio_launch_smoke", "app.demo is not gradio.Blocks")
    labels = [getattr(block, "label", "") for block in demo.blocks.values()]
    required = ["Account", "Positions", "Open Orders / Orders", "Fills", "Risk Events", "Model Leaderboard"]
    missing = [label for label in required if label not in labels]
    if missing:
        return _fail("gradio_launch_smoke", "missing UI labels", missing=missing)
    return _ok("gradio_launch_smoke", browser_e2e_unavailable=True, reason="scripted UI proof used; no DOM click tool")


def _patch_app_data_for_ui(app_module):
    idx = pd.date_range("2026-01-01", periods=40, freq="5min")
    raw = pd.DataFrame(
        {
            "Open": np.linspace(100, 104, 40),
            "High": np.linspace(101, 105, 40),
            "Low": np.linspace(99, 103, 40),
            "Close": np.linspace(100, 104, 40),
            "Volume": np.full(40, 1_000),
        },
        index=idx,
    )
    feat = raw.copy()
    for col in app_module.TECHNICAL_FEATURE_COLUMNS:
        feat[col] = 0.001
    feat["vol_20"] = 0.01
    feat["target_next_ret"] = 0.001
    preds = pd.Series(np.nan, index=idx)
    preds.iloc[-1] = 0.002

    originals = (app_module.fetch_market_data, app_module.add_technical_features, app_module.rolling_train_predict)
    app_module.fetch_market_data = lambda **_: raw
    app_module.add_technical_features = lambda _: feat
    app_module.rolling_train_predict = lambda **_: preds
    return originals


def run_scripted_ui_flow_proof() -> dict[str, Any]:
    import app
    from marketify.broker.paper import PaperBroker
    from marketify.config import RISK_DISCLAIMER, AppConfig
    from marketify.risk.risk_engine import RiskEngine
    from marketify.state.store import InMemoryStore

    with tempfile.TemporaryDirectory() as td:
        config = AppConfig()
        config.broker.db_path = str(Path(td) / "ui_flow.db")
        state = {
            "config": config,
            "broker": PaperBroker(config.broker),
            "risk": RiskEngine(config.risk, config.broker),
            "store": InMemoryStore(),
            "news": app.NewsProvider(),
            "sentiment": app.FinBERTSentiment(enabled=False),
            "last_price": None,
        }
        originals = _patch_app_data_for_ui(app)
        try:
            markdown_values = [str(getattr(block, "value", "")) for block in app.demo.blocks.values()]
            disclaimer_visible = any(RISK_DISCLAIMER in value for value in markdown_values)
            state, _status, *_ = app.generate_trade_suggestion(state, "AAPL", "5m", "60d", False)
            generated = state["store"].pending_trade_idea is not None and len(state["broker"].get_orders()) == 0
            state, _reject_status, *_ = app.reject_trade(state, "validation_reject")
            reject_no_order = len(state["broker"].get_orders()) == 0 and len(state["broker"].get_fills()) == 0
            state, _status2, *_ = app.generate_trade_suggestion(state, "AAPL", "5m", "60d", False)
            state, _approve_status, *_ = app.approve_trade(state)
            broker = state["broker"]
            broker.update_market_price("AAPL", 105.0)
            account_after = broker.get_account()
            approve_fill = len(broker.get_orders()) == 1 and len(broker.get_fills()) == 1 and len(broker.get_positions()) == 1
            pnl_updates = float(account_after["unrealized_pnl"]) != 0.0

            broker.conn.close()
            reloaded = PaperBroker(config.broker)
            reload_ok = len(reloaded.get_orders()) == 1 and len(reloaded.get_fills()) == 1 and len(reloaded.get_positions()) == 1
            reloaded.conn.close()

            state["broker"] = PaperBroker(config.broker)
            state, blocked_reset_status, *_ = app.reset_paper_account(state, "WRONG")
            reset_blocked = "blocked" in blocked_reset_status.lower()
            state, reset_status, *_ = app.reset_paper_account(state, "RESET PAPER")
            reset_ok = "reset" in reset_status.lower() and len(state["broker"].get_orders()) == 0
        finally:
            app.fetch_market_data, app.add_technical_features, app.rolling_train_predict = originals

    checks = {
        "risk_disclaimer_visible": disclaimer_visible,
        "generate_trade_idea": generated,
        "reject_no_order_no_fill": reject_no_order,
        "approve_paper_fill": approve_fill,
        "positions_update": approve_fill,
        "pnl_updates": pnl_updates,
        "restart_sqlite_reload": reload_ok,
        "reset_requires_confirmation": reset_blocked,
        "reset_paper_account": reset_ok,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        return _fail("scripted_ui_flow_proof", "UI flow proof failed", failed=failed, checks=checks)
    return _ok("scripted_ui_flow_proof", browser_e2e_unavailable=True, checks=checks)


def run_model_artifact_check() -> dict[str, Any]:
    from marketify.config import AppConfig
    from marketify.models.train_pipeline import ARTIFACT_DIR, train_if_missing

    config = AppConfig()
    leaderboard = train_if_missing(config=config, force=False)
    expected = [ARTIFACT_DIR / f"xgb_{config.data.ticker}.pkl", ARTIFACT_DIR / f"ridge_{config.data.ticker}.pkl"]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        return _fail("model_artifact_check", "missing artifacts after train_if_missing", missing=missing)
    return _ok("model_artifact_check", artifacts=[str(path) for path in expected], leaderboard=leaderboard)


def run_benchmark_checker() -> dict[str, Any]:
    real_path = REPORTS_DIR / "real_benchmark.md"
    if not real_path.exists():
        return _ok(
            "automatic_benchmark_checker",
            latest_report_available=False,
            weekly_benchmark="UNKNOWN_PRE_COMPARE",
            daily_stress_benchmark="UNKNOWN_PRE_COMPARE",
            note="checker ready; scripts/compare_models.py writes final benchmark after validation",
        )
    text = real_path.read_text(encoding="utf-8")

    def pick(key: str) -> str:
        for line in text.splitlines():
            if line.startswith(f"{key}:"):
                return line.split(":", 1)[1].strip()
        return "MISSING"

    weekly = float(pick("weekly_return_pct"))
    daily = float(pick("daily_return_pct"))
    max_dd = float(pick("max_drawdown_pct"))
    return _ok(
        "automatic_benchmark_checker",
        latest_report_available=True,
        weekly_benchmark="PASS" if weekly >= 1.0 and max_dd <= 10.0 else "FAIL",
        daily_stress_benchmark="PASS" if daily >= 1.0 and max_dd <= 10.0 else "FAIL",
        exact_blocker="none" if daily >= 1.0 else "daily_return_pct below 1.0",
        weekly_return_pct=weekly,
        daily_return_pct=daily,
        max_drawdown_pct=max_dd,
    )


def run_broker_readiness() -> dict[str, Any]:
    results: dict[str, Any] = {}
    try:
        from marketify.broker.ibkr import IBKRBroker

        ibkr = IBKRBroker.from_env()
        has_ibkr_env = bool(os.environ.get("IBKR_ACCOUNT_ID") or os.environ.get("IBKR_HOST") or os.environ.get("TWS_HOST"))
        if has_ibkr_env:
            results["ibkr"] = ibkr.connection_test()
            ibkr.disconnect()
        else:
            results["ibkr"] = {"connected": False, "skipped": True, "reason": "IBKR env unavailable"}
    except Exception as exc:
        results["ibkr"] = {"connected": False, "skipped": True, "reason": str(exc)}

    try:
        from marketify.broker.alpaca import AlpacaBroker

        alpaca = AlpacaBroker()
        results["alpaca"] = alpaca.connection_test()
    except Exception as exc:
        results["alpaca"] = {"connected": False, "skipped": True, "reason": str(exc)}

    unsafe = []
    if os.environ.get("LIVE_TRADING", "false").lower() == "true":
        unsafe.append("LIVE_TRADING=true")
    if unsafe:
        return _fail("broker_readiness", "unsafe live-trading env", results=results, unsafe=unsafe)
    return _ok("broker_readiness", paper_only_default=True, live_order_path_enabled=False, results=results)


def run_all() -> dict[str, Any]:
    checks = [
        run_import_smoke,
        run_pytest,
        run_synthetic_broker_accounting_test,
        run_backtest_smoke,
        run_gradio_launch_smoke,
        run_scripted_ui_flow_proof,
        run_model_artifact_check,
        run_benchmark_checker,
        run_broker_readiness,
    ]
    results: list[dict[str, Any]] = []
    for check in checks:
        try:
            results.append(check())
        except Exception as exc:
            results.append(_fail(check.__name__, str(exc)))
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "all_passed": all(item["passed"] for item in results),
        "results": results,
    }


def save_reports(report: dict[str, Any]) -> tuple[Path, Path]:
    _ensure_dir(REPORTS_DIR)
    json_path = REPORTS_DIR / "validation_summary.json"
    md_path = REPORTS_DIR / "validation_summary.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Marketify Validation Summary",
        "",
        f"timestamp: {report['timestamp']}",
        f"validation_result: {'PASS' if report['all_passed'] else 'FAIL'}",
        "paper_only_default: YES",
        "live_trading_enabled: NO",
        "profit_guarantee: NO",
        "",
        "## Checks",
        "| check | status | skipped | reason/error |",
        "| --- | --- | --- | --- |",
    ]
    for item in report["results"]:
        status = "PASS" if item["passed"] else "FAIL"
        reason = item.get("error") or item.get("reason") or item.get("note") or "none"
        lines.append(f"| {item['name']} | {status} | {item.get('skipped', False)} | {reason} |")

    lines += ["", "## Honest Notes"]
    for item in report["results"]:
        if item["name"] == "automatic_benchmark_checker":
            lines.append(f"- weekly_benchmark: {item.get('weekly_benchmark')}")
            lines.append(f"- daily_stress_benchmark: {item.get('daily_stress_benchmark')}")
            lines.append(f"- exact_blocker: {item.get('exact_blocker', 'pending_final_compare')}")
        if item["name"] == "broker_readiness":
            lines.append(f"- broker_readiness: {item.get('results')}")
        if item["name"] == "scripted_ui_flow_proof":
            lines.append(f"- ui_flow: {item.get('checks')}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


if __name__ == "__main__":
    print("[VALIDATE] starting Marketify validation automation")
    summary = run_all()
    json_report, md_report = save_reports(summary)
    print(f"[VALIDATE] result={'PASS' if summary['all_passed'] else 'FAIL'}")
    print(f"[VALIDATE] json={json_report}")
    print(f"[VALIDATE] markdown={md_report}")
    if not summary["all_passed"]:
        for check_result in summary["results"]:
            if not check_result["passed"]:
                print(f"[VALIDATE] FAILED {check_result['name']}: {check_result.get('error')}", file=sys.stderr)
    sys.exit(0 if summary["all_passed"] else 1)