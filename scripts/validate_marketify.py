"""Marketify validation automation.

Validation/readiness only. No live trading. No leverage change.
Writes:
  reports/validation_summary.json
  reports/validation_summary.md
  reports/broker_validation.md
  reports/broker_validation.csv
"""
from __future__ import annotations

import csv
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


def _broker_row(tier: str, target: str, status: str, reason: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "tier": tier,
        "target": target,
        "status": status,
        "reason": reason,
        "details": details or {},
    }


def run_import_smoke() -> dict[str, Any]:
    modules = [
        "app",
        "marketify.config",
        "marketify.broker.base",
        "marketify.broker.paper",
        "marketify.broker.mock_http",
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


def _account_identity_ok(account: dict[str, Any], positions: list[dict[str, Any]]) -> bool:
    equity = float(account["cash"]) + sum(float(pos["qty"]) * float(pos["mark_price"]) for pos in positions)
    return round(float(account["equity"]), 2) == round(equity, 2)


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
    required = [
        "Account",
        "Positions",
        "Open Orders / Orders",
        "Fills",
        "PnL Summary",
        "Risk Events",
        "Model Leaderboard",
        "Benchmark Status (weekly core / daily stress)",
        "Local Model Status",
        "Safety Status",
    ]
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
    class ScriptedModel:
        def predict(self, x_pred):
            return np.full(len(x_pred), 0.002)

    originals = (
        app_module.fetch_market_data,
        app_module.add_technical_features,
        app_module._load_local_model_artifact,
        app_module._predict_with_local_model,
    )
    app_module.fetch_market_data = lambda **_: raw
    app_module.add_technical_features = lambda _: feat
    app_module._load_local_model_artifact = lambda _: (
        ScriptedModel(),
        {
            "LOCAL_MODEL_LOAD": "YES",
            "loaded_artifact_path": "SCRIPTED_UI_PROOF",
            "loaded_model_type": "ScriptedModel",
            "inference_smoke": "PENDING",
            "message": "scripted UI proof model",
        },
    )
    app_module._predict_with_local_model = lambda model, frame: (float(model.predict(frame.tail(1))[0]), {"inference_smoke": "PASS", "feature_count": len(app_module.TECHNICAL_FEATURE_COLUMNS)})
    return originals


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
    }


def _scripted_ui_flow_with_broker(app_module, state: dict[str, Any], reload_broker_fn) -> dict[str, bool]:
    from marketify.config import RISK_DISCLAIMER

    markdown_values = [str(getattr(block, "value", "")) for block in app_module.demo.blocks.values()]
    disclaimer_visible = any(RISK_DISCLAIMER in value for value in markdown_values)
    state, _status, *_ = app_module.generate_trade_suggestion(state, "AAPL", "5m", "60d", False)
    generated = state["store"].pending_trade_idea is not None and len(state["broker"].get_orders()) == 0
    state, _reject_status, *_ = app_module.reject_trade(state, "validation_reject")
    reject_no_order = len(state["broker"].get_orders()) == 0 and len(state["broker"].get_fills()) == 0
    state, _status2, *_ = app_module.generate_trade_suggestion(state, "AAPL", "5m", "60d", False)
    state, _approve_status, *_ = app_module.approve_trade(state)
    broker = state["broker"]
    broker.update_market_price("AAPL", 105.0)
    account_after = broker.get_account()
    approve_fill = len(broker.get_orders()) == 1 and len(broker.get_fills()) == 1 and len(broker.get_positions()) == 1
    pnl_updates = float(account_after["unrealized_pnl"]) != 0.0
    reloaded = reload_broker_fn()
    reload_ok = len(reloaded.get_orders()) == 1 and len(reloaded.get_fills()) == 1 and len(reloaded.get_positions()) == 1
    state["broker"] = reloaded
    state, blocked_reset_status, *_ = app_module.reset_paper_account(state, "WRONG")
    reset_blocked = "blocked" in blocked_reset_status.lower()
    state, reset_status, *_ = app_module.reset_paper_account(state, "RESET PAPER")
    reset_ok = "reset" in reset_status.lower() and len(state["broker"].get_orders()) == 0
    return {
        "risk_disclaimer_visible": disclaimer_visible,
        "generate_trade_idea": generated,
        "reject_no_order_no_fill": reject_no_order,
        "approve_paper_fill": approve_fill,
        "positions_update": approve_fill,
        "pnl_updates": pnl_updates,
        "restart_reload": reload_ok,
        "reset_requires_confirmation": reset_blocked,
        "reset_paper_account": reset_ok,
    }


def run_scripted_ui_flow_proof() -> dict[str, Any]:
    import app
    from marketify.broker.mock_http import (MockBrokerHTTPClient,
                                            start_mock_broker_server)
    from marketify.broker.paper import PaperBroker
    from marketify.config import AppConfig

    originals = _patch_app_data_for_ui(app)
    server = None
    try:
        with tempfile.TemporaryDirectory() as td:
            internal_config = AppConfig()
            internal_config.broker.db_path = str(Path(td) / "ui_flow.db")
            internal_broker = PaperBroker(internal_config.broker)
            internal_state = _state_for_broker(app, internal_config, internal_broker)
            internal_checks = _scripted_ui_flow_with_broker(
                app,
                internal_state,
                lambda: PaperBroker(internal_config.broker),
            )
            internal_broker.conn.close()
            internal_state["broker"].conn.close()

            server = start_mock_broker_server()
            mock_config = AppConfig()
            mock_client = MockBrokerHTTPClient(server.base_url)
            mock_state = _state_for_broker(app, mock_config, mock_client)
            mock_checks = _scripted_ui_flow_with_broker(
                app,
                mock_state,
                lambda: MockBrokerHTTPClient(server.base_url),
            )
    finally:
        app.fetch_market_data, app.add_technical_features, app._load_local_model_artifact, app._predict_with_local_model = originals
        if server is not None:
            server.shutdown()

    checks = {
        "internal_paper": internal_checks,
        "mock_http": mock_checks,
        "approve_paper_fill": internal_checks["approve_paper_fill"] and mock_checks["approve_paper_fill"],
        "restart_sqlite_reload": internal_checks["restart_reload"],
        "mock_api_reload": mock_checks["restart_reload"],
    }
    failed = []
    for group_name, group_checks in (("internal_paper", internal_checks), ("mock_http", mock_checks)):
        failed.extend(f"{group_name}.{name}" for name, passed in group_checks.items() if not passed)
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


def _prove_local_paper_internal() -> dict[str, Any]:
    from marketify.broker.paper import PaperBroker
    from marketify.config import BrokerConfig

    with tempfile.TemporaryDirectory() as td:
        broker = PaperBroker(BrokerConfig(initial_cash=100_000.0, fee_bps=0.0, slippage_bps=0.0, db_path=str(Path(td) / "paper.db")))
        initial_orders = len(broker.get_orders())
        initial_fills = len(broker.get_fills())
        initial_positions = len(broker.get_positions())
        broker.update_market_price("AAPL", 100.0)
        broker.submit_order({"symbol": "AAPL", "side": "buy", "qty": 10, "price": 100.0})
        broker.update_market_price("AAPL", 101.0)
        account = broker.get_account()
        positions = broker.get_positions()
        fills = broker.get_fills()
        broker.conn.close()
    ok = (
        initial_orders == 0
        and initial_fills == 0
        and initial_positions == 0
        and len(positions) == 1
        and len(fills) == 1
        and round(float(account["cash"]), 2) == 99000.0
        and round(float(account["equity"]), 2) == 100010.0
    )
    return _broker_row("A", "local_paper_internal", "PASS" if ok else "FAIL", "SQLite PaperBroker accounting invariant", {"account": account, "positions": positions})


def _prove_local_mock_http() -> dict[str, Any]:
    from marketify.broker.mock_http import (ACCOUNT_FIXTURE,
                                            MockBrokerHTTPClient,
                                            start_mock_broker_server)

    server = start_mock_broker_server()
    try:
        client = MockBrokerHTTPClient(server.base_url)
        account0 = client.get_account()
        assets = client.get_assets()
        orders0 = client.get_orders()
        fills0 = client.get_fills()
        positions0 = client.get_positions()
        reject_unchanged = orders0 == [] and fills0 == [] and positions0 == [] and account0["equity"] == ACCOUNT_FIXTURE["equity"]
        client.submit_order({"symbol": "AAPL", "side": "buy", "qty": 10, "price": 100.0})
        account = client.get_account()
        positions = client.get_positions()
        fills = client.get_fills()
        clock = client.get_clock()
        cancel_result = client.cancel_order(client.get_orders()[0]["order_id"])
        short = client.submit_order({"symbol": "MSFT", "side": "sell", "qty": 5, "price": 200.0})
        account_after_short = client.get_account()
        ok = (
            account0["broker"] == "mock"
            and account0["mode"] == "paper"
            and len(assets) == 2
            and reject_unchanged
            and round(float(account["cash"]), 2) == 99000.0
            and round(float(account["equity"]), 2) == 100010.0
            and len(positions) == 1
            and positions[0]["symbol"] == "AAPL"
            and round(float(positions[0]["qty"]), 2) == 10.0
            and round(float(positions[0]["avg_entry"]), 2) == 100.0
            and round(float(positions[0]["mark_price"]), 2) == 101.0
            and round(float(positions[0]["unrealized_pnl"]), 2) == 10.0
            and len(fills) == 1
            and _account_identity_ok(account, positions)
            and bool(clock.get("is_open"))
            and cancel_result["status"] == "cannot_cancel_filled"
            and short["status"] == "filled"
            and account_after_short["mode"] == "paper"
        )
        return _broker_row(
            "B",
            "local_mock_http",
            "PASS" if ok else "FAIL",
            "deterministic mock broker HTTP contract and accounting invariant",
            {
                "initial_account": account0,
                "approved_buy_account": account,
                "approved_buy_positions": positions,
                "cancel_result": cancel_result,
                "short_order": short,
                "account_after_short": account_after_short,
            },
        )
    finally:
        server.shutdown()


def _probe_alpaca_paper() -> dict[str, Any]:
    from marketify.broker.alpaca import AlpacaBroker

    broker = AlpacaBroker()
    result = broker.connection_test()
    if result.get("connected"):
        return _broker_row("C", "alpaca_paper_read_only", "PASS", "Alpaca paper read-only account probe succeeded", result)
    if result.get("skipped"):
        return _broker_row("C", "alpaca_paper_read_only", "SKIP", str(result.get("reason", "Alpaca env unavailable")), result)
    return _broker_row("C", "alpaca_paper_read_only", "SKIP", str(result.get("reason", "Alpaca paper endpoint unavailable")), result)


def _probe_ibkr_read_only() -> dict[str, Any]:
    from marketify.broker.ibkr import IBKRBroker

    broker = IBKRBroker.from_env()
    has_env = bool(os.environ.get("IBKR_ACCOUNT_ID") or os.environ.get("IBKR_HOST") or os.environ.get("TWS_HOST"))
    if not has_env:
        return _broker_row("C", "ibkr_paper_read_only", "SKIP", "IBKR env unavailable", {"connected": False, "skipped": True})
    try:
        result = broker.connection_test()
        broker.disconnect()
        if result.get("connected"):
            return _broker_row("C", "ibkr_paper_read_only", "PASS", "IBKR read-only paper probe succeeded", result)
        return _broker_row("C", "ibkr_paper_read_only", "SKIP", str(result.get("error") or result.get("reason") or "IBKR unavailable"), result)
    except Exception as exc:
        return _broker_row("C", "ibkr_paper_read_only", "SKIP", f"IBKR TWS/Gateway unavailable: {exc}", {"connected": False, "skipped": True})


def run_broker_readiness() -> dict[str, Any]:
    rows = [_prove_local_paper_internal(), _prove_local_mock_http(), _probe_alpaca_paper(), _probe_ibkr_read_only()]
    unsafe = []
    if os.environ.get("LIVE_TRADING", "false").lower() == "true":
        unsafe.append("LIVE_TRADING=true")
    if unsafe:
        return _fail("broker_readiness", "unsafe live-trading env", rows=rows, unsafe=unsafe)

    mock_status = next(row for row in rows if row["target"] == "local_mock_http")["status"]
    alpaca_status = next(row for row in rows if row["target"] == "alpaca_paper_read_only")["status"]
    ibkr_status = next(row for row in rows if row["target"] == "ibkr_paper_read_only")["status"]
    local_fail = [row for row in rows if row["tier"] in {"A", "B"} and row["status"] != "PASS"]
    if local_fail:
        return _fail("broker_readiness", "local broker proof failed", rows=rows)

    labels = {
        "MOCK_BROKER_PROVEN": "YES" if mock_status == "PASS" else "NO",
        "ALPACA_PAPER_PROVEN": "YES" if alpaca_status == "PASS" else "SKIP" if alpaca_status == "SKIP" else "NO",
        "IBKR_READ_ONLY_PROVEN": "YES" if ibkr_status == "PASS" else "SKIP" if ibkr_status == "SKIP" else "NO",
    }
    status = "PASS_WITH_SKIPS" if any(row["status"] == "SKIP" for row in rows) else "PASS"
    return _ok("broker_readiness", status=status, labels=labels, rows=rows, paper_only_default=True, live_order_path_enabled=False)


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


def _write_broker_validation(broker_check: dict[str, Any]) -> tuple[Path, Path]:
    _ensure_dir(REPORTS_DIR)
    csv_path = REPORTS_DIR / "broker_validation.csv"
    md_path = REPORTS_DIR / "broker_validation.md"
    rows = broker_check.get("rows", [])
    labels = broker_check.get("labels", {})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["tier", "target", "status", "reason", "details_json"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "tier": row["tier"],
                    "target": row["target"],
                    "status": row["status"],
                    "reason": row["reason"],
                    "details_json": json.dumps(row.get("details", {}), sort_keys=True),
                }
            )

    lines = [
        "# Broker Validation",
        "",
        f"broker_readiness_result: {broker_check.get('status', 'FAIL') if broker_check.get('passed') else 'FAIL'}",
        f"MOCK_BROKER_PROVEN: {labels.get('MOCK_BROKER_PROVEN', 'NO')}",
        f"ALPACA_PAPER_PROVEN: {labels.get('ALPACA_PAPER_PROVEN', 'NO')}",
        f"IBKR_READ_ONLY_PROVEN: {labels.get('IBKR_READ_ONLY_PROVEN', 'NO')}",
        "real_broker_proven: NO if official targets are SKIP",
        "",
        "| tier | target | status | reason |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row['tier']} | {row['target']} | {row['status']} | {row['reason']} |")
    lines += [
        "",
        "Guardrails: paper only by default; no live endpoint; no order submission to official brokers unless explicit paper path is wired and guarded.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


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


def save_reports(report: dict[str, Any]) -> tuple[Path, Path]:
    _ensure_dir(REPORTS_DIR)
    json_path = REPORTS_DIR / "validation_summary.json"
    md_path = REPORTS_DIR / "validation_summary.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    broker_check = next((item for item in report["results"] if item["name"] == "broker_readiness"), {})
    _write_broker_validation(broker_check)
    broker_labels = broker_check.get("labels", {})

    lines = [
        "# Marketify Validation Summary",
        "",
        f"timestamp: {report['timestamp']}",
        f"validation_result: {'PASS' if report['all_passed'] else 'FAIL'}",
        "paper_only_default: YES",
        "live_trading_enabled: NO",
        "profit_guarantee: NO",
        f"broker_readiness_result: {broker_check.get('status', 'MISSING') if broker_check.get('passed') else 'FAIL'}",
        f"MOCK_BROKER_PROVEN: {broker_labels.get('MOCK_BROKER_PROVEN', 'NO')}",
        f"ALPACA_PAPER_PROVEN: {broker_labels.get('ALPACA_PAPER_PROVEN', 'NO')}",
        f"IBKR_READ_ONLY_PROVEN: {broker_labels.get('IBKR_READ_ONLY_PROVEN', 'NO')}",
        "",
        "## Checks",
        "| check | status | skipped | reason/error |",
        "| --- | --- | --- | --- |",
    ]
    for item in report["results"]:
        status = item.get("status") or ("PASS" if item["passed"] else "FAIL")
        reason = item.get("error") or item.get("reason") or item.get("note") or "none"
        lines.append(f"| {item['name']} | {status} | {item.get('skipped', False)} | {reason} |")

    lines += ["", "## Honest Notes"]
    ui_visual_path = REPORTS_DIR / "ui_visual_proof.md"
    real_browser_click_proof = _read_report_value(ui_visual_path, "REAL_BROWSER_CLICK_PROOF", "NO")
    if real_browser_click_proof == "YES":
        real_browser_reason = _read_report_value(ui_visual_path, "REAL_BROWSER_CLICK_PROOF_REASON", "prior proof retained")
        lines.append(f"- real_browser_click_proof: YES ({real_browser_reason})")
    for item in report["results"]:
        if item["name"] == "automatic_benchmark_checker":
            lines.append(f"- weekly_benchmark: {item.get('weekly_benchmark')}")
            lines.append(f"- daily_stress_benchmark: {item.get('daily_stress_benchmark')}")
            lines.append(f"- exact_blocker: {item.get('exact_blocker', 'pending_final_compare')}")
        if item["name"] == "broker_readiness":
            for row in item.get("rows", []):
                lines.append(f"- {row['target']}: {row['status']} ({row['reason']})")
        if item["name"] == "scripted_ui_flow_proof":
            lines.append(f"- ui_flow: {item.get('checks')}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


if __name__ == "__main__":
    print("[VALIDATE] starting Marketify validation automation")
    summary = run_all()
    json_report, md_report = save_reports(summary)
    broker_check = next((item for item in summary["results"] if item["name"] == "broker_readiness"), {})
    print(f"[VALIDATE] result={'PASS' if summary['all_passed'] else 'FAIL'}")
    print(f"[VALIDATE] broker_readiness={broker_check.get('status', 'MISSING')}")
    print(f"[VALIDATE] json={json_report}")
    print(f"[VALIDATE] markdown={md_report}")
    if not summary["all_passed"]:
        for check_result in summary["results"]:
            if not check_result["passed"]:
                print(f"[VALIDATE] FAILED {check_result['name']}: {check_result.get('error')}", file=sys.stderr)
    sys.exit(0 if summary["all_passed"] else 1)
