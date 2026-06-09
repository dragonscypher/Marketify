from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, cast

import gradio as gr
import pandas as pd

from marketify.broker.paper import PaperBroker
from marketify.config import RISK_DISCLAIMER, AppConfig
from marketify.data.market_data import fetch_market_data
from marketify.data.news_data import NewsProvider
from marketify.features.sentiment import FinBERTSentiment
from marketify.features.technical import (TECHNICAL_FEATURE_COLUMNS,
                                          add_technical_features)
from marketify.models.ensemble import (TradeIdeaInput, build_trade_idea,
                                       compute_cvar_95)
from marketify.onboarding import (apply_user_config, load_user_config,
                                  save_user_config)
from marketify.risk.risk_engine import RiskEngine
from marketify.state.store import InMemoryStore

REPORTS_DIR = Path("reports")
ARTIFACT_DIR = Path("artifacts")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


def _resolve_artifact_ref(path_value: Any) -> Path | None:
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    raw_path = Path(path_value.strip())
    parts = list(raw_path.parts)
    if "artifacts" in parts:
        raw_path = ARTIFACT_DIR / Path(*parts[parts.index("artifacts") + 1:])
    elif raw_path.is_absolute():
        return None
    else:
        raw_path = ARTIFACT_DIR / raw_path

    artifact_root = ARTIFACT_DIR.resolve()
    try:
        raw_path.resolve().relative_to(artifact_root)
    except ValueError:
        return None
    return raw_path


def _manifest_artifact_candidates(ticker: str, model_name: str) -> list[Path]:
    ticker = ticker.upper().strip()
    docs = [
        _read_json(ARTIFACT_DIR / f"latest_{ticker}.json"),
        _read_json(REPORTS_DIR / "artifact_manifest.json"),
    ]
    candidates: list[Path] = []
    for doc in docs:
        if not doc:
            continue
        model_doc = doc.get("models", {}).get(model_name, {}) if isinstance(doc.get("models"), dict) else {}
        refs = [
            doc.get(f"latest_{model_name}_artifact_path"),
            doc.get(f"static_{model_name}_artifact_path"),
            model_doc.get("artifact_path") if isinstance(model_doc, dict) else None,
            model_doc.get("latest_artifact_path") if isinstance(model_doc, dict) else None,
            model_doc.get("static_artifact_path") if isinstance(model_doc, dict) else None,
        ]
        for ref in refs:
            path = _resolve_artifact_ref(ref)
            if path is not None:
                candidates.append(path)
    return candidates


def _find_latest_artifact(ticker: str, model_name: str = "xgb") -> Path | None:
    ticker = ticker.upper().strip()
    manifest_candidates = _manifest_artifact_candidates(ticker, model_name)
    manifest_existing = list(dict.fromkeys(path for path in manifest_candidates if path.exists()))
    if manifest_existing:
        return manifest_existing[0]
    fallback_candidates = [
        ARTIFACT_DIR / f"{model_name}_{ticker}.pkl",
        *ARTIFACT_DIR.glob(f"training_runs/*/{model_name}_{ticker}.pkl"),
    ]
    existing = list(dict.fromkeys(path for path in fallback_candidates if path.exists()))
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def _load_local_model_artifact(ticker: str) -> tuple[Any | None, dict[str, Any]]:
    artifact_path = _find_latest_artifact(ticker, "xgb")
    if artifact_path is None:
        return None, {
            "LOCAL_MODEL_LOAD": "NO",
            "loaded_artifact_path": "MISSING",
            "loaded_model_type": "MISSING",
            "inference_smoke": "FAIL",
            "message": "model missing, run training or sync artifacts",
        }
    try:
        with artifact_path.open("rb") as f:
            model = pickle.load(f)
        return model, {
            "LOCAL_MODEL_LOAD": "YES",
            "loaded_artifact_path": str(artifact_path),
            "loaded_model_type": "xgb",
            "loaded_model_class": type(model).__name__,
            "inference_smoke": "PENDING",
            "message": "trained artifact loaded locally",
        }
    except Exception as exc:
        return None, {
            "LOCAL_MODEL_LOAD": "NO",
            "loaded_artifact_path": str(artifact_path),
            "loaded_model_type": "LOAD_ERROR",
            "inference_smoke": "FAIL",
            "message": f"model load failed: {exc}",
        }


def _predict_with_local_model(model: Any, feat: pd.DataFrame) -> tuple[float | None, dict[str, Any]]:
    feature_cols = [col for col in TECHNICAL_FEATURE_COLUMNS if col in feat.columns]
    if not feature_cols:
        return None, {"inference_smoke": "FAIL", "message": "no technical feature columns available"}
    try:
        latest_x = feat[feature_cols].tail(1).to_numpy()
        pred = float(model.predict(latest_x)[0])
        return pred, {"inference_smoke": "PASS", "feature_count": len(feature_cols)}
    except Exception as exc:
        return None, {"inference_smoke": "FAIL", "message": f"inference failed: {exc}"}


def _latest_numeric_column_value(frame: pd.DataFrame, column: str) -> float:
    if isinstance(frame.columns, pd.MultiIndex):
        matches = [col for col in frame.columns if isinstance(col, tuple) and column in col]
        values: Any = frame.loc[:, matches[0]] if matches else frame[column]
    else:
        values = frame[column]
    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]
    squeezed = values.squeeze()
    if isinstance(squeezed, pd.Series):
        series = pd.to_numeric(cast(Any, squeezed), errors="coerce").dropna()
    else:
        series = pd.Series([pd.to_numeric(squeezed, errors="coerce")]).dropna()
    if series.empty:
        raise ValueError(f"No numeric {column} value available.")
    return float(series.iloc[-1])


def _latest_scalar(frame: pd.DataFrame, column: str) -> float:
    return _latest_numeric_column_value(frame, column)


def bootstrap() -> dict:
    config = AppConfig()
    return {
        "config": config,
        "broker": PaperBroker(config.broker),
        "risk": RiskEngine(config.risk, config.broker),
        "store": InMemoryStore(),
        "news": NewsProvider(),
        "sentiment": FinBERTSentiment(enabled=False),
        "last_price": None,
        "model_status": {
            "LOCAL_MODEL_LOAD": "UNKNOWN",
            "loaded_artifact_path": "not checked yet",
            "loaded_model_type": "UNKNOWN",
            "inference_smoke": "UNKNOWN",
            "message": "Generate suggestion checks local trained artifact.",
        },
    }


def _ensure_state(state: dict | None) -> dict:
    return state if state is not None else bootstrap()


def _check_halt(state: dict) -> str | None:
    """Return halt reason string if engine should block, else None."""
    config: AppConfig = state["config"]
    if config.risk.kill_switch_enabled:
        return "kill_switch"
    if config.halt_reason:
        return config.halt_reason
    broker: PaperBroker = state["broker"]
    account = broker.get_account()
    if config.risk.auto_halt_on_daily_loss and account["daily_loss_pct"] > config.risk.max_daily_loss_pct:
        config.halt_reason = "daily_loss_limit_breached"
        broker.append_risk_event("daily_loss_halt", "critical", config.halt_reason, account)
        return config.halt_reason
    if config.risk.auto_halt_on_drawdown and account["drawdown_pct"] > config.risk.max_drawdown_pct:
        config.halt_reason = "max_drawdown_breached"
        broker.append_risk_event("drawdown_halt", "critical", config.halt_reason, account)
        return config.halt_reason
    return None


def _engine_status(state: dict) -> str:
    halt = _check_halt(state)
    return f"HALTED: {halt}" if halt else "ACTIVE"


def _frames(state: dict | None) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    state = _ensure_state(state)
    broker = state["broker"]
    account = pd.DataFrame([broker.get_account()])
    positions = pd.DataFrame(broker.get_positions())
    orders = pd.DataFrame(broker.get_orders())
    return state, account, positions, orders


def _pnl_frame(state: dict | None) -> pd.DataFrame:
    state = _ensure_state(state)
    account = state["broker"].get_account()
    return pd.DataFrame(
        [
            {
                "equity": account.get("equity", 0.0),
                "cash": account.get("cash", 0.0),
                "realized_pnl": account.get("realized_pnl", 0.0),
                "unrealized_pnl": account.get("unrealized_pnl", 0.0),
                "drawdown_pct": account.get("drawdown_pct", 0.0),
                "daily_loss_pct": account.get("daily_loss_pct", 0.0),
            }
        ]
    )


def _model_status_text(state: dict | None) -> str:
    state = _ensure_state(state)
    status = state.get("model_status", {})
    return (
        f"LOCAL_MODEL_LOAD={status.get('LOCAL_MODEL_LOAD', 'UNKNOWN')}; "
        f"loaded_artifact_path={status.get('loaded_artifact_path', 'MISSING')}; "
        f"loaded_model_type={status.get('loaded_model_type', 'MISSING')}; "
        f"inference_smoke={status.get('inference_smoke', 'UNKNOWN')}; "
        f"message={status.get('message', '')}"
    )


def _safety_status_text(state: dict | None) -> str:
    state = _ensure_state(state)
    mode = state["config"].trading_mode
    return (
        f"paper_only_default={'YES' if mode == 'paper' else 'NO'}; "
        "live_order_path_enabled=NO; approval_required=YES; "
        f"engine_status={_engine_status(state)}"
    )


def _fills_frame(state: dict | None) -> pd.DataFrame:
    state = _ensure_state(state)
    broker: PaperBroker = state["broker"]
    return pd.DataFrame(broker.get_fills())


def _risk_events_frame(state: dict | None) -> pd.DataFrame:
    state = _ensure_state(state)
    broker: PaperBroker = state["broker"]
    return pd.DataFrame(broker.get_risk_events())


def _model_leaderboard_frame() -> pd.DataFrame:
    path = REPORTS_DIR / "model_leaderboard.csv"
    if not path.exists():
        return pd.DataFrame(columns=["model_name", "weekly_return_pct", "daily_return_pct", "PASS/FAIL"])
    try:
        return pd.read_csv(path)
    except Exception as exc:
        return pd.DataFrame([{"error": f"failed_to_load_model_leaderboard: {exc}"}])


def _benchmark_status_text() -> str:
    path = REPORTS_DIR / "real_benchmark.md"
    if not path.exists():
        return "weekly_benchmark: UNKNOWN; daily_stress_benchmark: UNKNOWN; run scripts/compare_models.py"
    text = path.read_text(encoding="utf-8")

    def pick(key: str) -> str:
        prefix = f"{key}:"
        for line in text.splitlines():
            if line.startswith(prefix):
                return line.split(":", 1)[1].strip()
        return "MISSING"

    weekly = pick("weekly_return_pct")
    daily = pick("daily_return_pct")
    max_dd = pick("max_drawdown_pct")
    result = pick("PASS/FAIL")
    exact = pick("exact_blocker")
    try:
        daily_status = "PASS" if float(daily) >= 1.0 else "FAIL"
    except ValueError:
        daily_status = "UNKNOWN"
    return (
        f"weekly_benchmark: {result}; daily_stress_benchmark: {daily_status}; "
        f"weekly_return_pct={weekly}; daily_return_pct={daily}; max_drawdown_pct={max_dd}; exact_blocker={exact}"
    )


def _idea_frame(state: dict | None) -> pd.DataFrame:
    state = _ensure_state(state)
    idea = state["store"].pending_trade_idea
    if idea is None:
        return pd.DataFrame(columns=[
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
        ])
    return pd.DataFrame([idea.to_dict()])


def generate_trade_suggestion(state: dict | None, ticker: str, interval: str, period: str, prepost: bool):
    state = _ensure_state(state)
    config: AppConfig = state["config"]
    broker: PaperBroker = state["broker"]

    halt = _check_halt(state)
    if halt:
        broker.append_audit("generate_suggestion", "blocked", reason=halt)
        state, account, positions, orders = _frames(state)
        return (
            state,
            f"Engine HALTED: {halt}. No suggestions while halted.",
            _idea_frame(state),
            account,
            positions,
            orders,
        )

    ticker = ticker.upper().strip()
    config.data.ticker = ticker
    config.data.interval = interval
    config.data.period = period
    config.data.prepost = bool(prepost)

    model, model_status = _load_local_model_artifact(ticker)
    state["model_status"] = model_status
    if model is None:
        state["store"].clear_pending()
        state, account, positions, orders = _frames(state)
        return (
            state,
            f"[{_engine_status(state)}] {model_status['message']}. No local order path touched.",
            _idea_frame(state),
            account,
            positions,
            orders,
        )

    raw = fetch_market_data(ticker=ticker, interval=interval, period=period, prepost=prepost)
    feat = add_technical_features(raw)
    latest_pred, inference_status = _predict_with_local_model(model, feat)
    state["model_status"] = {**model_status, **inference_status}
    if latest_pred is None:
        state["store"].clear_pending()
        state, account, positions, orders = _frames(state)
        return (
            state,
            f"[{_engine_status(state)}] {state['model_status']['message']}. No local order path touched.",
            _idea_frame(state),
            account,
            positions,
            orders,
        )

    last_price = _latest_scalar(feat, "Close")
    state["last_price"] = last_price

    broker = state["broker"]
    broker.update_market_price(ticker, last_price)

    ret_lookback = feat["ret_1"].iloc[max(0, len(feat) - 250):]
    cvar_95 = compute_cvar_95(ret_lookback)
    news = state["news"].summarize_risk(ticker)
    sentiment = state["sentiment"].analyze(ticker)

    idea = build_trade_idea(
        info=TradeIdeaInput(
            symbol=ticker,
            last_price=last_price,
            prediction=latest_pred,
            sentiment_score=sentiment.score,
            volatility=_latest_scalar(feat, "vol_20"),
            news_risk=float(news["news_risk"]),
        ),
        equity=broker.get_account()["equity"],
        broker_cfg=config.broker,
        risk_cfg=config.risk,
        cvar_95=cvar_95,
    )

    if idea is None:
        state["store"].clear_pending()
        state, account, positions, orders = _frames(state)
        return (
            state,
            "No trade idea generated. Model signal weak or qty invalid.",
            _idea_frame(state),
            account,
            positions,
            orders,
        )

    decision = state["risk"].evaluate(
        trade=idea,
        account=broker.get_account(),
        volatility=_latest_scalar(feat, "vol_20"),
        news_risk=float(news["news_risk"]),
        mark_price=last_price,
    )

    if not decision.approved:
        state["store"].clear_pending()
        state["store"].add_risk_event({"symbol": ticker, "reasons": decision.reasons})
        broker.append_risk_event("risk_rejection", "warning", ", ".join(decision.reasons), {"symbol": ticker})
        broker.append_audit("generate_suggestion", "risk_rejected", symbol=ticker, reason=", ".join(decision.reasons))
        state, account, positions, orders = _frames(state)
        reason_text = ", ".join(decision.reasons)
        return (
            state,
            f"[{_engine_status(state)}] Trade rejected by RiskEngine: {reason_text}",
            _idea_frame(state),
            account,
            positions,
            orders,
        )

    state["store"].set_pending(idea)
    state, account, positions, orders = _frames(state)
    return (
        state,
        f"[{_engine_status(state)}] Trade suggestion ready. Review risk text. Approve required before paper order.",
        _idea_frame(state),
        account,
        positions,
        orders,
    )


def approve_trade(state: dict | None):
    state = _ensure_state(state)
    config: AppConfig = state["config"]
    broker: PaperBroker = state["broker"]

    halt = _check_halt(state)
    if halt:
        broker.append_audit("approve_trade", "blocked", reason=halt)
        state, account, positions, orders = _frames(state)
        return (
            state,
            f"Engine HALTED: {halt}. Approval blocked.",
            _idea_frame(state),
            account,
            positions,
            orders,
        )

    if config.trading_mode != "paper":
        state, account, positions, orders = _frames(state)
        return (
            state,
            "Live mode blocked. Paper mode only by default.",
            _idea_frame(state),
            account,
            positions,
            orders,
        )

    idea = state["store"].approve_pending()
    if idea is None:
        state, account, positions, orders = _frames(state)
        return (
            state,
            "No pending trade idea to approve.",
            _idea_frame(state),
            account,
            positions,
            orders,
        )

    price = state.get("last_price")
    if price is None or price <= 0:
        state, account, positions, orders = _frames(state)
        return (
            state,
            "No valid market price in state. Generate suggestion first.",
            _idea_frame(state),
            account,
            positions,
            orders,
        )

    receipt = broker.submit_order({**idea.to_dict(), "price": price})
    broker.append_audit(
        "order_approved",
        "filled",
        symbol=idea.symbol,
        side=idea.side,
        qty=float(idea.qty),
        metadata=receipt,
    )
    state, account, positions, orders = _frames(state)
    return (
        state,
        f"[{_engine_status(state)}] Approved and submitted paper order {receipt['order_id']}.",
        _idea_frame(state),
        account,
        positions,
        orders,
    )


def reject_trade(state: dict | None, reason: str):
    state = _ensure_state(state)
    broker: PaperBroker = state["broker"]
    if state["store"].pending_trade_idea is None:
        state, account, positions, orders = _frames(state)
        return (
            state,
            "No pending trade idea to reject.",
            _idea_frame(state),
            account,
            positions,
            orders,
        )

    idea = state["store"].pending_trade_idea
    state["store"].reject_pending(reason or "user_rejected")
    broker.append_audit(
        "order_rejected",
        "rejected",
        symbol=idea.symbol,
        side=idea.side,
        qty=float(idea.qty),
        reason=reason or "user_rejected",
    )
    state, account, positions, orders = _frames(state)
    return (
        state,
        f"[{_engine_status(state)}] Pending suggestion rejected. No trade submitted.",
        _idea_frame(state),
        account,
        positions,
        orders,
    )


def toggle_kill_switch(state: dict | None):
    state = _ensure_state(state)
    config: AppConfig = state["config"]
    broker: PaperBroker = state["broker"]
    config.risk.kill_switch_enabled = not config.risk.kill_switch_enabled
    action = "ENABLED" if config.risk.kill_switch_enabled else "DISABLED"
    broker.append_audit("kill_switch_toggle", action)
    broker.append_risk_event("kill_switch", "critical" if config.risk.kill_switch_enabled else "info", action)
    state, account, positions, orders = _frames(state)
    return (
        state,
        f"[{_engine_status(state)}] Kill switch {action}.",
        _idea_frame(state),
        account,
        positions,
        orders,
    )


def reset_halt(state: dict | None):
    state = _ensure_state(state)
    config: AppConfig = state["config"]
    broker: PaperBroker = state["broker"]
    old_reason = config.halt_reason
    config.halt_reason = ""
    config.risk.kill_switch_enabled = False
    broker.append_audit("reset_halt", "cleared", reason=old_reason)
    state, account, positions, orders = _frames(state)
    return (
        state,
        f"[{_engine_status(state)}] Halt cleared (was: {old_reason or 'kill_switch'}).",
        _idea_frame(state),
        account,
        positions,
        orders,
    )


def refresh_account(state: dict | None):
    state, account, positions, orders = _frames(state)
    return state, _idea_frame(state), account, positions, orders


def refresh_dashboard(state: dict | None):
    state, account, positions, orders = _frames(state)
    return (
        state,
        _idea_frame(state),
        account,
        positions,
        orders,
        _fills_frame(state),
        _pnl_frame(state),
        _risk_events_frame(state),
        _model_leaderboard_frame(),
        _benchmark_status_text(),
        _model_status_text(state),
        _safety_status_text(state),
    )


def reset_paper_account(state: dict | None, confirmation: str):
    state = _ensure_state(state)
    broker: PaperBroker = state["broker"]
    if confirmation != "RESET PAPER":
        state, account, positions, orders = _frames(state)
        return (
            state,
            "Reset blocked. Type RESET PAPER to confirm paper-account reset.",
            _idea_frame(state),
            account,
            positions,
            orders,
        )

    broker.reset_account()
    state["store"].clear_pending()
    broker.append_audit("reset_paper_account", "confirmed")
    broker.append_risk_event("paper_account_reset", "info", "user_confirmed_reset")
    state, account, positions, orders = _frames(state)
    return (
        state,
        "Paper account reset after explicit confirmation.",
        _idea_frame(state),
        account,
        positions,
        orders,
    )


def save_onboarding(
    state: dict | None,
    broker_backend: str,
    initial_cash: float,
    tickers: str,
    max_daily_loss: float,
    max_drawdown: float,
    max_position_frac: float,
    max_cvar: float,
    allow_shorts: bool,
    weekly_goal: float,
    approval_mode: str,
):
    state = _ensure_state(state)
    user_cfg = {
        "broker_backend": broker_backend,
        "initial_cash": initial_cash,
        "tickers": [t.strip().upper() for t in tickers.split(",") if t.strip()],
        "risk_limits": {
            "max_daily_loss_pct": max_daily_loss,
            "max_drawdown_pct": max_drawdown,
            "max_position_fraction": max_position_frac,
            "max_cvar_95": max_cvar,
        },
        "allow_shorts": allow_shorts,
        "benchmark_weekly_goal": weekly_goal,
        "approval_mode": approval_mode,
    }
    save_user_config(user_cfg)
    apply_user_config(state["config"], user_cfg)
    return state, f"Config saved. Ticker={user_cfg['tickers']}, Cash=${initial_cash:.0f}"


with gr.Blocks(title="Marketify Paper Engine") as demo:
    gr.Markdown(
        """
        # Marketify Paper Engine
        Experimental trade-suggestion system with mandatory human approval.

        **Safety notice:**
        """
    )
    gr.Markdown(RISK_DISCLAIMER)
    gr.Markdown("Benchmark goal can track 1% weekly paper performance, but benchmark is not guarantee.")

    app_state = gr.State(value=None)

    with gr.Tabs():
        # ── Setup Tab ──────────────────────────────────────────
        with gr.Tab("Setup"):
            gr.Markdown("### Onboarding Wizard")
            ob_broker = gr.Dropdown(["paper", "ibkr"], value="paper", label="Broker Backend")
            ob_cash = gr.Number(label="Initial Cash ($)", value=10000.0)
            ob_tickers = gr.Textbox(label="Tickers (comma-separated)", value="AAPL")
            with gr.Row():
                ob_daily_loss = gr.Number(label="Max Daily Loss %", value=0.02)
                ob_drawdown = gr.Number(label="Max Drawdown %", value=0.10)
                ob_position = gr.Number(label="Max Position Fraction", value=0.10)
                ob_cvar = gr.Number(label="Max CVaR 95", value=0.02)
            with gr.Row():
                ob_shorts = gr.Checkbox(label="Allow Shorts", value=True)
                ob_weekly = gr.Number(label="Benchmark Weekly Goal", value=0.01)
                ob_approval = gr.Dropdown(["manual", "auto"], value="manual", label="Approval Mode")
            ob_save_btn = gr.Button("Save Configuration", variant="primary")
            ob_status = gr.Textbox(label="Setup Status", interactive=False)

            ob_save_btn.click(
                save_onboarding,
                inputs=[
                    app_state, ob_broker, ob_cash, ob_tickers,
                    ob_daily_loss, ob_drawdown, ob_position, ob_cvar,
                    ob_shorts, ob_weekly, ob_approval,
                ],
                outputs=[app_state, ob_status],
            )

        # ── Trading Tab ────────────────────────────────────────
        with gr.Tab("Trading"):
            with gr.Row():
                ticker = gr.Textbox(label="Ticker", value="AAPL")
                interval = gr.Dropdown(["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d"], value="5m", label="Interval")
                period = gr.Dropdown(["7d", "14d", "30d", "60d"], value="60d", label="Period")
                prepost = gr.Checkbox(label="Include Pre/Post Market", value=False)

            with gr.Row():
                suggest_btn = gr.Button("Generate Suggestion", variant="primary")
                approve_btn = gr.Button("Approve")
                reject_btn = gr.Button("Reject")
                kill_switch_btn = gr.Button("Toggle Kill Switch", variant="stop")
                reset_halt_btn = gr.Button("Reset Halt")

            reject_reason = gr.Textbox(label="Reject reason", value="user_rejected")
            status = gr.Textbox(label="Status", interactive=False)
            suggestion = gr.Dataframe(label="Pending Trade Idea", interactive=False)
            account = gr.Dataframe(label="Account", interactive=False)
            positions = gr.Dataframe(label="Positions", interactive=False)
            orders = gr.Dataframe(label="Open Orders / Orders", interactive=False)
            fills = gr.Dataframe(label="Fills", interactive=False)
            pnl_summary = gr.Dataframe(label="PnL Summary", interactive=False)
            risk_events = gr.Dataframe(label="Risk Events", interactive=False)
            model_leaderboard = gr.Dataframe(label="Model Leaderboard", interactive=False)
            benchmark_status = gr.Textbox(label="Benchmark Status (weekly core / daily stress)", interactive=False)
            local_model_status = gr.Textbox(label="Local Model Status", interactive=False)
            safety_status = gr.Textbox(label="Safety Status", interactive=False)

            with gr.Row():
                refresh_dashboard_btn = gr.Button("Refresh Dashboard")
                reset_confirm = gr.Textbox(label="Reset paper account confirmation", value="", placeholder="Type RESET PAPER")
                reset_paper_btn = gr.Button("Reset Paper Account", variant="stop")

            timer = gr.Timer(value=30.0)

            suggest_btn.click(
                generate_trade_suggestion,
                inputs=[app_state, ticker, interval, period, prepost],
                outputs=[app_state, status, suggestion, account, positions, orders],
            )

            timer.tick(
                refresh_dashboard,
                inputs=[app_state],
                outputs=[
                    app_state,
                    suggestion,
                    account,
                    positions,
                    orders,
                    fills,
                    pnl_summary,
                    risk_events,
                    model_leaderboard,
                    benchmark_status,
                    local_model_status,
                    safety_status,
                ],
            )

            approve_btn.click(
                approve_trade,
                inputs=[app_state],
                outputs=[app_state, status, suggestion, account, positions, orders],
            )

            reject_btn.click(
                reject_trade,
                inputs=[app_state, reject_reason],
                outputs=[app_state, status, suggestion, account, positions, orders],
            )

            kill_switch_btn.click(
                toggle_kill_switch,
                inputs=[app_state],
                outputs=[app_state, status, suggestion, account, positions, orders],
            )

            reset_halt_btn.click(
                reset_halt,
                inputs=[app_state],
                outputs=[app_state, status, suggestion, account, positions, orders],
            )

            reset_paper_btn.click(
                reset_paper_account,
                inputs=[app_state, reset_confirm],
                outputs=[app_state, status, suggestion, account, positions, orders],
            )

            refresh_dashboard_btn.click(
                refresh_dashboard,
                inputs=[app_state],
                outputs=[
                    app_state,
                    suggestion,
                    account,
                    positions,
                    orders,
                    fills,
                    pnl_summary,
                    risk_events,
                    model_leaderboard,
                    benchmark_status,
                    local_model_status,
                    safety_status,
                ],
            )

    demo.load(
        refresh_dashboard,
        inputs=[app_state],
        outputs=[
            app_state,
            suggestion,
            account,
            positions,
            orders,
            fills,
            pnl_summary,
            risk_events,
            model_leaderboard,
            benchmark_status,
            local_model_status,
            safety_status,
        ],
    )


if __name__ == "__main__":
    demo.launch()
