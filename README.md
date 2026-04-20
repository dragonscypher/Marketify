# Marketify Paper Engine

Experimental modular paper-trading engine.

Risk warning: Trading involves risk. Losses possible. No financial advice. Any performance target (for example 1% weekly paper benchmark) is benchmark only, not guarantee.

## Project Layout

- app.py: Gradio UI shell with trade suggestion and manual approval gate.
- marketify/config.py: shared config, TradeIdea, risk disclaimer.
- marketify/data/market_data.py: yfinance fallback for research/backtest.
- marketify/data/news_data.py: news provider abstraction and neutral fallback.
- marketify/features/technical.py: technical features pipeline.
- marketify/features/sentiment.py: FinBERT stub wrapper.
- marketify/models/xgb_model.py: baseline XGBoost rolling predictor.
- marketify/models/ensemble.py: suggestion assembly and CVaR utility.
- marketify/broker/base.py: broker abstraction.
- marketify/broker/paper.py: SQLite-backed paper broker accounting.
- marketify/broker/alpaca.py: stub only.
- marketify/broker/ibkr.py: stub only.
- marketify/risk/risk_engine.py: pre-trade risk checks.
- marketify/state/store.py: pending suggestion + approval/rejection memory.
- marketify/backtest/simulator.py: walk-forward simulator using broker + risk path.

## Safety Model

- Paper trading mode default.
- Live trading disabled by default.
- TradeIdea generated first.
- User must Approve or Reject.
- Only approved idea reaches PaperBroker submit_order.

## Paper Broker Accounting

- Separate cash ledger.
- Signed quantities (long positive, short negative).
- Weighted average cost tracking.
- Realized and unrealized PnL.
- Equity identity enforced: equity = cash + sum(qty * mark_price).
- Orders and fills ledger persisted in SQLite.

## Time-Stop Bug Fix

Backtest/simulator time-stop now uses broker.time_stop_bars.
No dependency on signal.time_stop_bars.

## Install

1. Create or activate Python environment.
2. Install deps.

```powershell
python -m pip install -r requirements.txt
```

## Run UI

```powershell
python app.py
```

## Run Tests

```powershell
pytest
```

## Notes on Notebooks

- `marketify.ipynb` and `Marketify_Pro.ipynb` archived to `archive/notebooks/`.
- Notebooks are research artifacts only — not imported by production code.
- Any useful logic must be ported into `marketify/` modules with tests.
- Production code lives in `marketify/`, `scripts/`, and `app.py`.

## Colab Workflow

Heavy work (training, backtests, benchmarks, full test suite) runs in Google Colab high-RAM runtime.

```bash
# Verify Colab connection
python scripts/colab_check.py

# Run full pipeline in Colab
MARKETIFY_REQUIRE_COLAB=1 python scripts/colab_run_all.py
```

See [COLAB_RUNBOOK.md](COLAB_RUNBOOK.md) for full steps.
