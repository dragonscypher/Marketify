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

## Install (Local — Windows)

Local = editing + small tests only. Training/backtests/benchmarks run in Colab.

```powershell
# First time or broken venv:
powershell -ExecutionPolicy Bypass -File scripts/repair_local_venv.ps1

# Or manual:
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e . -r requirements.txt
```

Only light deps installed locally. Heavy deps (torch, chronos, transformers) live in `requirements-colab.txt` and install in Colab only.

## Run UI

```powershell
python app.py
```

## Run Tests (Local)

```powershell
python -m pytest tests/ -q --tb=short
python scripts/colab_check.py   # should print NOT Colab
```

Do NOT run `MARKETIFY_REQUIRE_COLAB=1 python scripts/colab_run_all.py` locally.

## Notes on Notebooks

- `marketify.ipynb` and `Marketify_Pro.ipynb` archived to `archive/notebooks/`.
- Notebooks are research artifacts only — not imported by production code.
- Any useful logic must be ported into `marketify/` modules with tests.
- Production code lives in `marketify/`, `scripts/`, and `app.py`.

## Colab Workflow

Heavy work (training, backtests, benchmarks, model sweeps) runs in Google Colab.
Colab installs `requirements-colab.txt` (includes torch, chronos, transformers).

```bash
# In Colab:
python scripts/colab_check.py
MARKETIFY_REQUIRE_COLAB=1 python scripts/colab_run_all.py
```

See [COLAB_RUNBOOK.md](COLAB_RUNBOOK.md) for full steps.

## Dependency Split

| File                     | Where         | What                                              |
| ------------------------ | ------------- | ------------------------------------------------- |
| `requirements.txt`       | Local + Colab | Light deps (pandas, xgboost, ta, gradio, etc.)    |
| `requirements-colab.txt` | Colab only    | Heavy deps (torch, chronos, transformers, HF hub) |

Project uses `ta` (pure Python), NOT `ta-lib` (C library).
