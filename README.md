# Marketify

Marketify is a modular paper-trading research app. It loads market history, builds technical/news/risk features, loads the local XGBoost artifact when present, proposes a paper trade, explains risk, and waits for user approval before any order reaches the local paper broker.

This project is experimental. It is not financial advice. Trading can lose money. Benchmark results are research measurements only, not profit promises.

## Safety defaults

- Paper trading is the default path.
- Live-money trading is disabled by default.
- A generated idea is only a pending suggestion.
- User approval is required before any paper order is submitted.
- Broker readiness checks for Alpaca and IBKR are read-only unless an explicit guarded paper path is added.
- Performance targets, including the weekly benchmark and daily stress check, are benchmarks only.

## What the app does

1. Fetches market data with `yfinance`.
2. Adds technical, news, sentiment, macro/regime, and risk features.
3. Loads the latest synced XGBoost artifact for the selected ticker.
4. Generates a `TradeIdea` with side, quantity, confidence, expected return, CVaR, stops, and risk text.
5. Runs pre-trade risk checks.
6. Waits for Approve or Reject.
7. Persists paper orders, fills, positions, account state, PnL, audit events, and risk events in SQLite.

## Prerequisites

- Python 3.12 recommended.
- Git.
- Optional CUDA GPU for heavy training or benchmark experiments.
- Optional Alpaca paper credentials for read-only paper readiness checks.
- Optional IBKR TWS/Gateway paper session for read-only IBKR readiness checks.

## Install locally

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-colab.txt -q
.\.venv\Scripts\python.exe -m pip install -e . -q
```

If the virtual environment is broken, use the repair script:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/repair_local_venv.ps1
```

## Select local Python, not a remote notebook kernel

Use the local virtual environment for all terminal commands and notebooks.

In VS Code:

1. Open Command Palette.
2. Run `Python: Select Interpreter`.
3. Pick `.venv\Scripts\python.exe` from this repo.
4. For notebooks, click the kernel picker and select the same `.venv` Python kernel.

Verify before any benchmark run:

```powershell
.\.venv\Scripts\python.exe -c "import sys, platform; print(sys.executable); print(platform.platform())"
```

If the path is not this repo's `.venv\Scripts\python.exe`, stop and switch interpreters first.

## Detect GPU

```powershell
nvidia-smi
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_GPU')"
.\.venv\Scripts\python.exe -c "import xgboost as xgb; print(xgb.__version__)"
```

If `nvidia-smi` works but PyTorch prints `False` or `NO_GPU`, the local virtual environment has CPU-only PyTorch. Install a CUDA PyTorch wheel that matches your driver before any GPU benchmark run. For the verified Windows CUDA 12.8 path:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
```

Then rerun the PyTorch check. Continue only when PyTorch sees the GPU, or when `nvidia-smi` confirms no usable NVIDIA GPU exists. If CUDA is unavailable, skip optional heavy retraining. Validation, UI proof, broker skip-proof, and report hygiene still run locally.

## Low-RAM local mode

Use low-RAM mode for local benchmark work. It runs one stage at a time, uses single-worker XGBoost, uses `float32` prediction arrays where safe, skips recurrent branches on CPU, and avoids extra feature-ablation retraining.

```powershell
$env:LOCAL_LOW_RAM_MODE = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
```

Only set GPU forcing after CUDA is confirmed:

```powershell
$env:LOCAL_FORCE_GPU = "1"
```

Do not set `LOCAL_FORCE_GPU` when CUDA reports `False`.

## Run the UI

```powershell
.\.venv\Scripts\python.exe app.py
```

Open the printed local Gradio URL. Use the Setup tab for paper account/risk settings. Use the Trading tab to generate, approve, reject, refresh, reset, or halt.

## UI approval flow

1. Click Generate Suggestion.
2. Read the pending idea, reason, confidence, CVaR, stop loss, take profit, and risk warning.
3. Click Reject to clear the pending idea without submitting an order.
4. Click Approve to submit only that pending idea to the local paper broker.
5. Click Refresh Dashboard to view account, positions, orders, fills, PnL, risk events, model status, and safety status.

Reject flow should leave cash, equity, positions, orders, and fills unchanged. Approve flow should create a filled paper order, a fill, a position, and updated account/PnL rows.

## State and reset

The local paper broker persists state in SQLite at `data/marketify_paper.db` by default.

To reset safely from the UI:

1. Open the Trading tab.
2. Type `RESET PAPER` in the reset confirmation field.
3. Click Reset Paper Account.

Reset is blocked unless the exact confirmation text is entered.

## Environment setup

Copy the example file and fill only what you need:

```powershell
Copy-Item .env.example .env
```

Important defaults:

- `PAPER_MODE=true`
- `LIVE_TRADING=false`
- `MARKETIFY_DB_PATH=data/marketify_paper.db`

Do not commit `.env` or `user_config.yaml`.

## Broker modes

### `local_paper`

Default mode. Uses SQLite-backed `PaperBroker`. Supports local paper orders, fills, positions, PnL, audit log, risk events, restart/reload proof, and reset confirmation.

### `alpaca_paper`

Read-only readiness adapter. Requires `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY`, and paper endpoint `https://paper-api.alpaca.markets`. Live endpoint is blocked by default. Order submission is disabled in the readiness adapter.

### `ibkr_paper`

Read-only readiness adapter. Requires a running IBKR TWS/Gateway paper session and optional `IBKR_ACCOUNT_ID`, `IBKR_HOST`, `IBKR_PORT`, and `IBKR_CLIENT_ID`. `IBKR_PAPER=true` and read-only mode are default. Order submission is blocked while read-only is enabled.

## Artifact sync and model loading

The app looks for the latest local XGBoost artifact in this order:

1. `artifacts/latest_<TICKER>.json`
2. `reports/artifact_manifest.json`
3. `artifacts/xgb_<TICKER>.pkl`
4. newest `artifacts/training_runs/*/xgb_<TICKER>.pkl`

When an artifact is found, the UI shows `LOCAL_MODEL_LOAD=YES`, `loaded_model_type=xgb`, and `inference_smoke=PASS` after a successful suggestion smoke check.

## Test and validation commands

Run local tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q
```

Run validation and broker readiness:

```powershell
.\.venv\Scripts\python.exe scripts/validate_marketify.py
```

Run local UI/model readiness after browser proof:

```powershell
.\.venv\Scripts\python.exe scripts/local_run_readiness.py --browser-opened YES --browser-interactive-proof YES --browser-note "local browser proof rerun"
```

## Training and benchmark commands

Run train-if-missing and artifact checks:

```powershell
.\.venv\Scripts\python.exe scripts/train_and_check.py
```

Run same-path benchmark comparison:

```powershell
$env:LOCAL_LOW_RAM_MODE = "1"
$env:LOCAL_FORCE_GPU = "1"  # only after torch.cuda.is_available() is True
.\.venv\Scripts\python.exe scripts/compare_models.py --low-ram
```

This cheap path runs XGBoost plus Ridge first and skips recurrent/fusion heavy branches unless CUDA is available and explicitly requested. It may try up to five small gate-threshold iterations for the daily stress benchmark. Keep a change only when weekly benchmark stays pass, expectancy stays positive, drawdown stays safe, and daily return improves.

If CUDA is not available and memory is low, do not brute-force CPU recurrent work. Keep the latest completed benchmark truth and report the skip honestly.

## Common troubleshooting

- Missing local model: run `scripts/train_and_check.py` or sync the latest `artifacts/latest_<TICKER>.json` and model files.
- `LOCAL_MODEL_LOAD=NO`: check that artifact paths in `artifacts/latest_<TICKER>.json` point to files that exist locally.
- Wrong interpreter: re-run `Python: Select Interpreter`, choose `.venv\Scripts\python.exe`, then rerun the `sys.executable` check.
- No GPU: keep `LOCAL_LOW_RAM_MODE=1`, leave `LOCAL_FORCE_GPU` unset, and skip recurrent/heavy branches.
- Timeout or RAM pressure: stop the run, keep prior honest report values, and do not retry with larger CPU fanout.
- Alpaca is skipped: set paper credentials in `.env`; keep `LIVE_TRADING=false`.
- IBKR is skipped: start TWS/Gateway in paper mode and set host/port/account values.
- No trade idea generated: model signal may be weak, risk gate may reject it, or data may be unavailable.
- Daily stress fails: this is allowed as an honest optional stress result; do not relabel it as pass unless a fresh run crosses the target.

## Reports

Key status reports live in `reports/`:

- `NEXT_STATUS.md`: current safety, benchmark, broker, browser, and blocker labels.
- `validation_summary.md`: local validation result.
- `broker_validation.md`: local mock broker plus official broker skip/pass status.
- `ui_visual_proof.md`: browser/UI proof summary.
- `daily_stress_iteration_log.md`: daily stress status and blocker.

Runtime outputs, caches, databases, and local secrets are ignored by Git.
