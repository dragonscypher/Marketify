# Colab Runbook — Marketify

## Prerequisites

- VS Code with Jupyter extension
- Google Colab high-RAM runtime (GPU optional)
- VS Code connected to Colab kernel via `Connect to Runtime`

## Steps

### 1. Connect VS Code to Colab

1. Open VS Code.
2. Open Command Palette → `Jupyter: Specify Jupyter Server for Connections`.
3. Paste Colab runtime URL.
4. Select the Colab Python interpreter as active kernel.

### 2. Confirm Colab Runtime

```bash
python scripts/colab_check.py
```

Expected output includes `COLAB_TAG: v<something>` and `runtime: Google Colab ✓`.

If output says `NOT Colab`, **STOP**. You are running locally. Switch interpreter.

### 3. Run Full Pipeline (Colab only)

```bash
MARKETIFY_REQUIRE_COLAB=1 python scripts/colab_run_all.py
```

This runs:
1. `pip install -r requirements.txt`
2. `pytest tests/ -q`
3. `python scripts/validate_marketify.py`
4. Train-if-missing pipeline
5. Benchmark checker
6. Generates `reports/NEXT_STATUS.md`

### 4. Launch Gradio UI

```bash
python app.py
```

Click the Gradio link (usually `http://127.0.0.1:7860` or a Colab public URL).

### 5. Test UI Flow

1. **Generate idea** — enter ticker (e.g., AAPL), click Generate.
2. **Reject** — click Reject. Verify no order or fill created.
3. **Generate again** — new idea appears.
4. **Approve** — click Approve. Verify paper fill, positions update, PnL update.
5. **Kill switch** — toggle kill switch ON. Try Generate. Should say HALTED.
6. **Reset halt** — click Reset Halt. Should re-enable.
7. **Restart app** — stop and rerun `python app.py`. Verify SQLite state reloads (positions, equity).

### 6. Check Reports

```bash
cat reports/NEXT_STATUS.md
cat reports/validation_summary.md
```

Verify PASS/FAIL for each step. Address any FAILs before proceeding.

## Heavy Work Definition

The following MUST run in Colab (not local):

- Model training (XGBoost, Ridge, LSTM, Chronos, TiRex, Mamba)
- Walk-forward backtests
- Benchmark sweeps
- Grid search / hyperparameter tuning
- Report generation over live market data
- Full pytest suite
- Anything using GPU or >4GB RAM

## Local Allowed

- File editing
- Static code inspection
- Small import smoke tests (<10s)
- No training locally
