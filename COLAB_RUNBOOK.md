# Colab Runbook — Marketify

## Architecture

- **GitHub repo** = source of truth for all code
- **`/content/Marketify`** = disposable Colab runtime copy (cloned from GitHub)
- **Google Drive** = persistent storage for reports/artifacts/models
- **Local machine** = editing + small tests only (no training/backtests)

## Rules

- **NEVER fake `COLAB_RELEASE_TAG`** — it is set by real Colab kernels only
- If `COLAB_RELEASE_TAG` missing and `/content` missing → you are NOT in Colab, stop
- If Colab tag missing but `/content` exists → possible Colab env quirk, proceed with caution
- Heavy work (training, backtests, benchmarks) = Colab only
- Local allowed: file editing, static analysis, small import tests (<10s)

## Quick Start (Colab)

### Option A: Bootstrap script (recommended)

In a Colab notebook cell:
```python
!git clone https://github.com/dragonscypher/Marketify.git /content/Marketify 2>/dev/null || (cd /content/Marketify && git pull)
!python /content/Marketify/scripts/colab_bootstrap.py
```

This clones, installs deps, runs check, and runs full pipeline.

### Option B: Manual steps

### 1. Connect VS Code to Colab

1. Open VS Code.
2. Command Palette → `Jupyter: Specify Jupyter Server for Connections`.
3. Paste Colab runtime URL.
4. Select Colab Python interpreter as active kernel.

### 2. Confirm Colab Runtime

```bash
python scripts/colab_check.py
```

Expected: `COLAB_TAG: v<something>` and `runtime: Google Colab ✓`.

If output says `NOT Colab ✗`:
- **STOP.** Do not run heavy work.
- Switch kernel to Colab or run inside Colab notebook.
- Do NOT set `COLAB_RELEASE_TAG` manually.

### 3. Run Full Pipeline (Colab only)

```bash
MARKETIFY_REQUIRE_COLAB=1 python scripts/colab_run_all.py
```

Runs:
1. `pip install -r requirements.txt`
2. `pytest tests/ -q`
3. `python scripts/validate_marketify.py`
4. Train-if-missing pipeline
5. Benchmark checker
6. Generates `reports/NEXT_STATUS.md` with exact runtime info

### 4. Launch Gradio UI

```bash
python app.py
```

### 5. Test UI Flow

1. **Generate idea** — enter ticker (e.g., AAPL), click Generate.
2. **Reject** — click Reject. Verify no order/fill.
3. **Generate again** — new idea appears.
4. **Approve** — click Approve. Verify paper fill, positions, PnL.
5. **Kill switch** — toggle ON. Try Generate → should say HALTED.
6. **Reset halt** — click Reset. Re-enables trading.
7. **Restart app** — verify SQLite state reloads.

### 6. Check Reports

```bash
cat reports/NEXT_STATUS.md
cat reports/benchmark_report.json
```

## Heavy Work Definition

Must run in Colab (NOT local):

- Model training (XGBoost, Ridge, LSTM, Chronos, TiRex, Mamba)
- Walk-forward backtests
- Benchmark sweeps
- Grid search / hyperparameter tuning
- Report generation over live market data
- Full pytest suite (when involving model inference)
- Anything using GPU or >4GB RAM

## Local Allowed

- File editing
- Static code inspection
- Small import smoke tests (<10s)
- `pytest tests/ -q` (lightweight unit tests only)
- No training, no backtests
