# NEXT_STATUS

timestamp: 2026-04-30T00:47:52.152117+00:00
commit_hash: 8aec818
CORE_PAPER_ENGINE: YES
LOCAL_KERNEL_FIXED: YES
PYTHON_EXECUTABLE: .venv/Scripts/python.exe
LOCAL_GPU_AVAILABLE: NO
LOCAL_GPU_USED: NO
LOCAL_GPU_NAME: NO_GPU
LOCAL_LOW_RAM_MODE: YES
RECURRENT_BRANCH_SKIPPED: YES
skip_reason: LOCAL_LOW_RAM_MODE=1 cheap path skips recurrent/fusion heavy branches
LOCAL_MODEL_LOAD: YES
local_artifact_present: YES
latest_xgb_artifact_path: artifacts/training_runs/20260429T152352Z_145e09b977d8/xgb_AAPL.pkl
latest_ridge_artifact_path: artifacts/training_runs/20260429T152352Z_145e09b977d8/ridge_AAPL.pkl
artifact_timestamp: 20260429T152352Z
config_hash: 145e09b977d8
loaded_artifact_path: artifacts/training_runs/20260429T152352Z_145e09b977d8/xgb_AAPL.pkl
loaded_model_type: xgb
loaded_model_class: XGBModel
inference_smoke: PASS
LOCAL_UI_VISUAL_PROOF: REAL_BROWSER_DOM_CLICK
REAL_BROWSER_CLICK_PROOF: YES
REAL_BROWSER_CLICK_PROOF_REASON: previous real browser DOM/click proof retained; current local runtime readiness rerun was scripted only
APPROVAL_GATE_WORKS: YES
REJECT_NO_TRADE_PROVEN: YES
APPROVE_FILL_PROVEN: YES
RESTART_RELOAD_PROVEN: YES
MOCK_BROKER_PROVEN: YES
ALPACA_PAPER_PROVEN: SKIP
IBKR_READ_ONLY_PROVEN: SKIP
WEEKLY_BENCHMARK: PASS
DAILY_STRESS: FAIL
weekly_return_pct: 1.3006
daily_return_pct: -0.0357
sharpe: 0.6981
max_drawdown_pct: 0.8666
trade_count: 32
expectancy: 2.664591
STRICT_LOCAL_DONE: YES
FULL_EXTERNAL_CLOSURE: YES
daily_stress_exact_blocker: daily_return_pct below 1.0
exact_blocker: none
paper_only_default: YES
live_order_path_enabled: NO
approval_required: YES
browser_tool_used: YES
browser_interactive_proof: NO
browser_note: local GPU/runtime rerun

## Exact Local Commands Run
- .venv/Scripts/python.exe -m pip install -r requirements-colab.txt -q
- .venv/Scripts/python.exe -m pip install -e . -q
- .venv/Scripts/python.exe -c "import sys, platform; print(sys.executable); print(platform.platform())"
- .venv/Scripts/python.exe -c "import torch; print('CUDA', torch.cuda.is_available()); print('GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_GPU')"
- .venv/Scripts/python.exe -c "import xgboost as xgb; print(xgb.__version__)"
- .venv/Scripts/python.exe -m pytest tests/ -q
- .venv/Scripts/python.exe scripts/validate_marketify.py
- LOCAL_LOW_RAM_MODE=1 .venv/Scripts/python.exe scripts/compare_models.py --low-ram
- .venv/Scripts/python.exe scripts/local_run_readiness.py --browser-opened YES --browser-note "local GPU/runtime rerun"
- train_and_check skipped: XGB/Ridge artifacts already present and valid; no retrain-everything run needed
