# NEXT_STATUS

timestamp: 2026-04-28T15:05:00+00:00
base_commit_before_final_external_closure: 9ba405a
pending_final_commit_hash: generated_after_this_report

CORE_PAPER_ENGINE: YES
STRICT_LOCAL_DONE: YES
FINAL_POLISH_DONE: YES
FULL_EXTERNAL_CLOSURE: YES

LOCAL_MODEL_LOAD: YES
local_artifact_present: YES
latest_xgb_artifact_path: artifacts/training_runs/20260427T020631Z_145e09b977d8/xgb_AAPL.pkl
loaded_artifact_path: artifacts\training_runs\20260427T020631Z_145e09b977d8\xgb_AAPL.pkl
loaded_model_type: xgb
loaded_model_class: XGBModel
inference_smoke: PASS

REAL_BROWSER_CLICK_PROOF: YES
REAL_BROWSER_CLICK_PROOF_REASON: DOM/click tools available; real browser clicked Trading, Generate Suggestion, Reject, Generate Suggestion, Approve, Refresh Dashboard; app was restarted and the persisted paper state was visible after reload.
LOCAL_UI_VISUAL_PROOF: REAL_BROWSER_DOM_CLICK
APPROVAL_GATE_WORKS: YES
REJECT_NO_TRADE_PROVEN: YES
APPROVE_FILL_PROVEN: YES
RESTART_RELOAD_PROVEN: YES

MOCK_BROKER_PROVEN: YES
ALPACA_PAPER_PROVEN: SKIP
ALPACA_SKIP_REASON: Alpaca paper env unavailable
IBKR_READ_ONLY_PROVEN: SKIP
IBKR_SKIP_REASON: IBKR env unavailable

WEEKLY_BENCHMARK: PASS
DAILY_STRESS: FAIL
daily_return_pct: 0.0119
daily_stress_target_pct: 1.0000
daily_stress_exact_blocker: daily_return_pct 0.0119 below 1.0
daily_stress_iterations_changed: 0
daily_stress_iteration_decision: no safe optional knob change applied; baseline weekly PASS and strict local done preserved

PII_GITHUB_AUDIT: FIXED
PII_GITHUB_AUDIT_REASON: no active secrets or personal data found; preventive cleanup added user_config.yaml to .gitignore so local onboarding state is not pushed.

paper_only_default: YES
live_order_path_enabled: NO
approval_required: YES
official_broker_live_orders_submitted: NO
profit_guarantee: NO
weekly_1pct_is_benchmark_only: YES
daily_1pct_is_optional_stress_benchmark_only: YES

## Real Browser DOM/Click Evidence
- Browser URL: http://127.0.0.1:7860/
- Safety disclaimer visible: "Experimental paper-trading engine. No financial advice. Trading involves risk, including loss of principal. Target metrics are benchmarks, not guarantees."
- Benchmark disclaimer visible: "Benchmark goal can track 1% weekly paper performance, but benchmark is not guarantee."
- Generate Suggestion produced pending idea: AAPL buy qty 3, confidence 1, expected_return 0.0003357288078404963, cvar_95 0.004063294591088058.
- Reject click status: "[ACTIVE] Pending suggestion rejected. No trade submitted."
- Reject account stayed unchanged: cash 10000, equity 10000, realized_pnl 0, unrealized_pnl 0; pending table returned to headers only.
- Approve click status: "[ACTIVE] Approved and submitted paper order 1e7dc72a-a652-4a33-85d1-877dbdf4bf9e."
- Approve account visible: cash 9185.300691169397, equity 9999.755647224085, unrealized_pnl -0.08144549560546466.
- Position visible: AAPL qty 3, avg_cost 271.51213385009765, mark_price 271.4849853515625.
- Order visible: order_id 1e7dc72a-a652-4a33-85d1-877dbdf4bf9e, side buy, qty 3, status filled.
- Fill visible after Refresh Dashboard: fill_id fb1286b4-9df5-43f2-88b2-1a5d72460f68, order_id 1e7dc72a-a652-4a33-85d1-877dbdf4bf9e, AAPL buy qty 3.
- PnL Summary visible: equity 9999.755647224085, cash 9185.300691169397, realized_pnl 0, unrealized_pnl -0.08144549560546466.
- Restart reload proof: app process killed/restarted, browser reloaded, Trading tab displayed the same account, AAPL position, filled order, fill, and PnL summary from SQLite.

## Exact Local Commands Run
- python app.py
- real browser DOM/click actions at http://127.0.0.1:7860/
- python -m pytest tests/test_local_run_readiness.py -q
- python scripts/validate_marketify.py
- git grep / Python audits for secrets and PII patterns across tracked files, selected reports, user_config.yaml, git history high-risk patterns, and sync_payload.zip.

## Exact Errors / Limitations
- RESOLVED_BROWSER_ERROR: first real Generate Suggestion click raised TypeError: float() argument must be a string or a real number, not 'Series' at app.py last_price extraction. Fixed by normalizing latest numeric column extraction for yfinance MultiIndex/duplicate-column layouts.
- ALPACA_PAPER_PROVEN: SKIP - Alpaca paper env unavailable.
- IBKR_READ_ONLY_PROVEN: SKIP - IBKR env unavailable.
- DAILY_STRESS: FAIL - daily_return_pct 0.0119 below 1.0. This remains an optional stress benchmark failure, not a product/profit guarantee.
- GHAS_SECRET_SCAN: unavailable because GitHub Advanced Security is not enabled for the repository; local/manual scan completed and found no active secrets.

## Final Remaining Blockers
- None for core paper engine, strict local proof, or final external closure.
- Optional only: daily 1% stress benchmark remains FAIL honestly.
- Official Alpaca/IBKR external proofs remain SKIP until the required paper/read-only environments are provided.
