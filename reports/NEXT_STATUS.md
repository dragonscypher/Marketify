# NEXT_STATUS

updated_at: 2026-04-29T16:05:00+00:00
source_commit_before_cleanup: 98449b6
CORE_PAPER_ENGINE: YES
STRICT_LOCAL_DONE: YES
FINAL_POLISH_DONE: YES
FULL_EXTERNAL_CLOSURE: YES
LOCAL_MODEL_LOAD: YES
loaded_model_type: xgb
inference_smoke: PASS
REAL_BROWSER_CLICK_PROOF: YES
ALPACA_PAPER_PROVEN: SKIP
ALPACA_SKIP_REASON: Alpaca paper env unavailable
IBKR_READ_ONLY_PROVEN: SKIP
IBKR_SKIP_REASON: IBKR env unavailable
MOCK_BROKER_PROVEN: YES
WEEKLY_BENCHMARK: PASS
weekly_return_pct: 1.9322
DAILY_STRESS: FAIL
daily_return_pct: 0.0119
daily_stress_target_pct: 1.0000
daily_stress_exact_blocker: daily_return_pct 0.0119 below 1.0
PII_GITHUB_AUDIT: FIXED
PII_GITHUB_AUDIT_REASON: cleanup removed tracked runtime zip, duplicate/bulky research notebooks, notebook output spam, and machine-specific tracked report paths; high-risk scan now reports zero findings.
paper_only_default: YES
live_order_path_enabled: NO
approval_required: YES
profit_guarantee: NO
weekly_1pct_is_benchmark_only: YES
daily_1pct_is_optional_stress_benchmark_only: YES
local_gpu_available: NO
local_gpu_used: NO
daily_stress_loop_attempted: NO
daily_stress_loop_skip_reason: CUDA unavailable; compare_models.py CPU run exceeded 1200000 ms at recurrent prediction stage and was killed to protect local RAM. Latest completed benchmark truth retained.

## Local Browser DOM/Click Rerun
- Browser URL: http://127.0.0.1:7860/
- Safety disclaimer visible: Experimental paper-trading engine. No financial advice. Trading involves risk, including loss of principal. Target metrics are benchmarks, not guarantees.
- Benchmark disclaimer visible: Benchmark goal can track 1% weekly paper performance, but benchmark is not guarantee.
- Reset proof setup: typed RESET PAPER, clicked Reset Paper Account, clicked Refresh Dashboard, clicked Reset Halt.
- Clean account before reject proof: cash 10000, equity 10000, realized_pnl 0, unrealized_pnl 0.
- Browser data combo: AAPL, interval 1m, period 7d, no pre/post market. No risk limits, leverage, architecture, model family, or exit rules changed.
- Generated idea for reject proof: AAPL buy qty 3, confidence 1, expected_return 0.0003389495250303298, cvar_95 0.0019211889435489793.
- Reject click status: [ACTIVE] Pending suggestion rejected. No trade submitted.
- Reject proof after Refresh Dashboard: account stayed cash 10000, equity 10000, realized_pnl 0, unrealized_pnl 0; no visible position, order, or fill rows were created by the rejected suggestion.
- Generated idea for approve proof: AAPL buy qty 3, confidence 1, expected_return 0.0003389495250303298, cvar_95 0.0019211889435489793.
- Approve click status: [ACTIVE] Approved and submitted paper order f5b12808-f64f-4b26-b578-abebd6c88ecc.
- Approve account visible: cash 9193.37302807701, equity 9999.758068360214, realized_pnl 0, unrealized_pnl -0.0806385040282862.
- Position visible: AAPL qty 3, avg_cost 268.82189292907714, mark_price 268.7950134277344, market_value 806.3850402832031.
- Order visible: order_id f5b12808-f64f-4b26-b578-abebd6c88ecc, AAPL buy qty 3, status filled.
- Fill visible: fill_id 42736fec-ed60-48e3-ab77-ccb311768d92, order_id f5b12808-f64f-4b26-b578-abebd6c88ecc, AAPL buy qty 3.
- Restart reload proof: app process killed/restarted, browser reopened, Trading tab displayed same account, AAPL position, filled order, fill, and PnL values from SQLite.

## Broker Proof
- local_paper_internal: PASS - SQLite PaperBroker accounting invariant.
- local_mock_http: PASS - deterministic mock broker HTTP contract and accounting invariant.
- alpaca_paper_read_only: SKIP - Alpaca paper env unavailable.
- ibkr_paper_read_only: SKIP - IBKR env unavailable.
- official_broker_live_orders_submitted: NO.

## Exact Local Commands Run
- python -m pip install -r requirements-colab.txt -q
- python -m pip install -e . -q
- python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_GPU')"
- python -m pytest tests/ -q
- python scripts/validate_marketify.py
- python scripts/train_and_check.py
- python scripts/compare_models.py (timed out after 1200000 ms on CPU and was killed)
- python scripts/local_run_readiness.py --browser-opened YES --browser-interactive-proof YES --browser-note "local browser proof rerun: DOM/click reset, reject, approve, refresh, restart reload verified"
- Python high-risk tracked-file scan.
- Python notebook-output scan.
- Browser DOM/click actions at http://127.0.0.1:7860/.
- Broker readiness read-only check via scripts.validate_marketify.run_broker_readiness().

## Exact Errors / Limitations
- Initial GPU check before dependency install: ModuleNotFoundError: No module named 'torch'. Resolved by installing requirements-colab.txt.
- Final GPU check: torch.cuda.is_available() False; device NO_GPU.
- train_and_check warning: InconsistentVersionWarning for Ridge pickle sklearn 1.6.1 loaded under sklearn 1.8.0. Command result still PASS.
- compare_models.py: timed out after 1200000 ms on CPU at recurrent prediction stage; killed to protect local resources. No new local compare PASS claimed from that killed run.
- DAILY_STRESS: FAIL - daily_return_pct 0.0119 below 1.0 in latest completed benchmark report.
- ALPACA_PAPER_PROVEN: SKIP - Alpaca paper env unavailable.
- IBKR_READ_ONLY_PROVEN: SKIP - IBKR env unavailable.

## Final Remaining Blockers
- None for core paper engine, strict local closure, final polish, real browser paper flow proof, local validation, or mock broker proof.
- Optional only: daily 1% stress benchmark remains FAIL honestly.
- Official Alpaca/IBKR external proofs remain SKIP until paper/read-only environments are provided.
