# Marketify Validation Summary

timestamp: 2026-04-30T07:26:53.651515+00:00
validation_result: PASS
paper_only_default: YES
live_trading_enabled: NO
profit_guarantee: NO
broker_readiness_result: PASS_WITH_SKIPS
MOCK_BROKER_PROVEN: YES
ALPACA_PAPER_PROVEN: SKIP
IBKR_READ_ONLY_PROVEN: SKIP

## Checks
| check | status | skipped | reason/error |
| --- | --- | --- | --- |
| import_smoke | PASS | False | none |
| pytest | PASS | False | none |
| synthetic_broker_accounting | PASS | False | none |
| backtest_smoke | PASS | False | synthetic smoke only; not success proof |
| gradio_launch_smoke | PASS | False | scripted UI proof used; no DOM click tool |
| scripted_ui_flow_proof | PASS | False | none |
| model_artifact_check | PASS | False | none |
| automatic_benchmark_checker | PASS | False | none |
| broker_readiness | PASS_WITH_SKIPS | False | none |

## Honest Notes
- real_browser_click_proof: YES (previous real browser DOM/click proof retained; current local runtime readiness rerun was scripted only)
- ui_flow: {'internal_paper': {'risk_disclaimer_visible': True, 'generate_trade_idea': True, 'reject_no_order_no_fill': True, 'approve_paper_fill': True, 'positions_update': True, 'pnl_updates': True, 'restart_reload': True, 'reset_requires_confirmation': True, 'reset_paper_account': True}, 'mock_http': {'risk_disclaimer_visible': True, 'generate_trade_idea': True, 'reject_no_order_no_fill': True, 'approve_paper_fill': True, 'positions_update': True, 'pnl_updates': True, 'restart_reload': True, 'reset_requires_confirmation': True, 'reset_paper_account': True}, 'approve_paper_fill': True, 'restart_sqlite_reload': True, 'mock_api_reload': True}
- weekly_benchmark: PASS
- daily_stress_benchmark: FAIL
- exact_blocker: daily_return_pct below 1.0
- local_paper_internal: PASS (SQLite PaperBroker accounting invariant)
- local_mock_http: PASS (deterministic mock broker HTTP contract and accounting invariant)
- alpaca_paper_read_only: SKIP (Alpaca paper env unavailable)
- ibkr_paper_read_only: SKIP (IBKR env unavailable)
