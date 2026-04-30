# UI Visual Proof

LOCAL_UI_VISUAL_PROOF: REAL_BROWSER_DOM_CLICK
REAL_BROWSER_CLICK_PROOF: YES
REAL_BROWSER_CLICK_PROOF_REASON: previous real browser DOM/click proof retained; current local runtime readiness rerun was scripted only
browser_tool_used: YES
browser_interactive_proof: NO
browser_note: local GPU/runtime rerun
screenshot_available: NO
browser_e2e_unavailable: NO

## Panels Visible
| panel | status | evidence |
| --- | --- | --- |
| risk disclaimer | PASS | RISK_DISCLAIMER markdown visible |
| benchmark disclaimer | PASS | 1% weekly benchmark not guarantee markdown visible |
| latest trade idea | PASS | Pending Trade Idea dataframe |
| approve button | PASS | Approve button |
| reject button | PASS | Reject button |
| account equity | PASS | Account dataframe includes equity |
| cash | PASS | Account dataframe includes cash |
| positions | PASS | Positions dataframe |
| open orders | PASS | Open Orders / Orders dataframe |
| fills | PASS | Fills dataframe |
| realized pnl | PASS | PnL Summary includes realized_pnl |
| unrealized pnl | PASS | PnL Summary includes unrealized_pnl |
| risk events | PASS | Risk Events dataframe |
| model leaderboard | PASS | Model Leaderboard dataframe |
| weekly benchmark PASS/FAIL | PASS | Benchmark Status textbox |
| daily stress PASS/FAIL | PASS | Benchmark Status textbox |
| kill switch | PASS | Toggle Kill Switch button |
| reset paper account with confirmation | PASS | Reset confirmation textbox + button |
| local model status | PASS | Local Model Status textbox |
| safety status | PASS | Safety Status textbox |

## Flow Proof
- disclaimer_visible: True
- idea_fields_present: True
- reject_no_trade_proven: True
- reject_before: {'account': {'cash': 10000.0, 'equity': 10000.0, 'realized_pnl': 0.0, 'unrealized_pnl': 0.0, 'max_equity': 10000.0, 'drawdown_pct': 0.0, 'daily_loss_pct': 0.0}, 'orders': 0, 'fills': 0, 'positions': 0}
- reject_after: {'account': {'cash': 10000.0, 'equity': 10000.0, 'realized_pnl': 0.0, 'unrealized_pnl': 0.0, 'max_equity': 10000.0, 'drawdown_pct': 0.0, 'daily_loss_pct': 0.0}, 'orders': 0, 'fills': 0, 'positions': 0}
- approve_fill_proven: True
- approve_after: {'account': {'cash': 9063.71918128, 'equity': 10008.71918128, 'realized_pnl': 0.0, 'unrealized_pnl': 8.906399999999962, 'max_equity': 10000.0, 'drawdown_pct': 0.0, 'daily_loss_pct': 0.0}, 'orders': 1, 'fills': 1, 'positions': 1, 'latest_fill': {'fill_id': '75c613e6-826b-4502-80ed-1008b7afa7b9', 'order_id': 'cb33ba55-27c8-4c76-a48f-2d0f1a9cb118', 'symbol': 'AAPL', 'side': 'buy', 'qty': 9.0, 'price': 104.0104, 'fee': 0.18721872, 'created_at': '2026-04-30T01:26:54.885642+00:00'}, 'latest_position': {'symbol': 'AAPL', 'qty': 9.0, 'avg_cost': 104.0104, 'mark_price': 105.0, 'market_value': 945.0, 'unrealized_pnl': 8.906399999999962}}
- restart_reload_proven: True
- reload_after: {'account': {'cash': 9063.71918128, 'equity': 10008.71918128, 'realized_pnl': 0.0, 'unrealized_pnl': 8.906399999999962, 'max_equity': 10000.0, 'drawdown_pct': 0.0, 'daily_loss_pct': 0.0}, 'orders': 1, 'fills': 1, 'positions': 1}

## Raw UI Metadata
- labels: ['Account', 'Allow Shorts', 'Approval Mode', 'Benchmark Status (weekly core / daily stress)', 'Benchmark Weekly Goal', 'Broker Backend', 'Fills', 'Include Pre/Post Market', 'Initial Cash ($)', 'Interval', 'Local Model Status', 'Max CVaR 95', 'Max Daily Loss %', 'Max Drawdown %', 'Max Position Fraction', 'Model Leaderboard', 'Open Orders / Orders', 'Pending Trade Idea', 'Period', 'PnL Summary', 'Positions', 'Reject reason', 'Reset paper account confirmation', 'Risk Events', 'Safety Status', 'Setup', 'Setup Status', 'Status', 'Ticker', 'Tickers (comma-separated)', 'Trading']
- buttons: ['Approve', 'Generate Suggestion', 'Refresh Dashboard', 'Reject', 'Reset Halt', 'Reset Paper Account', 'Save Configuration', 'Toggle Kill Switch']

## Prior Real Browser DOM/Click Proof Retained
Current local runtime readiness rerun was scripted only. Prior real browser proof below was not re-executed in this pass.

# UI Visual Proof

LOCAL_UI_VISUAL_PROOF: REAL_BROWSER_DOM_CLICK
REAL_BROWSER_CLICK_PROOF: YES
browser_tool_used: YES
browser_interactive_proof: YES
browser_url: http://127.0.0.1:7860/
screenshot_available: NO
browser_e2e_unavailable: NO

## Panels Visible
| panel | status | evidence |
| --- | --- | --- |
| risk disclaimer | PASS | Experimental paper-trading engine; no financial advice; trading risk; benchmarks not guarantees. |
| benchmark disclaimer | PASS | Benchmark goal can track 1% weekly paper performance, but benchmark is not guarantee. |
| pending trade idea | PASS | Pending Trade Idea table visible and populated after Generate Suggestion. |
| approve button | PASS | Approve clicked in browser. |
| reject button | PASS | Reject clicked in browser. |
| account equity/cash | PASS | Account table visible before reject, after reject, after approve, and after restart. |
| positions | PASS | Empty after reject; AAPL position visible after approve and restart. |
| orders | PASS | Empty after reject; filled order visible after approve and restart. |
| fills | PASS | Empty after reject; fill visible after approve and restart. |
| pnl summary | PASS | PnL Summary visible and updated after approve. |
| risk events | PASS | Risk Events table visible. |
| model leaderboard | PASS | Model Leaderboard panel visible. |
| benchmark status | PASS | Benchmark Status textbox visible. |
| local model status | PASS | Local Model Status textbox visible. |
| safety status | PASS | Safety Status textbox visible. |
| reset confirmation | PASS | Reset Paper Account required exact RESET PAPER confirmation. |

## Real Browser Flow Rerun
- Opened http://127.0.0.1:7860/ and clicked Trading.
- Reset setup: typed RESET PAPER, clicked Reset Paper Account, clicked Refresh Dashboard, clicked Reset Halt.
- Proof data combo: AAPL, interval 1m, period 7d, no pre/post market.
- No risk limits, leverage, model family, architecture, or exit rules changed.

### Reject flow
- Account before reject proof: cash 10000, equity 10000, realized_pnl 0, unrealized_pnl 0.
- Generate Suggestion produced pending idea: AAPL buy qty 3, confidence 1, expected_return 0.0003389495250303298, cvar_95 0.0019211889435489793.
- Clicked Reject.
- Status visible: [ACTIVE] Pending suggestion rejected. No trade submitted.
- After Refresh Dashboard: cash 10000, equity 10000, realized_pnl 0, unrealized_pnl 0.
- No visible position/order/fill row was created by the rejected suggestion.

### Approve flow
- Generated a new pending idea: AAPL buy qty 3, confidence 1, expected_return 0.0003389495250303298, cvar_95 0.0019211889435489793.
- Clicked Approve.
- Status visible: [ACTIVE] Approved and submitted paper order f5b12808-f64f-4b26-b578-abebd6c88ecc.
- Account visible: cash 9193.37302807701, equity 9999.758068360214, realized_pnl 0, unrealized_pnl -0.0806385040282862.
- Position visible: AAPL qty 3, avg_cost 268.82189292907714, mark_price 268.7950134277344, market_value 806.3850402832031.
- Order visible: order_id f5b12808-f64f-4b26-b578-abebd6c88ecc, AAPL buy qty 3, status filled.
- Fill visible: fill_id 42736fec-ed60-48e3-ab77-ccb311768d92, order_id f5b12808-f64f-4b26-b578-abebd6c88ecc, AAPL buy qty 3.
- PnL Summary visible: equity 9999.758068360214, cash 9193.37302807701, realized_pnl 0, unrealized_pnl -0.0806385040282862.

### Restart reload flow
- Killed running app process.
- Restarted app with python app.py.
- Opened a fresh browser page and clicked Trading.
- Verified persisted SQLite state: cash 9193.37302807701, equity 9999.758068360214, AAPL qty 3, filled order f5b12808-f64f-4b26-b578-abebd6c88ecc, fill 42736fec-ed60-48e3-ab77-ccb311768d92, and matching PnL Summary values visible after restart.

## Scripted Regression Proof
- scripts/local_run_readiness.py also passed reject, approve, fill, and restart reload checks with a temporary SQLite DB.
- This scripted check is regression support only; REAL_BROWSER_CLICK_PROOF above came from real browser DOM/click actions.

## Guardrails Verified
- paper_only_default: YES
- live_order_path_enabled: NO
- approval_required: YES
- official_broker_live_orders_submitted: NO
- profit_guarantee: NO
