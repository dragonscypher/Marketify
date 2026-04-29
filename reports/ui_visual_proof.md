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
| risk disclaimer | PASS | DOM text: Experimental paper-trading engine. No financial advice. Trading involves risk, including loss of principal. Target metrics are benchmarks, not guarantees. |
| benchmark disclaimer | PASS | DOM text: Benchmark goal can track 1% weekly paper performance, but benchmark is not guarantee. |
| pending trade idea | PASS | Pending Trade Idea dataframe visible and populated after Generate Suggestion. |
| approve button | PASS | Approve button clicked in browser. |
| reject button | PASS | Reject button clicked in browser. |
| account equity/cash | PASS | Account dataframe visible before and after approval. |
| positions | PASS | AAPL position visible after approval and after restart. |
| open orders | PASS | Filled order visible after approval and after restart. |
| fills | PASS | Fill visible after Refresh Dashboard and after restart. |
| realized/unrealized pnl | PASS | PnL Summary visible; unrealized_pnl updated after fill. |
| risk events | PASS | Risk Events dataframe visible. |
| model leaderboard | PASS | Model Leaderboard panel visible. |
| weekly benchmark PASS/FAIL | PASS | Benchmark Status textbox visible. |
| daily stress PASS/FAIL | PASS | Benchmark Status textbox visible. |
| kill switch | PASS | Toggle Kill Switch button visible. |
| reset paper account with confirmation | PASS | Reset confirmation textbox + Reset Paper Account button visible. |
| local model status | PASS | Local Model Status textbox visible. |
| safety status | PASS | Safety Status textbox visible with paper_only_default=YES and approval_required=YES. |

## Real Browser Flow Proof
- Opened http://127.0.0.1:7860/ and clicked the Trading tab.
- Initial account before trade: cash 10000, equity 10000, realized_pnl 0, unrealized_pnl 0.
- Default 5m/60d Generate Suggestion safely produced risk rejection expected_return_below_threshold and no order.
- Selected AAPL interval 1m and period 7d for proof. No risk limits, leverage, model family, architecture, or exit-rule settings changed.
- Generate Suggestion produced pending idea: AAPL buy qty 3, confidence 1, expected_return 0.0024088823702186346, cvar_95 0.0006221716875696939.

### Reject flow
- Clicked Reject.
- Status visible: [ACTIVE] Pending suggestion rejected. No trade submitted.
- Account unchanged: cash 10000, equity 10000, realized_pnl 0, unrealized_pnl 0.
- No order/fill/position was created by the rejected suggestion.

### Approve flow
- Clicked Generate Suggestion again.
- Clicked Approve.
- Status visible: [ACTIVE] Approved and submitted paper order f425332b-ec1b-4131-a120-f4b611c72df6.
- Account visible: cash 9187.236238739224, equity 9999.756227752896, unrealized_pnl -0.08125199890127988.
- Position visible: AAPL qty 3, avg_cost 270.8670803375244, mark_price 270.8399963378906, market_value 812.5199890136719.
- Order visible: order_id f425332b-ec1b-4131-a120-f4b611c72df6, symbol AAPL, side buy, qty 3, status filled.
- Clicked Refresh Dashboard.
- Fill visible: fill_id aa37f2af-daa5-459b-a035-c649a888d545, order_id f425332b-ec1b-4131-a120-f4b611c72df6, symbol AAPL, side buy, qty 3.
- PnL Summary visible: equity 9999.756227752896, cash 9187.236238739224, realized_pnl 0, unrealized_pnl -0.08125199890127988.

### Restart reload flow
- Killed the running app terminal.
- Restarted the app with python app.py.
- Opened a fresh browser page and clicked Trading.
- Verified persisted state returned from SQLite: cash 9187.236238739224, equity 9999.756227752896, AAPL qty 3, filled order f425332b-ec1b-4131-a120-f4b611c72df6, fill aa37f2af-daa5-459b-a035-c649a888d545, and PnL Summary values still visible.

## Guardrails Verified
- paper_only_default: YES
- live_order_path_enabled: NO
- approval_required: YES
- official_broker_live_orders_submitted: NO
- profit_guarantee: NO
