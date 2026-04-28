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
| local model status | PASS | Local Model Status textbox visible and updated after Refresh Dashboard. |
| safety status | PASS | Safety Status textbox visible with paper_only_default=YES and approval_required=YES. |

## Real Browser Flow Proof
- Opened http://127.0.0.1:7860/ and clicked the Trading tab.
- Initial account before trade: cash 10000, equity 10000, realized_pnl 0, unrealized_pnl 0.
- First Generate Suggestion click initially exposed a real browser-only bug: TypeError: float() argument must be a string or a real number, not 'Series'.
- Fixed the bug without architecture/model/leverage/exit-rule changes by normalizing latest numeric columns from yfinance MultiIndex/duplicate-column layouts.
- Re-ran browser proof after app restart.

### Reject flow
- Clicked Generate Suggestion.
- Pending idea visible: AAPL buy qty 3, confidence 1, expected_return 0.0003357288078404963, cvar_95 0.004063294591088058.
- Clicked Reject.
- Status visible: [ACTIVE] Pending suggestion rejected. No trade submitted.
- Account unchanged: cash 10000, equity 10000, realized_pnl 0, unrealized_pnl 0.
- Pending Trade Idea table returned to headers only; no order/fill/position was created.

### Approve flow
- Clicked Generate Suggestion again.
- Clicked Approve.
- Status visible: [ACTIVE] Approved and submitted paper order 1e7dc72a-a652-4a33-85d1-877dbdf4bf9e.
- Account visible: cash 9185.300691169397, equity 9999.755647224085, unrealized_pnl -0.08144549560546466.
- Position visible: AAPL qty 3, avg_cost 271.51213385009765, mark_price 271.4849853515625, market_value 814.4549560546875.
- Order visible: order_id 1e7dc72a-a652-4a33-85d1-877dbdf4bf9e, symbol AAPL, side buy, qty 3, status filled.
- Clicked Refresh Dashboard.
- Fill visible: fill_id fb1286b4-9df5-43f2-88b2-1a5d72460f68, order_id 1e7dc72a-a652-4a33-85d1-877dbdf4bf9e, symbol AAPL, side buy, qty 3.
- PnL Summary visible: equity 9999.755647224085, cash 9185.300691169397, realized_pnl 0, unrealized_pnl -0.08144549560546466.
- Local Model Status visible after refresh: LOCAL_MODEL_LOAD=YES; loaded_model_type=xgb; inference_smoke=PASS.

### Restart reload flow
- Killed the running app terminal.
- Restarted the app with python app.py.
- Reloaded the same browser page and clicked Trading.
- Verified persisted state returned from SQLite: cash 9185.300691169397, equity 9999.755647224085, AAPL qty 3, filled order 1e7dc72a-a652-4a33-85d1-877dbdf4bf9e, fill fb1286b4-9df5-43f2-88b2-1a5d72460f68, and PnL Summary values still visible.

## Guardrails Verified
- paper_only_default: YES
- live_order_path_enabled: NO
- approval_required: YES
- official_broker_live_orders_submitted: NO
- profit_guarantee: NO
