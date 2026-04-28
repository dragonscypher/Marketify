# Broker Validation

broker_readiness_result: PASS_WITH_SKIPS
MOCK_BROKER_PROVEN: YES
ALPACA_PAPER_PROVEN: SKIP
IBKR_READ_ONLY_PROVEN: SKIP
real_broker_proven: NO if official targets are SKIP

| tier | target | status | reason |
| --- | --- | --- | --- |
| A | local_paper_internal | PASS | SQLite PaperBroker accounting invariant |
| B | local_mock_http | PASS | deterministic mock broker HTTP contract and accounting invariant |
| C | alpaca_paper_read_only | SKIP | Alpaca paper env unavailable |
| C | ibkr_paper_read_only | SKIP | IBKR env unavailable |

Guardrails: paper only by default; no live endpoint; no order submission to official brokers unless explicit paper path is wired and guarded.
