# Daily Stress Iteration Log

Secondary benchmark only. No architecture change. No leverage change. No exit-rule work.

## Runtime Policy
- LOCAL_KERNEL_FIXED: YES
- LOCAL_GPU_AVAILABLE: YES
- LOCAL_GPU_USED: YES
- LOCAL_GPU_NAME: NVIDIA GeForce RTX 3060 Laptop GPU
- LOCAL_LOW_RAM_MODE: YES
- RECURRENT_BRANCH_SKIPPED: YES
- skip_reason: LOCAL_LOW_RAM_MODE=1 cheap path skips recurrent/fusion heavy branches

## Iteration 0 - current safe baseline
- iteration_number: 0
- knob_changes: none
- weekly_return_pct: 7.3841
- weekly_benchmark: PASS
- daily_return_pct: 0.0647
- daily_stress_benchmark: FAIL
- daily_stress_exact_blocker: daily_return_pct below 1.0
- expectancy: -2.374443
- max_drawdown_pct: 6.1996
- keep_coverage_pct: 0.97
- decision: keep baseline; daily stress tuning deferred until broker-readiness outputs are present
- allowed_future_knobs: confidence threshold; abstain threshold; disagreement threshold; cost buffer threshold
- stop_reason: daily_return_pct below 1.0; report FAIL honestly

## Safe Loop Attempts
| iteration | knobs | weekly_benchmark | daily_stress | weekly_return_pct | daily_return_pct | expectancy | max_drawdown_pct | keep_or_revert | reason |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 0 | none | PASS | FAIL | 7.3841 | 0.0647 | -2.374443 | 6.1996 | KEEP_BASELINE | current cheap-path baseline |
| 1 | approval_precision_threshold=0.300000 | PASS | FAIL | 7.3687 | 0.0649 | -4.109880 | 6.2121 | REVERT | guardrail failed or daily did not improve |
| 2 | cost_buffer_floor=0.000300 | PASS | FAIL | 7.3867 | 0.0650 | -4.105600 | 6.2266 | REVERT | guardrail failed or daily did not improve |
| 3 | abstain_margin=0.000025 | PASS | FAIL | 7.4446 | 0.0651 | -4.306625 | 6.2410 | REVERT | guardrail failed or daily did not improve |
| 4 | approval_precision_threshold=0.300000; cost_buffer_floor=0.000300 | PASS | FAIL | 7.4792 | 0.0652 | -8.182300 | 6.2532 | REVERT | guardrail failed or daily did not improve |
| 5 | approval_precision_threshold=0.400000 | PASS | FAIL | 7.4925 | 0.0654 | -8.200250 | 6.2642 | REVERT | guardrail failed or daily did not improve |

## Decision
- WEEKLY_BENCHMARK: PASS
- DAILY_STRESS: FAIL
- exact_numeric_reason: daily_return_pct below 1.0
- no_fake_pass: YES
