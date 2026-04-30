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
- weekly_return_pct: 7.5063
- weekly_benchmark: PASS
- daily_return_pct: 0.0655
- daily_stress_benchmark: FAIL
- daily_stress_exact_blocker: daily_return_pct below 1.0
- expectancy: -8.202500
- max_drawdown_pct: 6.2752
- keep_coverage_pct: 0.28
- decision: keep baseline; daily stress tuning deferred until broker-readiness outputs are present
- allowed_future_knobs: confidence threshold; abstain threshold; disagreement threshold; cost buffer threshold
- stop_reason: daily_return_pct below 1.0; report FAIL honestly

## Safe Loop Attempts
| iteration | knobs | weekly_benchmark | daily_stress | weekly_return_pct | daily_return_pct | expectancy | max_drawdown_pct | keep_or_revert | reason |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 0 | none | PASS | FAIL | 7.5063 | 0.0655 | -8.202500 | 6.2752 | KEEP_BASELINE | current cheap-path baseline |
| 1 | approval_precision_threshold=0.300000 | PASS | FAIL | 7.5195 | 0.0656 | -8.230850 | 6.2864 | REVERT | guardrail failed or daily did not improve |
| 2 | cost_buffer_floor=0.000300 | PASS | FAIL | 7.6261 | 0.0656 | -8.252400 | 6.2976 | REVERT | guardrail failed or daily did not improve |
| 3 | abstain_margin=0.000025 | PASS | FAIL | 7.6332 | 0.0657 | -8.252400 | 6.3032 | REVERT | guardrail failed or daily did not improve |
| 4 | approval_precision_threshold=0.300000; cost_buffer_floor=0.000300 | PASS | FAIL | 7.6403 | 0.0657 | -8.252400 | 6.3088 | REVERT | guardrail failed or daily did not improve |
| 5 | approval_precision_threshold=0.400000 | PASS | FAIL | 7.6474 | 0.0658 | -8.252400 | 6.3145 | REVERT | guardrail failed or daily did not improve |

## Decision
- WEEKLY_BENCHMARK: PASS
- DAILY_STRESS: FAIL
- exact_numeric_reason: daily_return_pct below 1.0
- no_fake_pass: YES
