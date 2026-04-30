# Daily Stress Iteration Log

Secondary benchmark only. No architecture change. No leverage change. No exit-rule work.

## Runtime Policy
- LOCAL_KERNEL_FIXED: YES
- LOCAL_GPU_AVAILABLE: NO
- LOCAL_GPU_USED: NO
- LOCAL_GPU_NAME: NO_GPU
- LOCAL_LOW_RAM_MODE: YES
- RECURRENT_BRANCH_SKIPPED: YES
- skip_reason: LOCAL_LOW_RAM_MODE=1 cheap path skips recurrent/fusion heavy branches

## Iteration 0 - current safe baseline
- iteration_number: 0
- knob_changes: none
- weekly_return_pct: 1.3006
- weekly_benchmark: PASS
- daily_return_pct: -0.0357
- daily_stress_benchmark: FAIL
- daily_stress_exact_blocker: daily_return_pct below 1.0
- expectancy: 2.664591
- max_drawdown_pct: 0.8666
- keep_coverage_pct: 4.51
- decision: keep baseline; daily stress tuning deferred until broker-readiness outputs are present
- allowed_future_knobs: confidence threshold; abstain threshold; disagreement threshold; cost buffer threshold
- stop_reason: daily_return_pct below 1.0; report FAIL honestly

## Safe Loop Attempts
| iteration | knobs | weekly_benchmark | daily_stress | weekly_return_pct | daily_return_pct | expectancy | max_drawdown_pct | keep_or_revert | reason |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 0 | none | PASS | FAIL | 1.3006 | -0.0357 | 2.664591 | 0.8666 | KEEP_BASELINE | current cheap-path baseline |
| 1 | approval_precision_threshold=0.300000 | PASS | FAIL | 1.8526 | -0.0531 | 2.664591 | 1.3336 | REVERT | guardrail failed or daily did not improve |
| 2 | cost_buffer_floor=0.000300 | PASS | FAIL | 1.6749 | -0.0708 | 0.372121 | 1.4991 | REVERT | guardrail failed or daily did not improve |
| 3 | abstain_margin=0.000025 | PASS | FAIL | 2.4796 | -0.0881 | 1.982845 | 1.9679 | REVERT | guardrail failed or daily did not improve |
| 4 | approval_precision_threshold=0.300000; cost_buffer_floor=0.000300 | PASS | FAIL | 2.9437 | -0.1056 | 1.303700 | 2.3577 | REVERT | guardrail failed or daily did not improve |
| 5 | approval_precision_threshold=0.400000 | PASS | FAIL | 3.5595 | -0.1229 | 1.804883 | 2.8317 | REVERT | guardrail failed or daily did not improve |

## Decision
- WEEKLY_BENCHMARK: PASS
- DAILY_STRESS: FAIL
- exact_numeric_reason: daily_return_pct below 1.0
- no_fake_pass: YES
