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
- weekly_return_pct: 7.7625
- weekly_benchmark: PASS
- daily_return_pct: 0.0659
- daily_stress_benchmark: FAIL
- daily_stress_exact_blocker: daily_return_pct below 1.0
- expectancy: 0.000000
- max_drawdown_pct: 6.3315
- keep_coverage_pct: 0.00
- decision: keep last retained baseline unless a later candidate passes all guardrails
- allowed_future_knobs: confidence threshold; abstain threshold; disagreement threshold; cost buffer threshold; entry threshold; min confidence
- stop_reason: daily_return_pct below 1.0; report FAIL honestly

## Safe Loop Attempts
- total_cycles_run: 3
- total_iterations_run: 6
- best_safe_daily_return_pct: 0.0659
| cycle | iteration | iteration_id | knobs | gpu_used | recurrent_skipped | weekly_benchmark | daily_stress | weekly_return_pct | daily_return_pct | sharpe | max_drawdown_pct | trade_count | expectancy | keep_or_revert | reason |
| ---: | ---: | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0 | 0 | baseline | none | YES | YES | PASS | FAIL | 7.7625 | 0.0659 | 0.5711 | 6.3315 | 0 | 0.000000 | KEEP_BASELINE | current cheap-path baseline |
| 1 | 1 | cycle_1_iter_1 | approval_precision_threshold=0.300000 | YES | YES | PASS | FAIL | 7.7625 | 0.0659 | 0.5711 | 6.3315 | 0 | 0.000000 | REVERT | expectancy<=0; trade_count=0; daily did not improve |
| 1 | 2 | cycle_1_iter_2 | cost_buffer_floor=0.000300 | YES | YES | PASS | FAIL | 7.7625 | 0.0659 | 0.5711 | 6.3315 | 0 | 0.000000 | REVERT | expectancy<=0; trade_count=0; daily did not improve |
| 2 | 1 | cycle_2_iter_1 | approval_precision_threshold=0.250000 | YES | YES | PASS | FAIL | 7.7625 | 0.0659 | 0.5711 | 6.3315 | 0 | 0.000000 | REVERT | expectancy<=0; trade_count=0; daily did not improve |
| 2 | 2 | cycle_2_iter_2 | cost_buffer_floor=0.000300 | YES | YES | PASS | FAIL | 7.7625 | 0.0659 | 0.5711 | 6.3315 | 0 | 0.000000 | REVERT | expectancy<=0; trade_count=0; daily did not improve |
| 3 | 1 | cycle_3_iter_1 | approval_precision_threshold=0.450000 | YES | YES | PASS | FAIL | 7.7625 | 0.0659 | 0.5711 | 6.3315 | 0 | 0.000000 | REVERT | expectancy<=0; trade_count=0; daily did not improve |
| 3 | 2 | cycle_3_iter_2 | cost_buffer_floor=0.000375 | YES | YES | PASS | FAIL | 7.7625 | 0.0659 | 0.6512 | 6.3315 | 0 | 0.000000 | REVERT | expectancy<=0; trade_count=0; daily did not improve |

## Decision
- WEEKLY_BENCHMARK: PASS
- DAILY_STRESS: FAIL
- total_cycles_run: 3
- total_iterations_run: 6
- best_safe_daily_return_pct: 0.0659
- exact_numeric_reason: daily_return_pct below 1.0
- no_fake_pass: YES
