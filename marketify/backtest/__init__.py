from marketify.backtest.benchmark import BenchmarkResult, compute_benchmark


def run_walk_forward_backtest(*args, **kwargs):
	from marketify.backtest.simulator import \
	    run_walk_forward_backtest as _run_walk_forward_backtest

	return _run_walk_forward_backtest(*args, **kwargs)

__all__ = ["run_walk_forward_backtest", "BenchmarkResult", "compute_benchmark"]
