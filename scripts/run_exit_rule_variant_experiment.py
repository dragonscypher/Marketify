"""Run real backtest exit-rule variants and compare on TEST split only.

Outputs:
  reports/exit_rule_experiment.csv
  reports/exit_rule_experiment.md
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

REPORTS = ROOT / "reports"
TEST_SPLIT_FRACTION = 0.15
TARGET_WEEKLY_PCT = 1.0


@dataclass
class VariantResult:
    variant: str
    split: str
    weekly_return_pct: float
    daily_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    cvar_95_pct: float
    trade_count: int
    status: str
    reason: str


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _test_start_index(n_rows: int, test_fraction: float = TEST_SPLIT_FRACTION) -> int:
    if n_rows <= 2:
        return 0
    split = int(round(n_rows * (1.0 - test_fraction)))
    split = max(0, min(split, n_rows - 2))
    return split


def _history_from_equity(equity: pd.Series) -> list[dict[str, str | float]]:
    clean = pd.to_numeric(equity, errors="coerce").dropna()
    if len(clean) < 2:
        raise ValueError("Need at least 2 equity points for benchmark computation.")
    return [{"ts": str(ts), "equity": float(val)} for ts, val in clean.items()]


def _select_best(df: pd.DataFrame) -> pd.Series | None:
    if df.empty:
        return None
    ranked = df.sort_values(
        ["weekly_return_pct", "sharpe_ratio", "max_drawdown_pct", "trade_count"],
        ascending=[False, False, True, False],
    )
    return ranked.iloc[0]


def _variant_overrides(config, variant: str) -> None:
    config.broker.exit_rule_variant = variant

    # Preserve existing safety envelope. No leverage increase.
    config.broker.max_position_fraction = min(float(config.broker.max_position_fraction), 0.10)

    if variant == "fixed":
        config.broker.exit_trailing_enabled = False
        config.broker.exit_time_stop_mode = "fixed"
        return
    if variant == "atr":
        config.broker.exit_k_stop = 1.5
        config.broker.exit_k_take = 2.2
        config.broker.exit_trailing_enabled = False
        config.broker.exit_time_stop_mode = "fixed"
        return
    if variant == "trailing":
        config.broker.exit_k_stop = 1.5
        config.broker.exit_k_take = 2.2
        config.broker.exit_trailing_enabled = True
        config.broker.exit_trailing_activate_profit_pct = 0.004
        config.broker.exit_trailing_distance_pct = 0.003
        config.broker.exit_time_stop_mode = "fixed"
        return
    if variant == "hybrid":
        config.broker.exit_k_stop = 1.5
        config.broker.exit_k_take = 2.5
        config.broker.exit_trailing_enabled = True
        config.broker.exit_trailing_activate_profit_pct = 0.004
        config.broker.exit_trailing_distance_pct = 0.003
        config.broker.exit_time_stop_mode = "volatility_adjusted"
        return

    raise ValueError(f"Unknown exit-rule variant: {variant}")


def _run_variant(variant: str) -> VariantResult:
    from marketify.backtest.benchmark import compute_benchmark
    from marketify.backtest.simulator import run_walk_forward_backtest
    from marketify.config import AppConfig

    config = AppConfig()
    _variant_overrides(config, variant)

    result = run_walk_forward_backtest(config)
    equity = result["equity"]
    diagnostics = result.get("trade_diagnostics", pd.DataFrame())

    split_start = _test_start_index(len(equity), TEST_SPLIT_FRACTION)
    test_equity = equity.iloc[split_start:]
    test_start_ts = test_equity.index[0] if len(test_equity) else None

    test_diag = diagnostics
    if not diagnostics.empty and test_start_ts is not None and "timestamp" in diagnostics.columns:
        ts = pd.to_datetime(diagnostics["timestamp"], errors="coerce")
        test_diag = diagnostics[ts >= test_start_ts]

    history = _history_from_equity(test_equity if len(test_equity) >= 2 else equity)
    bm = compute_benchmark(history, weekly_goal=TARGET_WEEKLY_PCT / 100.0)

    weekly_pct = round(bm.weekly_return * 100, 4)
    status = "PASS" if bm.weekly_pass else "FAIL"
    reason = (
        "meets 1.0% weekly target on TEST split"
        if status == "PASS"
        else f"weekly return {weekly_pct}% < {TARGET_WEEKLY_PCT}% target on TEST split"
    )

    return VariantResult(
        variant=variant,
        split="test_only",
        weekly_return_pct=weekly_pct,
        daily_return_pct=round(bm.daily_return * 100, 4),
        max_drawdown_pct=round(bm.max_drawdown * 100, 4),
        sharpe_ratio=round(bm.sharpe_ratio, 4),
        cvar_95_pct=round(bm.cvar_95 * 100, 4),
        trade_count=int(len(test_diag)) if not test_diag.empty else 0,
        status=status,
        reason=reason,
    )


def _write_markdown(results_df: pd.DataFrame, best_row: pd.Series | None, out_path: Path) -> None:
    lines = [
        "# Exit Rule Experiment (TEST Split)",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "Only TEST split used for ranking. Validation results not used for winner selection.",
        "",
        "| Variant | Split | Weekly % | Daily % | Max DD % | Sharpe | CVaR 5% | Trades | Status |",
        "|---------|-------|----------|---------|----------|--------|---------|--------|--------|",
    ]

    for row in results_df.to_dict(orient="records"):
        lines.append(
            f"| {row['variant']} | {row['split']} | {row['weekly_return_pct']} | {row['daily_return_pct']} | "
            f"{row['max_drawdown_pct']} | {row['sharpe_ratio']} | {row['cvar_95_pct']} | {row['trade_count']} | {row['status']} |"
        )

    lines.append("")
    if best_row is None:
        lines.append("No result produced.")
    else:
        target_status = "PASS" if float(best_row["weekly_return_pct"]) >= TARGET_WEEKLY_PCT else "FAIL"
        lines.append("## Best Variant (TEST split)")
        lines.append(f"- Variant: {best_row['variant']}")
        lines.append(f"- Weekly return: {best_row['weekly_return_pct']}%")
        lines.append(f"- Daily return: {best_row['daily_return_pct']}%")
        lines.append(f"- Max drawdown: {best_row['max_drawdown_pct']}%")
        lines.append(f"- Sharpe: {best_row['sharpe_ratio']}")
        lines.append(f"- CVaR 5%: {best_row['cvar_95_pct']}%")
        lines.append(f"- Trade count: {int(best_row['trade_count'])}")
        lines.append(f"- 1% weekly target: {target_status}")
        lines.append(f"- Reason: {best_row['reason']}")

    lines.extend(
        [
            "",
            "## Safety",
            "- Paper mode only",
            "- No leverage increase (max_position_fraction capped <= 0.10)",
            "- No lookahead in volatility reference",
        ]
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    _ensure_dir(REPORTS)
    variants = ["fixed", "atr", "trailing", "hybrid"]

    rows: list[VariantResult] = []
    for variant in variants:
        print(f"[EXIT-VARIANT] running {variant} ...")
        rows.append(_run_variant(variant))

    results_df = pd.DataFrame([asdict(row) for row in rows])
    csv_path = REPORTS / "exit_rule_experiment.csv"
    md_path = REPORTS / "exit_rule_experiment.md"
    results_df.to_csv(csv_path, index=False)

    best_row = _select_best(results_df)
    _write_markdown(results_df, best_row, md_path)

    print(json.dumps({"csv": str(csv_path), "md": str(md_path)}, indent=2))
    if best_row is None:
        print("No best variant. FAIL")
        return 1

    print(
        f"Best variant={best_row['variant']} weekly={best_row['weekly_return_pct']}% "
        f"sharpe={best_row['sharpe_ratio']} trades={int(best_row['trade_count'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
