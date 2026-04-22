from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from marketify.config import AppConfig

    default_horizon = AppConfig().broker.time_stop_bars
    parser = argparse.ArgumentParser(description="Train recurrent GRU/LSTM baselines.")
    parser.add_argument("--hold-horizon-bars", type=int, default=default_horizon)
    parser.add_argument("--label-mode", choices=["return", "direction"], default="return")
    parser.add_argument("--sequence-length", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--artifact-dir", default="artifacts")
    return parser.parse_args(argv)


def _artifact_path(artifact_dir: Path, architecture: str, ticker: str, hold_horizon_bars: int, label_mode: str) -> Path:
    return artifact_dir / f"{architecture}_{ticker}_h{hold_horizon_bars}_{label_mode}.pt"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    from marketify.config import AppConfig
    from marketify.data.market_data import fetch_market_data
    from marketify.features.technical import add_technical_features
    from marketify.models.rnn_model import (RecurrentModelConfig,
                                            build_sequence_dataset,
                                            make_recurrent_model,
                                            prepare_recurrent_training_frame)
    from scripts.run_exit_rule_experiment import (
        _ensure_optional_dependencies_for_exit_experiment,
        split_for_validation_experiment)

    try:
        _ensure_optional_dependencies_for_exit_experiment()
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", None) == "ta":
            return 1
        raise

    config = AppConfig()
    artifact_dir = _ensure_dir(ROOT / args.artifact_dir)

    print("[RECURRENT-TRAIN] Preparing market features...")
    raw = fetch_market_data(
        ticker=config.data.ticker,
        interval=config.data.interval,
        period=config.data.period,
        prepost=config.data.prepost,
    )
    feat = add_technical_features(raw)
    aligned, feature_cols, target_col = prepare_recurrent_training_frame(
        feat,
        hold_horizon_bars=args.hold_horizon_bars,
        mode=args.label_mode,
    )
    train_val, validation_index, split_meta = split_for_validation_experiment(aligned)
    train_index = train_val.index[:split_meta.train_end]

    summary: list[dict[str, object]] = []
    for architecture in ("gru", "lstm"):
        model_cfg = RecurrentModelConfig(
            architecture=architecture,
            sequence_length=args.sequence_length,
            epochs=args.epochs,
        )
        model = make_recurrent_model(architecture, model_cfg)
        x_train, y_train, _train_ts = build_sequence_dataset(
            train_val,
            feature_cols=feature_cols,
            target_col=target_col,
            sequence_length=model_cfg.sequence_length,
            selected_index=train_index,
        )
        if x_train.size == 0:
            raise ValueError(f"No train sequences available for {architecture}")

        model.fit(x_train, y_train)

        x_val, y_val, val_ts = build_sequence_dataset(
            train_val,
            feature_cols=feature_cols,
            target_col=target_col,
            sequence_length=model_cfg.sequence_length,
            selected_index=validation_index,
        )
        preds_val = model.predict(x_val)
        mae = float(mean_absolute_error(y_val, preds_val)) if len(y_val) > 0 else float("nan")
        rmse = float(np.sqrt(mean_squared_error(y_val, preds_val))) if len(y_val) > 0 else float("nan")

        artifact_path = _artifact_path(
            artifact_dir=artifact_dir,
            architecture=architecture,
            ticker=config.data.ticker,
            hold_horizon_bars=args.hold_horizon_bars,
            label_mode=args.label_mode,
        )
        model.save(
            artifact_path,
            feature_cols=feature_cols,
            target_col=target_col,
            extra_meta={
                "ticker": config.data.ticker,
                "hold_horizon_bars": args.hold_horizon_bars,
                "label_mode": args.label_mode,
                "validation_points": len(val_ts),
            },
        )
        summary.append(
            {
                "model": architecture,
                "ticker": config.data.ticker,
                "hold_horizon_bars": args.hold_horizon_bars,
                "label_mode": args.label_mode,
                "mae": round(mae, 6) if np.isfinite(mae) else None,
                "rmse": round(rmse, 6) if np.isfinite(rmse) else None,
                "artifact": str(artifact_path),
                "feature_count": len(feature_cols),
            }
        )
        print(
            f"[RECURRENT-TRAIN] {architecture.upper()} artifact={artifact_path} "
            f"mae={summary[-1]['mae']} rmse={summary[-1]['rmse']}"
        )

    summary_path = artifact_dir / f"recurrent_baseline_summary_{config.data.ticker}_h{args.hold_horizon_bars}_{args.label_mode}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[RECURRENT-TRAIN] summary={summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())