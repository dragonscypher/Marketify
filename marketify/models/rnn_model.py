from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

try:
    from marketify.features.technical import \
        TECHNICAL_FEATURE_COLUMNS as _TECHNICAL_FEATURE_COLUMNS
except ModuleNotFoundError as exc:
    if getattr(exc, "name", None) != "ta":
        raise
    _TECHNICAL_FEATURE_COLUMNS = [
        "ret_1",
        "ret_5",
        "ret_15",
        "ret_30",
        "ema_dist_10",
        "ema_dist_20",
        "ema_dist_50",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "atr_14",
        "atr_pct",
        "vol_20",
        "intrabar_range_pct",
        "close_location",
        "volume_z20",
        "dollar_volume_z20",
        "volume_regime_ratio",
        "vol_regime_ratio",
        "volatility_regime_code",
        "breakout_up_20",
        "breakout_down_20",
        "ret_vs_spy_5",
        "ret_vs_spy_20",
        "ret_vs_sector_5",
        "ret_vs_sector_20",
        "signal_quality_20",
        "signal_stability_20",
        "trend_alignment_score",
        "news_risk",
        "news_sentiment_score",
        "news_headline_count",
        "news_event_shock",
        "vix_proxy",
        "macro_event_risk",
        "macro_event_window",
        "macro_policy_bias",
        "macro_trend_signal",
        "macro_regime_bull",
        "macro_regime_bear",
        "macro_regime_neutral",
        "macro_vol_level_high",
        "sin_hour",
        "cos_hour",
        "sin_min",
        "cos_min",
        "is_morning",
        "is_afternoon",
        "is_late",
    ]

TECHNICAL_FEATURE_COLUMNS = list(_TECHNICAL_FEATURE_COLUMNS)

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
VOLATILITY_REGIME_LABEL_COLUMN = "volatility_regime"
VOLATILITY_REGIME_CODE_COLUMN = "volatility_regime_code"

RecurrentLabelMode = Literal["return", "direction"]
RecurrentArchitecture = Literal["gru", "lstm"]


@dataclass(frozen=True)
class RecurrentModelConfig:
    architecture: RecurrentArchitecture = "gru"
    sequence_length: int = 48
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.1
    batch_size: int = 64
    epochs: int = 6
    learning_rate: float = 1e-3
    random_state: int = 42


@dataclass(frozen=True)
class SequenceStandardizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, sequences: np.ndarray) -> np.ndarray:
        if sequences.size == 0:
            return sequences.astype(np.float32)
        return ((sequences - self.mean) / self.std).astype(np.float32)


def _require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", None) == "torch":
            raise RuntimeError(
                "Recurrent baseline requires torch. Run python -m pip install -r requirements-colab.txt"
            ) from exc
        raise
    return torch, nn, DataLoader, TensorDataset


def _as_1d_series(frame: pd.DataFrame, col: str) -> pd.Series:
    import pandas as pd

    if isinstance(frame.columns, pd.MultiIndex):
        candidates = []
        for candidate in frame.columns:
            if isinstance(candidate, tuple) and (candidate[0] == col or candidate[-1] == col):
                candidates.append(candidate)
        if not candidates:
            raise KeyError(f"Missing {col}. Columns={list(frame.columns)[:10]}")
        data = frame.loc[:, candidates[0]]
    else:
        data = frame.loc[:, col]

    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]

    data = data.squeeze()

    if not isinstance(data, pd.Series):
        data = pd.Series(data, index=frame.index)

    data = pd.to_numeric(data, errors="coerce")
    data.index = frame.index
    data.name = col

    if getattr(data, "ndim", 1) != 1:
        raise ValueError(f"{col} not 1D after normalization. shape={getattr(data, 'shape', None)}")

    return data


def _flat_series(frame: pd.DataFrame, col: str) -> pd.Series:
    return _as_1d_series(frame, col)


def _flatten_recurrent_frame(frame: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd

    out = pd.DataFrame(index=frame.index)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        out[col] = _flat_series(frame, col)

    for col in frame.columns:
        if isinstance(col, str) and col not in out.columns:
            data = frame[col]
            if isinstance(data, pd.DataFrame):
                data = data.iloc[:, 0]
            data = data.squeeze()
            if isinstance(data, pd.Series):
                out[col] = pd.to_numeric(data, errors="coerce")

    for col in TECHNICAL_FEATURE_COLUMNS:
        if col in out.columns:
            continue
        try:
            out[col] = _flat_series(frame, col)
        except KeyError:
            continue

    return out


def add_volatility_regime_feature(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    vol = pd.to_numeric(out["vol_20"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    non_null = vol.dropna()
    if non_null.empty:
        low_cutoff = 0.0
        high_cutoff = 0.0
    else:
        low_cutoff = float(non_null.quantile(0.33))
        high_cutoff = float(non_null.quantile(0.67))

    def _label(one: float) -> str:
        if not np.isfinite(one):
            return "unknown"
        if one <= low_cutoff:
            return "low"
        if one <= max(low_cutoff, high_cutoff):
            return "medium"
        return "high"

    out[VOLATILITY_REGIME_LABEL_COLUMN] = vol.map(_label)
    out[VOLATILITY_REGIME_CODE_COLUMN] = out[VOLATILITY_REGIME_LABEL_COLUMN].map(
        {"unknown": -1.0, "low": 0.0, "medium": 1.0, "high": 2.0}
    ).astype(float)
    return out


def build_aligned_labels(
    frame: pd.DataFrame,
    hold_horizon_bars: int,
    mode: RecurrentLabelMode = "return",
    price_col: str = "Close",
) -> pd.Series:
    if hold_horizon_bars < 1:
        raise ValueError("hold_horizon_bars must be >= 1")
    if mode not in {"return", "direction"}:
        raise ValueError("mode must be 'return' or 'direction'")

    price = _as_1d_series(frame, price_col)
    future_return = price.shift(-hold_horizon_bars) / price - 1.0
    column_name = f"target_h{hold_horizon_bars}_{mode}"
    if mode == "direction":
        labels = (future_return > 0).astype(float)
    else:
        labels = future_return.astype(float)
    labels.name = column_name
    return labels


def build_recurrent_feature_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = _flatten_recurrent_frame(frame)
    out = add_volatility_regime_feature(out)
    feature_cols: list[str] = []
    for col in [*OHLCV_COLUMNS, *TECHNICAL_FEATURE_COLUMNS, VOLATILITY_REGIME_CODE_COLUMN]:
        if col in out.columns and col not in feature_cols:
            feature_cols.append(col)
    out[feature_cols] = out[feature_cols].apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out, feature_cols


def prepare_recurrent_training_frame(
    frame: pd.DataFrame,
    hold_horizon_bars: int,
    mode: RecurrentLabelMode = "return",
) -> tuple[pd.DataFrame, list[str], str]:
    out = _flatten_recurrent_frame(frame)
    out = add_volatility_regime_feature(out)
    feature_cols: list[str] = []
    for col in [*OHLCV_COLUMNS, *TECHNICAL_FEATURE_COLUMNS, VOLATILITY_REGIME_CODE_COLUMN]:
        if col in out.columns and col not in feature_cols:
            feature_cols.append(col)
    if feature_cols:
        out[feature_cols] = out[feature_cols].apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)

    target = build_aligned_labels(out, hold_horizon_bars=hold_horizon_bars, mode=mode)
    out[target.name] = target
    required = [*feature_cols, target.name]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise KeyError(f"Missing recurrent columns after flatten: {missing}. columns={list(out.columns)[:80]}")
    out = out.dropna(subset=required).copy()
    return out, feature_cols, target.name


def fit_sequence_standardizer(sequences: np.ndarray) -> SequenceStandardizer:
    if sequences.ndim != 3:
        raise ValueError(f"Expected 3-D sequences, got shape {sequences.shape}")
    mean = sequences.mean(axis=(0, 1), keepdims=True)
    std = sequences.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return SequenceStandardizer(mean=mean.astype(np.float32), std=std.astype(np.float32))


def build_sequence_dataset(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    sequence_length: int,
    selected_index: pd.Index | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1")

    feature_values = frame[feature_cols].to_numpy(dtype=np.float32)
    target_values = frame[target_col].to_numpy(dtype=np.float32)
    timestamps = frame.index
    wanted = set(selected_index) if selected_index is not None else None

    xs: list[np.ndarray] = []
    ys: list[float] = []
    idx: list[pd.Timestamp] = []

    for pos in range(sequence_length - 1, len(frame)):
        ts = timestamps[pos]
        if wanted is not None and ts not in wanted:
            continue
        seq = feature_values[pos - sequence_length + 1: pos + 1]
        target = float(target_values[pos])
        if seq.shape[0] != sequence_length:
            continue
        if not np.isfinite(seq).all() or not np.isfinite(target):
            continue
        xs.append(seq)
        ys.append(target)
        idx.append(ts)

    if not xs:
        return (
            np.empty((0, sequence_length, len(feature_cols)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            pd.Index([], dtype=object),
        )
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32), pd.Index(idx)


class _RecurrentHead:
    def __init__(self, architecture: RecurrentArchitecture, config: RecurrentModelConfig):
        if architecture not in {"gru", "lstm"}:
            raise ValueError("architecture must be 'gru' or 'lstm'")
        self.architecture = architecture
        self.config = config
        self.standardizer: SequenceStandardizer | None = None
        self.model_state: dict[str, Any] | None = None
        self.input_size: int | None = None

    def _make_network(self, input_size: int):
        torch, nn, _data_loader, _tensor_dataset = _require_torch()

        rnn_cls = nn.GRU if self.architecture == "gru" else nn.LSTM

        class _Net(nn.Module):
            def __init__(self, cfg: RecurrentModelConfig, inputs: int):
                super().__init__()
                self.rnn = rnn_cls(
                    input_size=inputs,
                    hidden_size=cfg.hidden_size,
                    num_layers=cfg.num_layers,
                    dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
                    batch_first=True,
                )
                self.head = nn.Linear(cfg.hidden_size, 1)

            def forward(self, x):
                output, _state = self.rnn(x)
                last_hidden = output[:, -1, :]
                return self.head(last_hidden).squeeze(-1)

        return _Net(self.config, input_size)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        torch, nn, DataLoader, TensorDataset = _require_torch()
        if x_train.size == 0:
            raise ValueError("x_train empty")

        torch.manual_seed(self.config.random_state)
        self.standardizer = fit_sequence_standardizer(x_train)
        x_scaled = self.standardizer.transform(x_train)
        self.input_size = int(x_scaled.shape[-1])

        dataset = TensorDataset(
            torch.tensor(x_scaled, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=min(self.config.batch_size, len(dataset)), shuffle=True)

        network = self._make_network(self.input_size)
        network.train()
        optimizer = torch.optim.Adam(network.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.MSELoss()

        for _epoch in range(self.config.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                preds = network(batch_x)
                loss = loss_fn(preds, batch_y)
                loss.backward()
                optimizer.step()

        self.model_state = {key: value.detach().cpu() for key, value in network.state_dict().items()}

    def predict(self, x_pred: np.ndarray) -> np.ndarray:
        torch, _nn, _DataLoader, _TensorDataset = _require_torch()
        if self.model_state is None or self.standardizer is None or self.input_size is None:
            raise RuntimeError("Model not fit")
        if x_pred.size == 0:
            return np.empty((0,), dtype=np.float32)

        network = self._make_network(self.input_size)
        network.load_state_dict(self.model_state)
        network.eval()
        x_scaled = self.standardizer.transform(x_pred)
        with torch.no_grad():
            preds = network(torch.tensor(x_scaled, dtype=torch.float32)).cpu().numpy()
        return preds.astype(np.float32)

    def save(self, artifact_path: Path, feature_cols: list[str], target_col: str, extra_meta: dict[str, Any] | None = None) -> Path:
        torch, _nn, _DataLoader, _TensorDataset = _require_torch()
        if self.model_state is None or self.standardizer is None or self.input_size is None:
            raise RuntimeError("Model not fit")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "architecture": self.architecture,
            "config": asdict(self.config),
            "feature_cols": feature_cols,
            "target_col": target_col,
            "input_size": self.input_size,
            "standardizer_mean": self.standardizer.mean,
            "standardizer_std": self.standardizer.std,
            "state_dict": self.model_state,
            "extra_meta": extra_meta or {},
        }
        torch.save(payload, artifact_path)
        return artifact_path


class GRUBaseline(_RecurrentHead):
    def __init__(self, config: RecurrentModelConfig | None = None):
        super().__init__("gru", config or RecurrentModelConfig(architecture="gru"))


class LSTMBaseline(_RecurrentHead):
    def __init__(self, config: RecurrentModelConfig | None = None):
        super().__init__("lstm", config or RecurrentModelConfig(architecture="lstm"))


def make_recurrent_model(
    architecture: RecurrentArchitecture,
    config: RecurrentModelConfig | None = None,
) -> _RecurrentHead:
    if architecture == "gru":
        return GRUBaseline(config or RecurrentModelConfig(architecture="gru"))
    if architecture == "lstm":
        return LSTMBaseline(config or RecurrentModelConfig(architecture="lstm"))
    raise ValueError("architecture must be 'gru' or 'lstm'")


def load_recurrent_artifact(artifact_path: Path) -> tuple[_RecurrentHead, dict[str, Any]]:
    torch, _nn, _DataLoader, _TensorDataset = _require_torch()
    try:
        payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(artifact_path, map_location="cpu")
    architecture = str(payload["architecture"])
    config = RecurrentModelConfig(**payload["config"])
    model = make_recurrent_model(architecture, config)
    model.input_size = int(payload["input_size"])
    model.model_state = payload["state_dict"]
    model.standardizer = SequenceStandardizer(
        mean=np.asarray(payload["standardizer_mean"], dtype=np.float32),
        std=np.asarray(payload["standardizer_std"], dtype=np.float32),
    )
    return model, payload


def fit_predict_recurrent_split(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_index: pd.Index,
    predict_index: pd.Index,
    architecture: RecurrentArchitecture,
    config: RecurrentModelConfig | None = None,
) -> pd.Series:
    model = make_recurrent_model(architecture, config)
    cfg = model.config
    x_train, y_train, _train_ts = build_sequence_dataset(
        frame,
        feature_cols=feature_cols,
        target_col=target_col,
        sequence_length=cfg.sequence_length,
        selected_index=train_index,
    )
    if x_train.size == 0:
        raise ValueError("No train sequences available for recurrent model")
    model.fit(x_train, y_train)

    x_pred, _y_pred, pred_ts = build_sequence_dataset(
        frame,
        feature_cols=feature_cols,
        target_col=target_col,
        sequence_length=cfg.sequence_length,
        selected_index=predict_index,
    )
    preds = model.predict(x_pred)
    return pd.Series(preds, index=pred_ts, name=f"pred_{architecture}")


# ---------------------------------------------------------------------------
# Rolling-window predictor — same interface as xgb_model / ridge_model
# ---------------------------------------------------------------------------

def rolling_train_predict(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    config: "ModelConfig",  # marketify.config.ModelConfig
) -> pd.Series:
    """Walk-forward GRU prediction with same interface as XGB / Ridge.

    Converts ``ModelConfig`` RNN fields into a ``RecurrentModelConfig`` and
    drives rolling retraining using ``build_sequence_dataset``.

    Falls back to zeros-series when PyTorch is not installed.
    """
    from marketify.config import \
        ModelConfig  # local import avoids circular at module level

    if not isinstance(config, ModelConfig):
        raise TypeError(f"Expected ModelConfig, got {type(config)}")

    seq_len = getattr(config, "rnn_seq_len", 20)
    hidden = getattr(config, "rnn_hidden", 32)
    epochs = getattr(config, "rnn_epochs", 10)

    rnn_cfg = RecurrentModelConfig(
        architecture="gru",
        sequence_length=seq_len,
        hidden_size=hidden,
        num_layers=1,
        dropout=0.0,
        epochs=epochs,
        learning_rate=1e-3,
        random_state=config.random_state,
    )

    min_rows = config.train_window + seq_len + 10
    if len(frame) <= min_rows:
        raise ValueError(
            f"Not enough rows for GRU rolling training. "
            f"Need > {min_rows}, got {len(frame)}."
        )

    preds = np.full(len(frame), np.nan, dtype=float)
    idx_map = {ts: i for i, ts in enumerate(frame.index)}

    start = config.train_window
    while start < len(frame):
        train_slice = frame.iloc[max(0, start - config.train_window) : start]
        end_pred = min(len(frame), start + config.retrain_every)
        pred_slice = frame.iloc[start:end_pred]

        try:
            x_train, y_train, _ = build_sequence_dataset(
                train_slice, feature_cols, target_col, seq_len
            )
        except Exception:
            start = end_pred
            continue

        if x_train.size == 0:
            start = end_pred
            continue

        model = GRUBaseline(rnn_cfg)
        try:
            model.fit(x_train, y_train)
        except Exception:
            start = end_pred
            continue

        x_pred, _, pred_ts = build_sequence_dataset(
            # Pass larger window for sequences that straddle the boundary
            pd.concat([train_slice.iloc[-seq_len:], pred_slice]),
            feature_cols,
            target_col,
            seq_len,
            selected_index=pred_slice.index,
        )
        if x_pred.size > 0:
            pred_vals = model.predict(x_pred)
            for ts, val in zip(pred_ts, pred_vals):
                i = idx_map.get(ts)
                if i is not None:
                    preds[i] = float(val)

        start = end_pred

    return pd.Series(preds, index=frame.index, name="pred_next_ret")