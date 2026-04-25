"""Deterministic fusion model for multimodal trade scoring.

Combines:
  - tabular_score  (from XGB or Ridge)
  - sequence_score (from GRU)
  - sentiment_score (from news/FinBERT, default 0.0)
  - macro/regime features (from MacroProvider)

into a single fused prediction with deterministic policy gates:
  - abstain band     (edge too weak)
  - confidence gate  (below min threshold)
  - expected-return gate (below min threshold after costs)
  - CVaR gate        (portfolio tail risk too high)
  - regime gate      (high volatility → reduce or abstain)
  - news-risk gate   (news risk too high)

Output: FusionScore dataclass — NOT a model that rewrites its own logic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from marketify.config import ModelConfig

# ---------------------------------------------------------------------------
# Fusion weights (fixed, deterministic — no self-modification)
# ---------------------------------------------------------------------------
_W_TABULAR = 0.50
_W_SEQUENCE = 0.30
_W_SENTIMENT = 0.20


@dataclass
class FusionScore:
    """Output of the fusion model for a single bar."""
    expected_return: float
    confidence: float
    risk_flag: bool    # True → risk threshold exceeded
    abstain_flag: bool  # True → edge too weak, do not trade
    reason: str


@dataclass
class FusionConfig:
    """Deterministic policy thresholds.  All values are fixed at construction."""
    min_confidence: float = 0.30      # below → abstain
    min_expected_return: float = 3e-4  # below |return| → abstain (cost threshold)
    abstain_margin: float = 0.0        # extra edge buffer above cost threshold
    max_cvar_95: float = 0.02          # above → risk_flag
    max_news_risk: float = 0.75        # above → risk_flag + abstain
    high_vol_vix_threshold: float = 30.0  # VIX proxy above → risk_flag
    max_model_disagreement: float = 0.006  # tabular vs recurrent disagreement gate


class FusionPredictor:
    """Deterministic fusion predictor.

    No trainable weights; no self-modifying logic.
    Policy rules are purely arithmetic / comparison operations.
    """

    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig()

    def score(
        self,
        tabular_score: float,
        sequence_score: float,
        sentiment_score: float = 0.0,
        cvar_95: float = 0.0,
        news_risk: float = 0.0,
        vix_proxy: float = 20.0,
        model_disagreement: float | None = None,
    ) -> FusionScore:
        """Compute fused score and apply deterministic policy gates.

        All inputs are scalars; all gates are deterministic comparisons.
        """
        # 1. Weighted linear combination — fixed weights, no learning.
        raw = (
            _W_TABULAR * float(tabular_score)
            + _W_SEQUENCE * float(sequence_score)
            + _W_SENTIMENT * float(sentiment_score)
        )

        # 2. Confidence: normalised magnitude clamped to [0, 1].
        edge_floor = self.config.min_expected_return + self.config.abstain_margin
        scale = max(edge_floor, 1e-6)
        confidence = float(np.clip(abs(raw) / scale, 0.0, 1.0))

        # 3. Risk flags (deterministic).
        cvar_flag = float(cvar_95) > self.config.max_cvar_95
        news_flag = float(news_risk) > self.config.max_news_risk
        regime_flag = float(vix_proxy) > self.config.high_vol_vix_threshold
        disagreement = abs(float(tabular_score) - float(sequence_score))
        if model_disagreement is not None:
            disagreement = float(model_disagreement)
        disagreement_flag = disagreement > self.config.max_model_disagreement
        risk_flag = cvar_flag or news_flag or regime_flag

        # 4. Abstain decision.
        weak_edge = abs(raw) <= edge_floor
        low_confidence = confidence < self.config.min_confidence
        abstain_flag = weak_edge or low_confidence or news_flag or regime_flag or disagreement_flag

        reasons: list[str] = []
        if weak_edge:
            reasons.append(f"weak_edge(|{raw:.5f}|<={edge_floor:.5f})")
        if low_confidence:
            reasons.append(f"low_conf({confidence:.3f}<{self.config.min_confidence})")
        if news_flag:
            reasons.append(f"news_risk({news_risk:.2f}>{self.config.max_news_risk})")
        if cvar_flag:
            reasons.append(f"cvar_gate({cvar_95:.4f}>{self.config.max_cvar_95})")
        if regime_flag:
            reasons.append(f"regime(vix={vix_proxy:.1f}>{self.config.high_vol_vix_threshold})")
        if disagreement_flag:
            reasons.append(
                f"disagreement({disagreement:.5f}>{self.config.max_model_disagreement:.5f})"
            )

        return FusionScore(
            expected_return=float(raw),
            confidence=confidence,
            risk_flag=risk_flag,
            abstain_flag=abstain_flag,
            reason=", ".join(reasons) if reasons else "trade_ok",
        )


# ---------------------------------------------------------------------------
# Rolling-window fusion predictor — same interface as xgb/ridge/gru
# ---------------------------------------------------------------------------

def rolling_train_predict(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    config: ModelConfig,
    fusion_config: FusionConfig | None = None,
) -> pd.Series:
    """Walk-forward fusion prediction.

    Steps per window:
      1. Compute tabular (XGB) predictions via rolling window.
      2. Compute sequence (GRU) predictions via rolling window.
      3. Fuse with deterministic weights.
      4. Apply abstain gate: set NaN for abstained bars (no trade).

    Falls back gracefully when GRU / torch unavailable.
    """
    from marketify.models.rnn_model import rolling_train_predict as _gru_rtp
    from marketify.models.xgb_model import rolling_train_predict as _xgb_rtp

    # --- tabular lane ---
    tabular_preds = _xgb_rtp(frame, feature_cols, target_col, config)

    # --- sequence lane (GRU) — fall back to zeros on error ---
    try:
        seq_preds = _gru_rtp(frame, feature_cols, target_col, config)
    except Exception:
        seq_preds = pd.Series(0.0, index=frame.index, name="pred_next_ret")

    return fuse_prediction_components(
        frame=frame,
        tabular_preds=tabular_preds,
        sequence_preds=seq_preds,
        fusion_config=fusion_config,
    )


def fuse_prediction_components(
    frame: pd.DataFrame,
    tabular_preds: pd.Series,
    sequence_preds: pd.Series,
    fusion_config: FusionConfig | None = None,
) -> pd.Series:
    """Fuse cached tabular + recurrent predictions under deterministic policy rules."""
    f_cfg = fusion_config or FusionConfig()
    predictor = FusionPredictor(f_cfg)

    # Align indices
    tab = tabular_preds.reindex(frame.index)
    seq = sequence_preds.reindex(frame.index).fillna(0.0)
    sent = (
        pd.to_numeric(frame["news_sentiment_score"], errors="coerce").reindex(frame.index).fillna(0.0)
        if "news_sentiment_score" in frame.columns
        else pd.Series(0.0, index=frame.index)
    )
    news_risk = (
        pd.to_numeric(frame["news_risk"], errors="coerce").reindex(frame.index).fillna(0.0)
        if "news_risk" in frame.columns
        else pd.Series(0.0, index=frame.index)
    )
    if "news_event_shock" in frame.columns:
        shock = pd.to_numeric(frame["news_event_shock"], errors="coerce").reindex(frame.index).fillna(0.0)
        news_risk = pd.Series(np.maximum(news_risk.to_numpy(), shock.to_numpy()), index=frame.index)
    if "vix_proxy" in frame.columns:
        vix_proxy = pd.to_numeric(frame["vix_proxy"], errors="coerce").reindex(frame.index).fillna(20.0)
    elif "vol_20" in frame.columns:
        # Deterministic local proxy for regime stress from realized vol.
        vix_proxy = pd.to_numeric(frame["vol_20"], errors="coerce").reindex(frame.index).fillna(0.0) * 1000.0
    else:
        vix_proxy = pd.Series(20.0, index=frame.index)
    disagreement = (tab.fillna(0.0) - seq.fillna(0.0)).abs()

    fused = np.full(len(frame), np.nan, dtype=float)
    for i, ts in enumerate(frame.index):
        t_val = tab.iloc[i]
        if np.isnan(t_val):
            continue  # warmup: leave as NaN
        s_val = float(seq.iloc[i])
        snt = float(sent.iloc[i])
        fs = predictor.score(
            tabular_score=t_val,
            sequence_score=s_val,
            sentiment_score=snt,
            news_risk=float(news_risk.iloc[i]),
            vix_proxy=float(vix_proxy.iloc[i]),
            model_disagreement=float(disagreement.iloc[i]),
        )
        if fs.abstain_flag:
            fused[i] = np.nan  # abstain → no trade this bar
        else:
            fused[i] = fs.expected_return

    return pd.Series(fused, index=frame.index, name="pred_next_ret")
