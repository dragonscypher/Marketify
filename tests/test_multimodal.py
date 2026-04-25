"""Tests for multimodal pipeline:
- neutral fallback when news unavailable
- fusion model returns abstain on weak edge
- no trade when news-risk gate triggered
- multimodal benchmark writes all 5 report files
- 1% target reported honestly as PASS or FAIL
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# news_data — neutral fallback
# ---------------------------------------------------------------------------

class TestNewsNeutralFallback:
    def test_no_api_key_returns_empty(self):
        from marketify.data.news_data import NewsProvider

        provider = NewsProvider(alpha_vantage_key=None, finnhub_key=None)
        items = provider.fetch("AAPL")
        assert items == []

    def test_summarize_risk_neutral_when_no_key(self):
        from marketify.data.news_data import NEUTRAL_SUMMARY, NewsProvider

        provider = NewsProvider(alpha_vantage_key=None, finnhub_key=None)
        summary = provider.summarize_risk("AAPL")
        assert summary["news_risk"] == 0.0
        assert summary["sentiment_score"] == 0.0
        assert summary["headline_count"] == 0

    def test_aggregate_sentiment_empty_list(self):
        from marketify.data.news_data import NewsProvider

        provider = NewsProvider()
        assert provider.compute_aggregate_sentiment([]) == 0.0

    def test_aggregate_sentiment_items(self):
        from marketify.data.news_data import NewsItem, NewsProvider

        items = [
            NewsItem("Headline A", "src", "2026-01-01", sentiment_hint=0.5),
            NewsItem("Headline B", "src", "2026-01-01", sentiment_hint=-0.1),
        ]
        provider = NewsProvider()
        score = provider.compute_aggregate_sentiment(items)
        assert abs(score - 0.2) < 1e-9


# ---------------------------------------------------------------------------
# sentiment.py — compute_sentiment_features + news_risk_gate
# ---------------------------------------------------------------------------

class TestSentimentFeatures:
    def test_empty_items_returns_neutral_defaults(self):
        from marketify.features.sentiment import compute_sentiment_features

        feats = compute_sentiment_features([])
        assert feats["sentiment_mean"] == 0.0
        assert feats["sentiment_std"] == 0.0
        assert feats["headline_count"] == 0.0

    def test_positive_items(self):
        from marketify.data.news_data import NewsItem
        from marketify.features.sentiment import compute_sentiment_features

        items = [NewsItem("Good news", "src", "2026-01-01", sentiment_hint=0.8)]
        feats = compute_sentiment_features(items)
        assert feats["sentiment_mean"] > 0.0
        assert feats["headline_count"] == 1.0

    def test_news_risk_gate_below_threshold(self):
        from marketify.features.sentiment import news_risk_gate

        # 0.5 < 0.75 → should NOT abstain
        assert news_risk_gate(0.5, max_news_risk=0.75) is False

    def test_news_risk_gate_above_threshold(self):
        from marketify.features.sentiment import news_risk_gate

        # 0.9 > 0.75 → should abstain
        assert news_risk_gate(0.9, max_news_risk=0.75) is True

    def test_news_risk_gate_at_boundary(self):
        from marketify.features.sentiment import news_risk_gate

        # Exactly at threshold is not above → False
        assert news_risk_gate(0.75, max_news_risk=0.75) is False


# ---------------------------------------------------------------------------
# macro_data — safe neutral regime
# ---------------------------------------------------------------------------

class TestMacroProvider:
    def test_regime_features_returns_dict(self):
        from marketify.data.macro_data import MacroProvider

        provider = MacroProvider(alpha_vantage_key=None)
        feats = provider.regime_features("AAPL")
        assert isinstance(feats, dict)
        assert "vix_proxy" in feats
        assert "regime_neutral" in feats

    def test_neutral_regime_values(self):
        from marketify.data.macro_data import MacroProvider

        provider = MacroProvider()
        r = provider.get_regime("AAPL")
        assert r.regime in {"bull", "bear", "neutral"}
        assert 0.0 <= r.vix_proxy <= 100.0


# ---------------------------------------------------------------------------
# fusion_model — abstain on weak edge + news-risk gate
# ---------------------------------------------------------------------------

class TestFusionModel:
    def test_abstain_on_weak_edge(self):
        from marketify.models.fusion_model import FusionConfig, FusionPredictor

        cfg = FusionConfig(min_expected_return=1e-3, min_confidence=0.3)
        predictor = FusionPredictor(cfg)
        # Near-zero signal → should abstain
        score = predictor.score(
            tabular_score=1e-6,
            sequence_score=1e-6,
            sentiment_score=0.0,
        )
        assert score.abstain_flag is True

    def test_trade_ok_on_strong_edge(self):
        from marketify.models.fusion_model import FusionConfig, FusionPredictor

        cfg = FusionConfig(min_expected_return=1e-4, min_confidence=0.1)
        predictor = FusionPredictor(cfg)
        score = predictor.score(
            tabular_score=0.01,
            sequence_score=0.008,
            sentiment_score=0.0,
        )
        assert score.abstain_flag is False
        assert score.expected_return > 0.0

    def test_abstain_when_news_risk_too_high(self):
        from marketify.models.fusion_model import FusionConfig, FusionPredictor

        cfg = FusionConfig(max_news_risk=0.75)
        predictor = FusionPredictor(cfg)
        score = predictor.score(
            tabular_score=0.01,
            sequence_score=0.01,
            sentiment_score=0.0,
            news_risk=0.9,  # above threshold
        )
        assert score.risk_flag is True
        assert score.abstain_flag is True

    def test_risk_flag_on_high_cvar(self):
        from marketify.models.fusion_model import FusionConfig, FusionPredictor

        cfg = FusionConfig(max_cvar_95=0.02)
        predictor = FusionPredictor(cfg)
        score = predictor.score(
            tabular_score=0.01,
            sequence_score=0.01,
            sentiment_score=0.0,
            cvar_95=0.05,  # above threshold
        )
        assert score.risk_flag is True

    def test_abstain_when_model_disagreement_too_wide(self):
        from marketify.models.fusion_model import FusionConfig, FusionPredictor

        cfg = FusionConfig(min_expected_return=1e-4, min_confidence=0.1, max_model_disagreement=0.001)
        predictor = FusionPredictor(cfg)
        score = predictor.score(
            tabular_score=0.010,
            sequence_score=-0.010,
            sentiment_score=0.0,
        )
        assert score.abstain_flag is True
        assert "disagreement" in score.reason

    def test_no_self_modification(self):
        """Fusion predictor must not mutate its own policy thresholds."""
        from marketify.models.fusion_model import FusionConfig, FusionPredictor

        cfg = FusionConfig(min_confidence=0.5)
        predictor = FusionPredictor(cfg)
        original_threshold = predictor.config.min_confidence

        for _ in range(10):
            predictor.score(tabular_score=0.01, sequence_score=0.01)

        assert predictor.config.min_confidence == original_threshold

    def test_fusion_weights_deterministic(self):
        """Same inputs → same output every call."""
        from marketify.models.fusion_model import FusionPredictor

        p = FusionPredictor()
        s1 = p.score(tabular_score=0.005, sequence_score=0.003, sentiment_score=0.1)
        s2 = p.score(tabular_score=0.005, sequence_score=0.003, sentiment_score=0.1)
        assert s1.expected_return == s2.expected_return
        assert s1.confidence == s2.confidence


# ---------------------------------------------------------------------------
# run_multimodal_benchmark — report file writing
# ---------------------------------------------------------------------------

class TestMultimodalBenchmarkReports:
    """Integration test: run minimal benchmark, verify all 5 report files are written."""

    def _make_dummy_rows(self, tmp_dir: Path) -> list[dict]:
        """Return fake row list mimicking benchmark runner output."""
        # Fake equity curve
        equity = pd.Series([10000.0 + i * 5 for i in range(50)])
        trade_diag = pd.DataFrame({
            "pnl": [10.0, -5.0, 8.0, -2.0, 12.0],
        })
        return [
            {
                "model": "ridge",
                "metrics": {
                    "weekly_return_pct": 0.25,
                    "daily_return_pct": 0.05,
                    "sharpe_ratio": 0.30,
                    "sortino_ratio": 0.28,
                    "max_drawdown_pct": 0.40,
                    "cvar_95_pct": 1.20,
                    "trade_count": 5,
                    "win_rate_pct": 60.0,
                    "expectancy": 4.6,
                    "weekly_pass": False,
                    "daily_pass": False,
                    "weekly_status": "FAIL",
                    "daily_status": "FAIL",
                    "target_weekly_pct": 1.0,
                    "target_daily_pct": 1.0,
                },
                "equity": equity,
                "trade_diagnostics": trade_diag,
            }
        ]

    def test_all_five_report_files_written(self, tmp_path):
        import scripts.run_multimodal_benchmark as mb

        rows = self._make_dummy_rows(tmp_path)
        best = rows[0]
        mb._write_leaderboard(rows, tmp_path)
        mb._write_real_benchmark_md(best, rows, tmp_path)
        mb._write_trades_and_equity(best["model"], rows, tmp_path)

        for fname in [
            "multimodal_leaderboard.csv",
            "multimodal_leaderboard.md",
            "multimodal_real_benchmark.md",
            "multimodal_trades.csv",
            "multimodal_equity_curve.csv",
        ]:
            assert (tmp_path / fname).exists(), f"Missing: {fname}"

    def test_fail_reported_honestly_in_markdown(self, tmp_path):
        import scripts.run_multimodal_benchmark as mb

        rows = self._make_dummy_rows(tmp_path)
        best = rows[0]
        mb._write_real_benchmark_md(best, rows, tmp_path)
        content = (tmp_path / "multimodal_real_benchmark.md").read_text()
        assert "FAIL" in content
        assert "0.25" in content  # actual weekly return present

    def test_pass_reported_when_above_target(self, tmp_path):
        import scripts.run_multimodal_benchmark as mb

        equity = pd.Series([10000.0 * (1.01 ** i) for i in range(50)])
        rows = [
            {
                "model": "fusion",
                "metrics": {
                    "weekly_return_pct": 1.50,
                    "daily_return_pct": 0.20,
                    "sharpe_ratio": 1.20,
                    "sortino_ratio": 1.10,
                    "max_drawdown_pct": 0.30,
                    "cvar_95_pct": 0.80,
                    "trade_count": 20,
                    "win_rate_pct": 65.0,
                    "expectancy": 12.0,
                    "weekly_pass": True,
                    "daily_pass": False,
                    "weekly_status": "PASS",
                    "daily_status": "FAIL",
                    "target_weekly_pct": 1.0,
                    "target_daily_pct": 1.0,
                },
                "equity": equity,
                "trade_diagnostics": pd.DataFrame({"pnl": [10.0] * 20}),
            }
        ]
        best = rows[0]
        mb._write_real_benchmark_md(best, rows, tmp_path)
        content = (tmp_path / "multimodal_real_benchmark.md").read_text()
        assert "PASS" in content
        # Weekly PASS → no WARNING block should appear
        assert "WARNING: FAIL" not in content

    def test_pick_best_model_by_weekly_return(self):
        import scripts.run_multimodal_benchmark as mb

        rows = [
            {"model": "ridge", "metrics": {"weekly_return_pct": 0.5, "sharpe_ratio": 0.4,
                                           "max_drawdown_pct": 0.3, "trade_count": 10}},
            {"model": "fusion", "metrics": {"weekly_return_pct": 0.9, "sharpe_ratio": 0.3,
                                            "max_drawdown_pct": 0.2, "trade_count": 5}},
        ]
        best = mb._pick_best_model(rows)
        assert best["model"] == "fusion"

    def test_policy_tuning_and_next_status_files_written(self, tmp_path):
        import scripts.run_multimodal_benchmark as mb

        tuning_rows = [
            {
                "policy": {
                    "fusion_min_confidence": 0.35,
                    "fusion_expected_return_floor": 0.00035,
                    "fusion_max_model_disagreement": 0.004,
                    "fusion_news_risk_cutoff": 0.70,
                    "fusion_regime_vix_cutoff": 28.0,
                    "fusion_min_holding_bars": 6,
                },
                "metrics": {
                    "weekly_return_pct": 0.9,
                    "daily_return_pct": 0.05,
                    "sharpe_ratio": 0.4,
                    "sortino_ratio": 0.8,
                    "max_drawdown_pct": 0.3,
                    "cvar_95_pct": 0.1,
                    "trade_count": 12,
                    "win_rate_pct": 58.0,
                    "expectancy": 1.0,
                    "weekly_pass": False,
                    "daily_pass": False,
                    "weekly_status": "FAIL",
                    "daily_status": "FAIL",
                    "target_weekly_pct": 1.0,
                    "target_daily_pct": 1.0,
                },
                "validation_start_ts": "2026-01-01 00:00:00",
            }
        ]
        baseline = {
            "model": "fusion",
            "metrics": {
                "weekly_return_pct": 0.9833,
                "daily_return_pct": 0.0405,
                "sharpe_ratio": 0.3512,
                "sortino_ratio": 1.2457,
                "max_drawdown_pct": 0.7898,
                "cvar_95_pct": 0.0556,
                "trade_count": 45,
                "win_rate_pct": 55.56,
                "expectancy": 1.2453,
                "weekly_pass": False,
                "daily_pass": False,
                "weekly_status": "FAIL",
                "daily_status": "FAIL",
                "target_weekly_pct": 1.0,
                "target_daily_pct": 1.0,
            },
        }
        tuned = {
            "model": "fusion",
            "metrics": {
                **baseline["metrics"],
                "weekly_return_pct": 1.01,
                "weekly_pass": True,
                "weekly_status": "PASS",
            },
        }
        winner = tuned

        mb._write_fusion_policy_tuning(tuning_rows, tmp_path)
        mb._write_next_status(baseline, tuning_rows[0], tuned, winner, tmp_path)

        assert (tmp_path / "fusion_policy_tuning.csv").exists()
        assert (tmp_path / "fusion_policy_tuning.md").exists()
        assert (tmp_path / "NEXT_STATUS.md").exists()


# ---------------------------------------------------------------------------
# FinBERTSentiment — neutral fallback
# ---------------------------------------------------------------------------

class TestFinBERTNeutralFallback:
    def test_disabled_returns_neutral(self):
        from marketify.features.sentiment import FinBERTSentiment

        sent = FinBERTSentiment(enabled=False)
        result = sent.analyze("AAPL earnings beat")
        assert result.score == 0.0
        assert result.label == "neutral"
