"""
Combine the pattern-similarity score, current momentum, and news sentiment
into one 0-100 "Trend Score" per ticker, plus a handful of human-readable
tags (volume spike, near 52w high, sentiment surge) for the dashboard.

The composite score is a transparent weighted average, not a black box --
config.WEIGHT_PATTERN / WEIGHT_MOMENTUM / WEIGHT_SENTIMENT control it and
are shown on the dashboard's methodology panel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# Imported lazily (not at module load) so this module -- and everything that
# depends on it -- stays importable/testable even where vaderSentiment isn't
# installed (e.g. this project's dev sandbox, which can't reach PyPI for
# anything beyond its preinstalled packages; see README). On GitHub Actions,
# where requirements.txt is actually installed, this simply succeeds.
_analyzer = None


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def score_sentiment(news: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by ticker with columns:
    sentiment_compound (-1..1 average VADER compound over recent headlines),
    headline_count (buzz volume), sentiment_score (0-100, rescaled).
    Tickers with no headlines get neutral sentiment (50) and are flagged
    with headline_count == 0 so the dashboard can show "no recent coverage"
    rather than implying a real neutral reading.
    """
    if news.empty:
        base = pd.DataFrame(index=tickers)
        base["sentiment_compound"] = 0.0
        base["headline_count"] = 0
        base["sentiment_score"] = 50.0
        return base

    analyzer = _get_analyzer()
    scored = news.copy()
    scored["compound"] = scored["title"].fillna("").map(
        lambda t: analyzer.polarity_scores(t)["compound"]
    )
    grouped = scored.groupby("ticker").agg(
        sentiment_compound=("compound", "mean"),
        headline_count=("compound", "count"),
    )
    grouped["sentiment_score"] = ((grouped["sentiment_compound"] + 1) / 2) * 100

    result = pd.DataFrame(index=tickers)
    result = result.join(grouped)
    result["sentiment_compound"] = result["sentiment_compound"].fillna(0.0)
    result["headline_count"] = result["headline_count"].fillna(0).astype(int)
    result["sentiment_score"] = result["sentiment_score"].fillna(50.0)
    return result


def score_momentum(today_features: pd.DataFrame) -> pd.Series:
    """
    Percentile-rank today's blended momentum (5d/20d/60d, volume-weighted
    toward the shorter windows) across the whole scanned universe, so the
    score reflects *relative* strength within today's scan rather than an
    arbitrary absolute cutoff.
    """
    blend = (
        0.5 * today_features.get("mom_5d", 0)
        + 0.3 * today_features.get("mom_20d", 0)
        + 0.2 * today_features.get("mom_60d", 0)
    )
    pct_rank = blend.rank(pct=True) * 100
    return pct_rank.rename("momentum_score")


def composite_score(pattern: pd.Series, momentum: pd.Series, sentiment: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"pattern_score": pattern, "momentum_score": momentum, "sentiment_score": sentiment})
    df = df.dropna(subset=["pattern_score", "momentum_score"])
    df["sentiment_score"] = df["sentiment_score"].fillna(50.0)

    df["trend_score"] = (
        config.WEIGHT_PATTERN * df["pattern_score"]
        + config.WEIGHT_MOMENTUM * df["momentum_score"]
        + config.WEIGHT_SENTIMENT * df["sentiment_score"]
    ).clip(0, 100).round(1)

    return df


def build_tags(row: pd.Series) -> list[str]:
    tags = []
    if row.get("rel_volume", np.nan) and row["rel_volume"] >= 2.0:
        tags.append("volume spike")
    if row.get("pct_from_52w_high", np.nan) is not None and row.get("pct_from_52w_high", -1) >= -0.03:
        tags.append("near 52w high")
    if row.get("headline_count", 0) >= 8 and row.get("sentiment_compound", 0) > 0.25:
        tags.append("sentiment surge")
    if row.get("rsi_14", 50) >= 70:
        tags.append("overbought (RSI)")
    return tags
