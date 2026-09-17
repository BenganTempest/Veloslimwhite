"""
Feature engineering: turn raw OHLCV into the momentum/volume/volatility
indicators used both to train the historical pattern model and to score
today's snapshot. Every function here is pure (DataFrame in, DataFrame out)
so it can be unit-tested with synthetic data, independent of any network
access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df must have columns: Open, High, Low, Close, Volume, indexed by date.
    Returns a new DataFrame with feature columns aligned to the same index.
    Early rows (shorter history than the longest window) will be NaN and
    should be dropped by the caller before training/scoring.
    """
    out = pd.DataFrame(index=df.index)
    close = df["Close"]
    volume = df["Volume"]

    for w in config.MOMENTUM_WINDOWS:
        out[f"mom_{w}d"] = close.pct_change(w)

    out["volatility_20d"] = close.pct_change().rolling(config.VOLATILITY_WINDOW).std()

    roll_max = close.rolling(252, min_periods=20).max()
    out["pct_from_52w_high"] = (close - roll_max) / roll_max

    vol_short = volume.rolling(config.VOLUME_SPIKE_SHORT).mean()
    vol_long = volume.rolling(config.VOLUME_SPIKE_LONG).mean()
    out["rel_volume"] = vol_short / vol_long.replace(0, np.nan)

    out["rsi_14"] = _rsi(close, config.RSI_WINDOW)

    out["close"] = close
    return out


def forward_return(close: pd.Series, window: int) -> pd.Series:
    """Forward N-day return, used only for labeling training data."""
    return close.shift(-window) / close - 1.0


FEATURE_COLUMNS = [
    *[f"mom_{w}d" for w in config.MOMENTUM_WINDOWS],
    "volatility_20d",
    "pct_from_52w_high",
    "rel_volume",
    "rsi_14",
]
