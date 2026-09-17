"""
Central configuration for the trend scanner.

Tune these constants to change behavior without touching pipeline logic.
Nothing here requires an API key -- every data source used by this project
is free and keyless (Wikipedia for the ticker universe, Yahoo Finance via
yfinance for prices, Google News RSS for headlines).
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"
UNIVERSE_CACHE = DATA_DIR / "universe.csv"
SNAPSHOT_HISTORY_CSV = DATA_DIR / "history.csv"  # small, committed, grows over time

# --- Universe ------------------------------------------------------------
# Which index constituent lists to combine into the scan universe.
INCLUDE_SP500 = True
INCLUDE_NASDAQ100 = True
INCLUDE_OMX_STOCKHOLM_ALL = True  # Nasdaq Stockholm all-share, incl. First North small-caps

# The Stockholm list has no built-in sector column, so missing sectors are
# backfilled with a per-ticker yfinance lookup (much heavier than the bulk
# price download, hence capped/threaded gently and cached across runs).
ENRICH_MISSING_SECTORS = True
SECTOR_ENRICH_MAX_WORKERS = 4
SECTOR_ENRICH_MAX_PER_RUN = 600

# --- Price / feature settings --------------------------------------------
PRICE_LOOKBACK = "2y"        # yfinance period string used for both training + features
MOMENTUM_WINDOWS = [5, 20, 60]   # trading days
VOLATILITY_WINDOW = 20
RSI_WINDOW = 14
VOLUME_SPIKE_SHORT = 5
VOLUME_SPIKE_LONG = 20

# --- Breakout / pattern-matching model -----------------------------------
# A "breakout" is defined purely mechanically: did the stock gain at least
# BREAKOUT_RETURN over the following BREAKOUT_WINDOW trading days.
# This is a labeling rule for backward-looking research, not a prediction.
BREAKOUT_RETURN = 0.20
BREAKOUT_WINDOW = 10
# Most recent slice of history held out for an honest out-of-sample check
# (time-based split -- never a random shuffle, to avoid look-ahead leakage).
VALIDATION_HOLDOUT_DAYS = 126  # ~6 trading months

# --- News / sentiment ------------------------------------------------------
FETCH_NEWS = True
NEWS_LOOKBACK_DAYS = 7
NEWS_MAX_HEADLINES_PER_TICKER = 15
# If the news source is unreachable or rate-limited, the pipeline should not
# hard-fail -- it just zeroes out the sentiment weight for that run.
NEWS_TIMEOUT_SECONDS = 8

# --- Composite score weights (must sum to 1.0) -----------------------------
WEIGHT_PATTERN = 0.55
WEIGHT_MOMENTUM = 0.25
WEIGHT_SENTIMENT = 0.20

# --- Misc ------------------------------------------------------------------
REQUEST_USER_AGENT = "stock-trend-scanner/1.0 (personal research project)"
MAX_WORKERS = 8
