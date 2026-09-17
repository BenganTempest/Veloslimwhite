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
# Kept low on purpose: this is a burst of individual Yahoo requests right
# before the much bigger bulk price download, and a big burst here was
# observed to trip Yahoo's rate limiter for the price-download step right
# after it (160+ tickers failed with YFRateLimitError in one run). Because
# resolved sectors are cached in data/universe.csv and reused daily, a low
# cap here just means the one-time Swedish sector backfill trickles in over
# several days instead of happening in one risky burst.
SECTOR_ENRICH_MAX_PER_RUN = 150
# Give Yahoo's short-term rate-limit window a chance to reset between the
# sector-enrichment burst and the much larger bulk price download.
PAUSE_AFTER_SECTOR_ENRICHMENT_SECONDS = 20

# --- Price download resilience --------------------------------------------
# yfinance/Yahoo rate-limiting is common at this scale (~1000+ tickers) and
# is usually transient. One retry of just the tickers that failed the first
# pass, after a pause, recovers most of them without re-downloading
# everything that already succeeded.
PRICE_DOWNLOAD_MAX_RETRIES = 1
PRICE_DOWNLOAD_RETRY_PAUSE_SECONDS = 60

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

# --- Day-over-day movers ---------------------------------------------------
# Reuses data/history.csv (already collected for the honesty panel) to show
# which tickers' Trend Score moved the most since the previous run, not just
# who's highest today. No new data source, no extra cost.
SHOW_MOVERS = True
MOVERS_TOP_N = 15  # how many gainers/decliners to show on the dashboard

# --- Earnings-date awareness -------------------------------------------------
# A per-ticker yfinance call (like the sector backfill), so it's deliberately
# scoped to only the top-scoring tickers each run, not the whole universe --
# doing this for 1000+ tickers daily would risk the exact Yahoo rate-limiting
# problem the sector backfill already caused once (see README). Runs AFTER
# the bulk price download finishes and after its own pause, so it never
# competes with that download for Yahoo's short-term rate-limit budget.
CHECK_EARNINGS_DATES = True
EARNINGS_CHECK_TOP_N = 100          # only check the top N tickers by today's score
EARNINGS_CHECK_MAX_WORKERS = 3
EARNINGS_SOON_DAYS = 7              # tag as "earnings soon" if within this many days
PAUSE_BEFORE_EARNINGS_CHECK_SECONDS = 15

# --- Telegram alerts ---------------------------------------------------------
# Completely free (no billing, ever) -- a bot token from Telegram's own
# @BotFather and your personal chat ID, both stored as GitHub Actions
# secrets (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID), never in this repo. If
# those secrets aren't set, alerting is silently skipped -- the pipeline
# doesn't require this to work. See the README for setup steps.
TELEGRAM_ALERTS_ENABLED = True
TELEGRAM_SCORE_THRESHOLD = 75  # alert when a ticker crosses this for the first time
# Optional: your GitHub Pages URL, appended to alert messages so you can tap
# straight through. Leave blank to omit it.
DASHBOARD_URL = ""

# --- Misc ------------------------------------------------------------------
REQUEST_USER_AGENT = "stock-trend-scanner/1.0 (personal research project)"
MAX_WORKERS = 8
