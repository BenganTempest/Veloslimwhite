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
# Downloading ~1000 tickers in one request is what trips Yahoo's limiter
# most often, so the first pass is split into smaller batches with a short
# pause between them. Missing tickers are then retried a few times with a
# growing pause (60s, 120s, 180s...).
PRICE_DOWNLOAD_BATCH_SIZE = 150
PRICE_DOWNLOAD_BATCH_PAUSE_SECONDS = 5
PRICE_DOWNLOAD_MAX_RETRIES = 3
PRICE_DOWNLOAD_RETRY_PAUSE_SECONDS = 60
# If more than this share of the universe is still missing after retries,
# the dashboard shows a warning banner and the weekly summary flags the run.
DATA_QUALITY_WARN_MISSING_PCT = 5.0
RUN_LOG_CSV = DATA_DIR / "run_log.csv"  # one row per run: how many tickers were fetched/missing

# --- Benchmarks (for market-relative momentum and the follow-up panel) ------
# Keyed by the same market labels the dashboard uses.
BENCHMARKS = {"US": "^GSPC", "Sweden": "^OMX"}  # S&P 500, OMX Stockholm 30
BENCHMARK_NAMES = {"US": "S&P 500", "Sweden": "OMXS30"}
BENCHMARK_HISTORY_CSV = DATA_DIR / "benchmarks.csv"

# --- Liquidity filter -------------------------------------------------------
# Very thinly traded stocks swing a lot on tiny volume and float to the top
# on momentum alone. Tickers whose average daily turnover (close x volume,
# in their own currency) over the last LIQUIDITY_WINDOW days is below these
# floors are left out of scoring. Tickers on WATCHLIST are always kept.
LIQUIDITY_WINDOW = 20
MIN_AVG_TURNOVER = {"USD": 1_000_000, "SEK": 2_000_000}

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

# --- Momentum: relative to market and sector -------------------------------
# Momentum is percentile-ranked within each market (US vs Sweden), so a broad
# rally in one market doesn't crowd the other out, and blended with a rank
# within the ticker's own sector (in the same market). Sector groups smaller
# than MOMENTUM_MIN_SECTOR_SIZE fall back to the market rank.
MOMENTUM_SECTOR_WEIGHT = 0.5
MOMENTUM_MIN_SECTOR_SIZE = 8

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
# Alert when a ticker crosses this. Scores are RELATIVE (momentum is a
# percentile rank, the pattern model is class-balanced), so on a normal day
# dozens of tickers sit above 75 -- 85 keeps alerts to the genuine top names.
TELEGRAM_SCORE_THRESHOLD = 85
# Hysteresis: once a ticker has alerted, it must fall BELOW this level before
# it can alert again. Stops a ticker hovering around the threshold from
# pinging you every other day.
TELEGRAM_RESET_THRESHOLD = 75
# Keep the message short even on a busy day: list at most this many tickers,
# then "+N till" and the dashboard link.
TELEGRAM_MAX_TICKERS_IN_MESSAGE = 5
# Remembers which tickers have already alerted (and not yet reset). Committed
# by the workflow so the memory survives between daily runs.
TELEGRAM_ALERT_STATE_FILE = DATA_DIR / "alert_state.json"
# Optional: your GitHub Pages URL, appended to alert messages so you can tap
# straight through. Leave blank to omit it.
DASHBOARD_URL = "https://bengantempest.github.io/Veloslimwhite/"

# --- Watchlist -----------------------------------------------------------------
# Tickers you own or follow (Yahoo symbols, e.g. "VOLV-B.ST", "AAPL"). They are
# always scanned (even if they fail the liquidity filter), always get a detail
# view on the dashboard, and get their own section in the Telegram message
# when their score moves at least WATCHLIST_ALERT_CHANGE points in a day or
# they cross TELEGRAM_SCORE_THRESHOLD. (Stars you set on the dashboard itself
# are saved in your browser only and don't reach Telegram.)
WATCHLIST: list[str] = []
WATCHLIST_ALERT_CHANGE = 10.0

# --- Follow-up ("did it work?") ------------------------------------------------
# Every alert is logged and later checked: how did the stock do N trading days
# later compared with its market's index? Same check for each day's top
# FOLLOWUP_TOP_N tickers, which gives data much sooner than alerts alone.
ALERT_LOG_CSV = DATA_DIR / "alert_log.csv"
FOLLOWUP_HORIZONS = [5, 10, 20]   # trading days (= scanner runs)
FOLLOWUP_TOP_N = 10

# --- Model versioning ----------------------------------------------------------
# Every snapshot, alert and run is tagged with the scoring-model version that
# produced it, so the follow-up and paper-portfolio results can be judged per
# version instead of silently mixing old and new methods. Bump MODEL_VERSION
# (and describe the change) whenever you change how scores are calculated.
# If you forget, the pipeline notices -- it fingerprints every
# scoring-relevant setting below -- registers an automatic sub-version and
# flags it in the Actions log. History is in data/model_versions.json.
MODEL_VERSION = "2.0"
MODEL_VERSION_NOTE = (
    "Momentum rankas inom marknad och sektor, mönsterdelen är en rankning, "
    "omsättningsfilter, kalibrerad chans."
)
# Snapshots written before versioning existed get this label.
LEGACY_MODEL_VERSION = "1.0"
MODEL_VERSIONS_FILE = DATA_DIR / "model_versions.json"

# --- Paper portfolio ---------------------------------------------------------------
# A simulated, cost-aware portfolio that follows the scanner mechanically:
# every PAPER_REBALANCE_EVERY runs (5 = about weekly) it holds the top
# PAPER_TOP_N tickers by Trend Score, equal-weighted. Compared with the same
# mix of market indexes (S&P 500 / OMXS30, weighted by the portfolio's own
# US/Sweden split). No real money, no broker, no paid data -- it is computed
# from data/history.csv and data/benchmarks.csv only.
PAPER_TOP_N = 10
PAPER_REBALANCE_EVERY = 5
# Trading cost per side, in basis points (0.01 %), covering brokerage plus
# half the bid/ask spread. Deliberately on the conservative side for small
# caps; adjust to your own broker.
PAPER_COST_BPS = {"US": 15, "Sweden": 30}
# The validation bar the strategy has to clear before it's considered more
# than an experiment (shown as a status on the dashboard).
PAPER_VALIDATION_DAYS = 126   # ~6 months of runs
PAPER_MIN_HIT_RATE = 55.0     # % of rebalance periods beating the index

# --- Detail view -------------------------------------------------------------
# Price chart, score history and headlines are embedded for the top
# DETAIL_TOP_N tickers (plus the watchlist) -- not all ~1000, to keep
# data.json (committed daily) reasonably small.
DETAIL_TOP_N = 150
DETAIL_PRICE_DAYS = 120
DETAIL_SCORE_DAYS = 60
DETAIL_HEADLINES = 5

# --- Weekly summary -------------------------------------------------------------
# Sent by .github/workflows/weekly.yml (Sunday morning) via the same bot.
WEEKLY_SUMMARY_ENABLED = True
WEEKLY_TOP_CLIMBERS = 5

# --- Repo-staleness alert -----------------------------------------------------
# GitHub auto-disables a PUBLIC repo's scheduled Actions workflows after 60
# days with no repository activity (commits) -- after that, this whole
# pipeline just silently stops running. Under healthy operation this is a
# non-issue: the daily run itself commits docs/data.json whenever the
# numbers change, which resets the clock every day on its own. This alert
# only matters if something breaks badly enough that the pipeline stops
# reaching its own commit step for a long stretch -- two early warnings,
# well before the 60-day cutoff, via the same free Telegram bot as above
# (reuses TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID; no extra setup).
STALENESS_WARN_DAYS = 30    # first, low-key heads-up
STALENESS_URGENT_DAYS = 45  # louder nudge if it's still stale two weeks later
STALENESS_STATE_FILE = DATA_DIR / "staleness_state.json"

# --- Misc ------------------------------------------------------------------
REQUEST_USER_AGENT = "stock-trend-scanner/1.0 (personal research project)"
MAX_WORKERS = 8
