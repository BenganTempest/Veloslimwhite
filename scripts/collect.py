"""
Data collection: OHLCV price history via yfinance, and recent headlines via
Google News RSS (free, keyless). Both are best-effort -- a ticker that fails
to fetch is dropped from that run rather than crashing the whole pipeline,
and a logged summary at the end tells you how many were skipped.
"""

from __future__ import annotations

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import feedparser
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

log = logging.getLogger("collect")


def _download_batch(tickers: list[str]) -> dict[str, pd.DataFrame]:
    if not tickers:
        return {}
    raw = yf.download(
        tickers=tickers,
        period=config.PRICE_LOOKBACK,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )

    out: dict[str, pd.DataFrame] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            try:
                df = raw[t].dropna(how="all")
            except KeyError:
                continue
            if df.empty or len(df) < 60:
                continue
            out[t] = df
    else:
        # single-ticker call falls back to a flat frame
        if not raw.empty and len(raw) >= 60:
            out[tickers[0]] = raw.dropna(how="all")
    return out


def _download_in_batches(tickers: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    size = max(1, int(config.PRICE_DOWNLOAD_BATCH_SIZE))
    batches = [tickers[i:i + size] for i in range(0, len(tickers), size)]
    for n, batch in enumerate(batches):
        try:
            out.update(_download_batch(batch))
        except Exception as exc:  # noqa: BLE001
            log.warning("Price batch %d/%d failed outright (%s) -- will be retried", n + 1, len(batches), exc)
        if n < len(batches) - 1 and config.PRICE_DOWNLOAD_BATCH_PAUSE_SECONDS:
            time.sleep(config.PRICE_DOWNLOAD_BATCH_PAUSE_SECONDS)
    return out


def fetch_prices(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """
    Download daily OHLCV for every ticker, in batches of
    config.PRICE_DOWNLOAD_BATCH_SIZE. Returns {ticker: DataFrame} for
    tickers that returned usable data.

    A ticker can come back missing either because it's genuinely delisted/
    invalid, or because Yahoo rate-limited the batch (YFRateLimitError) --
    common at this scale and usually transient. Up to
    config.PRICE_DOWNLOAD_MAX_RETRIES retries of just the still-missing
    tickers, each after a longer pause, recover most of the rate-limited
    ones without re-downloading everything that already succeeded.
    """
    log.info("Downloading price history for %d tickers...", len(tickers))
    out = _download_in_batches(tickers)

    for attempt in range(1, config.PRICE_DOWNLOAD_MAX_RETRIES + 1):
        missing = [t for t in tickers if t not in out]
        if not missing:
            break
        pause = config.PRICE_DOWNLOAD_RETRY_PAUSE_SECONDS * attempt
        log.warning(
            "%d/%d tickers missing (delistings and/or rate limiting) -- retry %d/%d in %ds",
            len(missing), len(tickers), attempt, config.PRICE_DOWNLOAD_MAX_RETRIES, pause,
        )
        time.sleep(pause)
        retry_out = _download_in_batches(missing)
        out.update(retry_out)
        log.info("Retry %d recovered %d/%d previously-missing tickers", attempt, len(retry_out), len(missing))

    log.info("Got usable price history for %d/%d tickers", len(out), len(tickers))
    return out


def _fetch_headlines_one(ticker: str, company_name: str) -> list[dict]:
    query = quote(f"{ticker} stock OR {company_name}")
    url = f"https://news.google.com/rss/search?q={query}+when:{config.NEWS_LOOKBACK_DAYS}d&hl=en-US&gl=US&ceid=US:en"
    try:
        parsed = feedparser.parse(url)
    except Exception as exc:  # noqa: BLE001
        log.debug("News fetch failed for %s: %s", ticker, exc)
        return []
    items = []
    for entry in parsed.entries[: config.NEWS_MAX_HEADLINES_PER_TICKER]:
        items.append(
            {
                "ticker": ticker,
                "title": getattr(entry, "title", ""),
                "published": getattr(entry, "published", ""),
                "link": getattr(entry, "link", ""),
            }
        )
    return items


def fetch_news(universe: pd.DataFrame) -> pd.DataFrame:
    """
    Best-effort headline fetch for every ticker in the universe.
    If this whole source is unreachable, returns an empty DataFrame and the
    scoring stage simply drops the sentiment term for this run.
    """
    if not config.FETCH_NEWS:
        return pd.DataFrame(columns=["ticker", "title", "published", "link"])

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = {
            pool.submit(_fetch_headlines_one, r.ticker, r.name): r.ticker
            for r in universe.itertuples()
        }
        for fut in as_completed(futures):
            try:
                rows.extend(fut.result())
            except Exception as exc:  # noqa: BLE001
                log.debug("News future failed for %s: %s", futures[fut], exc)

    df = pd.DataFrame(rows)
    log.info("Collected %d headlines across %d tickers", len(df), df["ticker"].nunique() if not df.empty else 0)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from universe import build_universe

    u = build_universe().head(10)
    prices = fetch_prices(u["ticker"].tolist())
    news = fetch_news(u)
    print(f"Prices for {len(prices)} tickers, {len(news)} headlines")
