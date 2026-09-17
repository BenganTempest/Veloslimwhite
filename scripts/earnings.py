"""
Best-effort "earnings soon" awareness for the top-scoring tickers each run.

Deliberately NOT run for the whole universe: this is a per-ticker yfinance
call (yf.Ticker(t).calendar), same shape as the sector backfill in
universe.py, and that backfill already taught us the hard way that a big
burst of individual Yahoo requests can trip the rate limiter for whatever
comes after it (see README limitations). So this only checks
config.EARNINGS_CHECK_TOP_N tickers -- the ones already interesting enough
to be near the top of today's scan -- runs after the bulk price download
is fully done, and fails soft: any ticker that errors, times out, or has no
calendar data just doesn't get the tag, it never blocks the pipeline.
"""

from __future__ import annotations

import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

log = logging.getLogger("earnings")


def _next_earnings_date_one(ticker: str) -> tuple[str, date | None]:
    try:
        # Imported lazily, same reasoning as elsewhere in this project: keeps
        # this module importable/testable without yfinance installed -- the
        # import itself must be inside the try, since a missing yfinance
        # install should fail soft here just like a bad network call would.
        import yfinance as yf

        cal = yf.Ticker(ticker).calendar
        if not cal:
            return ticker, None
        raw_dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if not raw_dates:
            return ticker, None
        # yfinance can return one date or a [start, end] estimate window --
        # either way, take the earliest one that hasn't already passed.
        candidates = raw_dates if isinstance(raw_dates, (list, tuple)) else [raw_dates]
        upcoming = sorted(d for d in (_coerce_date(c) for c in candidates) if d is not None)
        return ticker, (upcoming[0] if upcoming else None)
    except Exception:  # noqa: BLE001
        return ticker, None


def _coerce_date(value) -> date | None:
    try:
        if isinstance(value, date):
            return value
        return pd.Timestamp(value).date()
    except Exception:  # noqa: BLE001
        return None


def check_upcoming_earnings(tickers: list[str]) -> dict[str, date]:
    """
    Returns {ticker: next_earnings_date} for whichever of the given tickers
    (already capped to config.EARNINGS_CHECK_TOP_N by the caller) have a
    resolvable upcoming earnings date. Missing entirely just means "unknown
    or none found" -- never an error the pipeline needs to handle.
    """
    if not tickers:
        return {}
    capped = tickers[: config.EARNINGS_CHECK_TOP_N]
    log.info("Checking upcoming earnings dates for %d tickers...", len(capped))

    found: dict[str, date] = {}
    with ThreadPoolExecutor(max_workers=config.EARNINGS_CHECK_MAX_WORKERS) as pool:
        futures = {pool.submit(_next_earnings_date_one, t): t for t in capped}
        for fut in as_completed(futures):
            ticker, when = fut.result()
            if when is not None:
                found[ticker] = when

    log.info("Earnings check: found an upcoming date for %d/%d tickers", len(found), len(capped))
    return found


def tickers_with_earnings_soon(earnings_by_ticker: dict[str, date], today: date | None = None) -> set[str]:
    today = today or date.today()
    cutoff = today + timedelta(days=config.EARNINGS_SOON_DAYS)
    return {t for t, d in earnings_by_ticker.items() if today <= d <= cutoff}
