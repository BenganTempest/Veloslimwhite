"""
Build the ticker universe to scan: S&P 500 + Nasdaq-100 constituents,
pulled from their Wikipedia pages (no API key, no rate limit issues).

Wikipedia's tables occasionally change column names or structure -- if a
table fetch fails, we log a warning and fall back to whatever is already
cached on disk (data/universe.csv) rather than crashing the whole pipeline.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

log = logging.getLogger("universe")

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NASDAQ100_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"

HEADERS = {"User-Agent": config.REQUEST_USER_AGENT}


def _clean_symbol(sym: str) -> str:
    """Wikipedia lists e.g. 'BRK.B' -- yfinance wants 'BRK-B'."""
    return str(sym).strip().upper().replace(".", "-")


def _fetch_table(url: str) -> list[pd.DataFrame]:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return pd.read_html(resp.text)


def fetch_sp500() -> pd.DataFrame:
    tables = _fetch_table(SP500_URL)
    df = tables[0]
    df = df.rename(columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector"})
    df["ticker"] = df["ticker"].map(_clean_symbol)
    df["index"] = "S&P 500"
    return df[["ticker", "name", "sector", "index"]]


def fetch_nasdaq100() -> pd.DataFrame:
    tables = _fetch_table(NASDAQ100_URL)
    # The constituents table is identified by having a Ticker/Symbol column;
    # its position on the page has moved before, so search for it instead of
    # hardcoding a table index.
    candidate = None
    for t in tables:
        cols = {c.strip().lower() for c in t.columns.astype(str)}
        if {"ticker", "company"} & cols or {"symbol", "company"} & cols:
            candidate = t
            break
    if candidate is None:
        raise ValueError("Could not locate Nasdaq-100 constituents table")

    candidate = candidate.rename(columns=lambda c: str(c).strip())
    sym_col = "Ticker" if "Ticker" in candidate.columns else "Symbol"
    name_col = "Company" if "Company" in candidate.columns else candidate.columns[0]
    sector_col = "GICS Sector" if "GICS Sector" in candidate.columns else None

    df = pd.DataFrame(
        {
            "ticker": candidate[sym_col].map(_clean_symbol),
            "name": candidate[name_col],
            "sector": candidate[sector_col] if sector_col else "Unknown",
        }
    )
    df["index"] = "Nasdaq-100"
    return df


def build_universe() -> pd.DataFrame:
    frames = []
    if config.INCLUDE_SP500:
        try:
            frames.append(fetch_sp500())
        except Exception as exc:  # noqa: BLE001
            log.warning("S&P 500 fetch failed: %s", exc)
    if config.INCLUDE_NASDAQ100:
        try:
            frames.append(fetch_nasdaq100())
        except Exception as exc:  # noqa: BLE001
            log.warning("Nasdaq-100 fetch failed: %s", exc)

    if not frames:
        if config.UNIVERSE_CACHE.exists():
            log.warning("Falling back to cached universe on disk")
            return pd.read_csv(config.UNIVERSE_CACHE)
        raise RuntimeError("No universe source succeeded and no cache exists")

    combined = pd.concat(frames, ignore_index=True)
    # A ticker can appear in both indices -- keep one row, remember both.
    agg = (
        combined.groupby("ticker", as_index=False)
        .agg({"name": "first", "sector": "first", "index": lambda s: "+".join(sorted(set(s)))})
    )
    agg = agg.sort_values("ticker").reset_index(drop=True)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_csv(config.UNIVERSE_CACHE, index=False)
    return agg


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    u = build_universe()
    print(f"Universe: {len(u)} tickers")
    print(u.head(10))
