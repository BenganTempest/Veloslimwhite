"""
Build the ticker universe to scan: S&P 500 + Nasdaq-100 constituents from
Wikipedia, plus (optionally) the Nasdaq Stockholm all-share list from
stockanalysis.com -- covering everything from Volvo/Ericsson down to small
First North names, since small-caps are exactly where a big % move is most
plausible.

Wikipedia's tables occasionally change column names or structure -- if a
table fetch fails, we log a warning and fall back to whatever is already
cached on disk (data/universe.csv) rather than crashing the whole pipeline.
Same fail-soft treatment for the Stockholm list: it's a less-proven scrape
than Wikipedia (an ordinary web page, not a documented API, and its exact
completeness -- full all-share vs. a partial page -- wasn't verifiable from
the environment this project was built in, which can't reach financial data
hosts; see the README). Check the Actions log line "OMX Stockholm: got N
tickers" after a run -- if N looks too low (a few hundred when ~700+ are
expected), the scraper likely needs adjusting for a page-structure change,
the same kind of fix already made once for the Wikipedia sources.
"""

from __future__ import annotations

import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

log = logging.getLogger("universe")

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NASDAQ100_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"
OMX_STOCKHOLM_URL = "https://stockanalysis.com/list/nasdaq-stockholm/"

HEADERS = {"User-Agent": config.REQUEST_USER_AGENT}


def _clean_symbol(sym: str) -> str:
    """Wikipedia lists e.g. 'BRK.B' -- yfinance wants 'BRK-B'."""
    return str(sym).strip().upper().replace(".", "-")


def _clean_symbol_stockholm(sym: str) -> str:
    """
    stockanalysis.com lists Stockholm tickers like 'VOLV.B', 'ATCO.B' with no
    exchange suffix. Yahoo Finance (and so yfinance) needs the share-class
    dot turned into a hyphen AND a '.ST' exchange suffix, e.g. 'VOLV-B.ST'.
    """
    cleaned = str(sym).strip().upper().replace(".", "-")
    return cleaned if cleaned.endswith(".ST") else f"{cleaned}.ST"


def _fetch_table(url: str) -> list[pd.DataFrame]:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    # pandas needs a file-like object here, not a raw string -- passing
    # resp.text directly makes newer pandas try to treat the HTML text
    # itself as a file path and fail with FileNotFoundError.
    return pd.read_html(StringIO(resp.text))


def fetch_sp500() -> pd.DataFrame:
    tables = _fetch_table(SP500_URL)
    df = tables[0]
    df = df.rename(columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector"})
    df["ticker"] = df["ticker"].map(_clean_symbol)
    df["index"] = "S&P 500"
    return df[["ticker", "name", "sector", "index"]]


def _find_column(columns, *substrings) -> str | None:
    """Case-insensitive substring match, e.g. 'Ticker symbol' matches 'ticker'.
    More resilient to Wikipedia renaming a header slightly than an exact match."""
    for col in columns:
        low = str(col).strip().lower()
        if any(s in low for s in substrings):
            return col
    return None


def fetch_nasdaq100() -> pd.DataFrame:
    tables = _fetch_table(NASDAQ100_URL)
    # The constituents table is identified by having a ticker-like AND a
    # company-like column; its position on the page has moved before, so
    # search for it (by substring, not exact name) instead of hardcoding a
    # table index or an exact header string.
    candidate = None
    sym_col = name_col = None
    for t in tables:
        t = t.rename(columns=lambda c: str(c).strip())
        sc = _find_column(t.columns, "ticker", "symbol")
        nc = _find_column(t.columns, "company")
        if sc and nc:
            candidate, sym_col, name_col = t, sc, nc
            break
    if candidate is None:
        # Log every table's columns so a future failure is diagnosable
        # straight from the Actions log instead of needing another round trip.
        seen = [list(t.columns.astype(str)) for t in tables]
        log.error("Nasdaq-100: no table had both a ticker-like and company-like column. "
                  "Tables found on the page: %s", seen)
        raise ValueError(f"Could not locate Nasdaq-100 constituents table (saw {len(tables)} tables, see log for their columns)")

    sector_col = _find_column(candidate.columns, "gics sector", "sector")

    df = pd.DataFrame(
        {
            "ticker": candidate[sym_col].map(_clean_symbol),
            "name": candidate[name_col],
            "sector": candidate[sector_col] if sector_col else "Unknown",
        }
    )
    df["index"] = "Nasdaq-100"
    return df


def fetch_omx_stockholm() -> pd.DataFrame:
    tables = _fetch_table(OMX_STOCKHOLM_URL)
    candidate = None
    for t in tables:
        cols = {c.strip().lower() for c in t.columns.astype(str)}
        if {"symbol", "company name"} & cols or {"symbol"} & cols:
            candidate = t
            break
    if candidate is None:
        raise ValueError("Could not locate Nasdaq Stockholm constituents table")

    candidate = candidate.rename(columns=lambda c: str(c).strip())
    sym_col = "Symbol"
    name_col = "Company Name" if "Company Name" in candidate.columns else candidate.columns[1]

    df = pd.DataFrame(
        {
            "ticker": candidate[sym_col].map(_clean_symbol_stockholm),
            "name": candidate[name_col],
            "sector": "Unknown",  # this source doesn't provide a sector column
        }
    )
    df = df.dropna(subset=["ticker", "name"])
    df["index"] = "OMX Stockholm"
    log.info("OMX Stockholm: got %d tickers", len(df))
    return df


def _fetch_sector_one(ticker: str) -> tuple[str, str]:
    # Imported lazily, same reasoning as vaderSentiment in score.py: keeps
    # this module importable/testable without yfinance installed.
    import yfinance as yf

    try:
        info = yf.Ticker(ticker).info
        sector = info.get("sector") or info.get("industry") or "Unknown"
        return ticker, str(sector)
    except Exception:  # noqa: BLE001
        return ticker, "Unknown"


def enrich_missing_sectors(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Wikipedia's S&P 500 / Nasdaq-100 tables come with a sector column built
    in; the Stockholm all-share scrape doesn't. This fills in "Unknown" rows
    with sector/industry from yfinance's per-ticker `.info` -- a much
    heavier call than the bulk price download, so it's capped, threaded
    gently, and entirely best-effort: any ticker that fails or times out
    just stays "Unknown" rather than blocking the pipeline.
    """
    missing = agg.loc[agg["sector"] == "Unknown", "ticker"].tolist()
    if not missing:
        return agg
    if len(missing) > config.SECTOR_ENRICH_MAX_PER_RUN:
        log.warning(
            "%d tickers missing sector, capping enrichment at %d this run "
            "(the rest stay Unknown and get a chance on a later run once cached)",
            len(missing), config.SECTOR_ENRICH_MAX_PER_RUN,
        )
        missing = missing[: config.SECTOR_ENRICH_MAX_PER_RUN]

    log.info("Fetching sector/industry for %d tickers with unknown sector...", len(missing))
    found: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=config.SECTOR_ENRICH_MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_sector_one, t): t for t in missing}
        for fut in as_completed(futures):
            t, sector = fut.result()
            found[t] = sector

    resolved = sum(1 for v in found.values() if v != "Unknown")
    log.info("Sector enrichment: resolved %d/%d tickers", resolved, len(missing))

    agg = agg.copy()
    mask = agg["ticker"].isin(found)
    agg.loc[mask, "sector"] = agg.loc[mask, "ticker"].map(found)
    return agg


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
    if config.INCLUDE_OMX_STOCKHOLM_ALL:
        try:
            frames.append(fetch_omx_stockholm())
        except Exception as exc:  # noqa: BLE001
            log.warning("OMX Stockholm fetch failed: %s", exc)

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

    # Reuse sectors already resolved on a previous run (cached on disk and
    # committed to the repo each day) before paying for any new lookups --
    # sector essentially never changes, so there's no reason to re-fetch it
    # daily once it's known.
    if config.UNIVERSE_CACHE.exists():
        try:
            prior = pd.read_csv(config.UNIVERSE_CACHE)[["ticker", "sector"]].dropna()
            prior_map = dict(zip(prior["ticker"], prior["sector"]))
            still_unknown = agg["sector"] == "Unknown"
            agg.loc[still_unknown, "sector"] = agg.loc[still_unknown, "ticker"].map(prior_map).fillna("Unknown")
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not reuse cached sectors: %s", exc)

    if config.ENRICH_MISSING_SECTORS:
        agg = enrich_missing_sectors(agg)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_csv(config.UNIVERSE_CACHE, index=False)
    return agg


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    u = build_universe()
    print(f"Universe: {len(u)} tickers")
    print(u.head(10))
