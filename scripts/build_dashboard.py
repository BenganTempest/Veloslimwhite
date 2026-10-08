"""
Writes docs/data.json (everything the dashboard needs) and ensures
docs/index.html (static shell + JS renderer) exists. index.html itself
rarely changes -- it's written once by scaffold_dashboard_html() below and
then just reads a fresh data.json on every GitHub Pages visit. Keeping data
and markup separate means a daily pipeline run only touches data.json plus
the small history.csv, not a multi-hundred-KB HTML file, which keeps git
diffs small and readable.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from backtest import BacktestResult  # noqa: E402


def _sparkline(closes: pd.Series, n: int = 30) -> list[float]:
    tail = closes.dropna().tail(n)
    return [round(float(x), 4) for x in tail.tolist()]


def _num(v, digits: int, scale: float = 1.0):
    """Float rounded for JSON, or None for missing/NaN (data.json is written with allow_nan=False)."""
    try:
        if v is None or pd.isna(v):
            return None
        return round(float(v) * scale, digits)
    except (TypeError, ValueError):
        return None


def _sig(x: float) -> float:
    """Round a price to 5 significant digits -- plenty for a chart, much smaller JSON."""
    return float(f"{x:.5g}")


def build_details(
    tickers: list[str],
    prices: dict[str, pd.DataFrame],
    history: pd.DataFrame,
    news: pd.DataFrame,
) -> dict:
    """
    Per-ticker data for the dashboard's detail view: recent closes, the
    ticker's own Trend Score history (aligned to the shared list of recent
    run dates) and its latest headlines. Only built for the tickers passed
    in (top DETAIL_TOP_N + watchlist) to keep data.json small.
    """
    run_dates = sorted(history["date"].astype(str).unique())[-config.DETAIL_SCORE_DAYS:] if not history.empty else []
    score_lookup = {}
    if run_dates:
        recent = history[history["date"].astype(str).isin(run_dates)]
        score_lookup = {(str(d), t): s for d, t, s in zip(recent["date"], recent["ticker"], recent["trend_score"])}
    heads_by_ticker = {}
    if news is not None and not news.empty:
        for t, g in news.groupby("ticker"):
            heads_by_ticker[t] = [
                {"title": str(x.title), "published": str(getattr(x, "published", "") or "")[:16]}
                for x in g.head(config.DETAIL_HEADLINES).itertuples()
                if str(x.title)
            ]
    out = {}
    for t in tickers:
        closes = prices[t]["Close"].dropna().tail(config.DETAIL_PRICE_DAYS) if t in prices else pd.Series(dtype=float)
        scores = []
        for d in run_dates:
            v = score_lookup.get((d, t))
            scores.append(None if v is None or pd.isna(v) else round(float(v), 1))
        out[t] = {
            "closes": [_sig(float(c)) for c in closes.tolist()],
            "close_from": closes.index[0].strftime("%Y-%m-%d") if len(closes) else None,
            "close_to": closes.index[-1].strftime("%Y-%m-%d") if len(closes) else None,
            "scores": scores,
            "headlines": heads_by_ticker.get(t, []),
        }
    return {"score_dates": run_dates, "tickers": out}


def _currency_for(ticker: str) -> str:
    """
    Cheap heuristic instead of an extra API call per ticker: this project
    only scans US-listed names and Nasdaq Stockholm (.ST suffix) right now,
    and those trade in USD and SEK respectively. If another exchange gets
    added later, extend this mapping.
    """
    return "SEK" if ticker.endswith(".ST") else "USD"


def _market_for(ticker: str) -> str:
    return "Sweden" if ticker.endswith(".ST") else "US"


def build_data_json(
    scored: pd.DataFrame,
    universe: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    model_meta: dict,
    backtests: list[BacktestResult],
    score_change: dict[str, float] | None = None,
    *,
    details: dict | None = None,
    followup: dict | None = None,
    data_quality: dict | None = None,
    weights: dict | None = None,
) -> dict:
    universe_idx = universe.set_index("ticker")
    score_change = score_change or {}
    weights = weights or {
        "pattern": config.WEIGHT_PATTERN,
        "momentum": config.WEIGHT_MOMENTUM,
        "sentiment": config.WEIGHT_SENTIMENT,
    }

    rows = []
    for ticker, r in scored.iterrows():
        info = universe_idx.loc[ticker] if ticker in universe_idx.index else None
        close_series = prices[ticker]["Close"] if ticker in prices else pd.Series(dtype=float)
        # A non-empty series can still have a NaN *last* value (a halted,
        # delisted, or just-listed ticker with a gap in its price history --
        # this happened for real on the live site: Yahoo returned a NaN
        # close for one ticker, json.dump() wrote it out as the bare token
        # `NaN`, which isn't valid JSON, and JS's JSON.parse() rejected the
        # ENTIRE file for every ticker, not just that one row). Treat it the
        # same as "no price data" -- None becomes JSON null, which the
        # dashboard already renders as "--" for every other numeric field.
        last_close = None
        if not close_series.empty:
            candidate = close_series.iloc[-1]
            if pd.notna(candidate):
                last_close = float(candidate)
        rows.append(
            {
                "ticker": ticker,
                "name": str(info["name"]) if info is not None else ticker,
                "sector": str(info["sector"]) if info is not None else "Unknown",
                "index": str(info["index"]) if info is not None else "",
                "price": last_close,
                "currency": _currency_for(ticker),
                "market": _market_for(ticker),
                "mom_5d": round(float(r.get("mom_5d", float("nan"))) * 100, 2) if pd.notna(r.get("mom_5d")) else None,
                "mom_20d": round(float(r.get("mom_20d", float("nan"))) * 100, 2) if pd.notna(r.get("mom_20d")) else None,
                "mom_60d": round(float(r.get("mom_60d", float("nan"))) * 100, 2) if pd.notna(r.get("mom_60d")) else None,
                "rel_volume": round(float(r["rel_volume"]), 2) if pd.notna(r.get("rel_volume")) else None,
                "sentiment_score": round(float(r["sentiment_score"]), 1) if pd.notna(r.get("sentiment_score")) else None,
                "headline_count": int(r["headline_count"]) if pd.notna(r.get("headline_count")) else 0,
                "pattern_score": round(float(r["pattern_score"]), 1) if pd.notna(r.get("pattern_score")) else None,
                "momentum_score": round(float(r["momentum_score"]), 1) if pd.notna(r.get("momentum_score")) else None,
                "trend_score": round(float(r["trend_score"]), 1) if pd.notna(r.get("trend_score")) else None,
                # the model's calibrated chance (%) of a +BREAKOUT_RETURN move within BREAKOUT_WINDOW days
                "breakout_prob": _num(r.get("breakout_prob"), 1),
                # 20-day change minus the same change in the ticker's market index (percentage points)
                "excess_20d": _num(r.get("excess_20d"), 2, scale=100),
                "avg_turnover": _num(r.get("avg_turnover"), 0),
                # None means "no prior-day score to compare against" (new
                # ticker, or the scanner's first run) -- the dashboard shows
                # that as "new" rather than a fabricated 0.0 change.
                "score_change": score_change.get(ticker),
                "tags": r.get("tags", []),
                "sparkline": _sparkline(close_series),
            }
        )

    backtest_payload = [
        {
            "horizon": b.horizon_label,
            "ready": b.ready,
            "n_pairs": b.n_pairs,
            "decile_avg_return": b.decile_avg_return,
            "top_vs_bottom_spread": b.top_vs_bottom_spread,
        }
        for b in backtests
    ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "universe_size": len(scored),
        "movers_top_n": config.MOVERS_TOP_N,
        "earnings_check_top_n": config.EARNINGS_CHECK_TOP_N,
        "telegram_score_threshold": config.TELEGRAM_SCORE_THRESHOLD,
        "telegram_reset_threshold": config.TELEGRAM_RESET_THRESHOLD,
        "weights": weights,
        "momentum_sector_weight": config.MOMENTUM_SECTOR_WEIGHT,
        "min_avg_turnover": config.MIN_AVG_TURNOVER,
        "benchmark_names": config.BENCHMARK_NAMES,
        "watchlist": list(config.WATCHLIST),
        "detail_top_n": config.DETAIL_TOP_N,
        "data_quality": data_quality or {},
        "followup": followup or {},
        "details": details or {},
        "breakout_definition": {
            "return": config.BREAKOUT_RETURN,
            "window_trading_days": config.BREAKOUT_WINDOW,
        },
        "model": model_meta,
        "backtests": backtest_payload,
        "rows": rows,
    }


def write_data_json(payload: dict) -> None:
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.DOCS_DIR / "data.json", "w") as f:
        # allow_nan=False is deliberate, not a default we forgot to
        # override: Python's json module otherwise happily writes NaN /
        # Infinity as bare (invalid) JSON tokens instead of raising, which
        # is exactly how one bad ticker price took down the entire live
        # dashboard (see the price-field fix above). With this set, ANY
        # stray NaN/Infinity anywhere in the payload -- this field or a
        # future one -- fails the pipeline run loudly right here, in the
        # Actions log, instead of silently shipping a data.json that
        # JSON.parse() rejects wholesale days later with no error anywhere.
        json.dump(payload, f, separators=(",", ":"), allow_nan=False)


def ensure_html_shell() -> None:
    """index.html is a static shell shipped with the repo (see docs/index.html
    in this project) -- this function only exists so run_pipeline can call it
    unconditionally without caring whether it's the first run or the 500th."""
    index_path = config.DOCS_DIR / "index.html"
    if not index_path.exists():
        raise FileNotFoundError(
            "docs/index.html is missing. It should ship with the repo -- "
            "see the project README for how to restore it."
        )
