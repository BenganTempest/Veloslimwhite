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
) -> dict:
    universe_idx = universe.set_index("ticker")

    rows = []
    for ticker, r in scored.iterrows():
        info = universe_idx.loc[ticker] if ticker in universe_idx.index else None
        close_series = prices[ticker]["Close"] if ticker in prices else pd.Series(dtype=float)
        last_close = float(close_series.iloc[-1]) if not close_series.empty else None
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
        "weights": {
            "pattern": config.WEIGHT_PATTERN,
            "momentum": config.WEIGHT_MOMENTUM,
            "sentiment": config.WEIGHT_SENTIMENT,
        },
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
        json.dump(payload, f, separators=(",", ":"))


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
