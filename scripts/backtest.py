"""
The honesty panel: uses the scanner's OWN accumulated daily snapshots
(data/history.csv) to check, in arrears, whether stocks it scored highly
actually went on to do better than stocks it scored low. This has nothing
to do with the historical pattern-model's own internal validation (that's
in pattern_model.py) -- this is checking the live, real-world output of
the whole pipeline (pattern + momentum + sentiment blended) against what
actually happened after the fact.

Returns "not enough data yet" until the snapshot history spans at least
one full horizon, which is the honest state for a freshly started scanner.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

HORIZONS = {"1 month": 21, "3 months": 63}


@dataclass
class BacktestResult:
    horizon_label: str
    ready: bool
    n_pairs: int
    decile_avg_return: dict[int, float] | None
    top_vs_bottom_spread: float | None


def _trading_day_index(dates: pd.Series) -> pd.Series:
    unique_sorted = pd.Series(sorted(dates.unique()))
    return pd.Series(range(len(unique_sorted)), index=unique_sorted)


def run_backtests(history: pd.DataFrame) -> list[BacktestResult]:
    results = []
    if history.empty:
        return [BacktestResult(label, False, 0, None, None) for label in HORIZONS]

    history = history.copy()
    history["date"] = pd.to_datetime(history["date"])
    day_rank = _trading_day_index(history["date"])
    history["day_idx"] = history["date"].map(day_rank)

    for label, horizon in HORIZONS.items():
        max_idx = history["day_idx"].max()
        if max_idx < horizon:
            results.append(BacktestResult(label, False, 0, None, None))
            continue

        early = history[history["day_idx"] <= max_idx - horizon]
        pairs = []
        for start_idx, group in early.groupby("day_idx"):
            target_idx = start_idx + horizon
            future = history[history["day_idx"] == target_idx][["ticker", "close"]].rename(
                columns={"close": "future_close"}
            )
            merged = group.merge(future, on="ticker", how="inner")
            if merged.empty:
                continue
            merged["fwd_return"] = merged["future_close"] / merged["close"] - 1
            merged["decile"] = pd.qcut(merged["trend_score"], 10, labels=False, duplicates="drop")
            pairs.append(merged[["decile", "fwd_return"]])

        if not pairs:
            results.append(BacktestResult(label, False, 0, None, None))
            continue

        all_pairs = pd.concat(pairs, ignore_index=True)
        decile_avg = all_pairs.groupby("decile")["fwd_return"].mean().round(4).to_dict()
        top = decile_avg.get(max(decile_avg), None)
        bottom = decile_avg.get(min(decile_avg), None)
        spread = (top - bottom) if (top is not None and bottom is not None) else None

        results.append(
            BacktestResult(
                horizon_label=label,
                ready=True,
                n_pairs=len(all_pairs),
                decile_avg_return=decile_avg,
                top_vs_bottom_spread=spread,
            )
        )

    return results
