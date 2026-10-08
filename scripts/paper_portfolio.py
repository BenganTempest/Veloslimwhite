"""
Paper portfolio: would following the scanner mechanically have beaten
simply owning the index, AFTER trading costs?

Rules (all in config.py):
  * every PAPER_REBALANCE_EVERY scanner runs, hold the top PAPER_TOP_N
    tickers by Trend Score, equal-weighted
  * between rebalances, positions drift with their prices
  * every trade pays PAPER_COST_BPS (brokerage + half spread) per side
  * the comparison is the same US/Sweden mix of index funds (S&P 500 /
    OMXS30), re-weighted at each rebalance to match the portfolio's split,
    with no costs -- i.e. the cheap alternative the strategy has to beat

Computed only from data/history.csv (daily closes + scores of every scanned
ticker) and data/benchmarks.csv -- no broker, no money, no paid data.

Known simplifications (shown on the dashboard): trades at the scan's
closing prices, no currency effect between USD and SEK, no dividends in
either the portfolio or the index (Yahoo's adjusted closes include
dividends for stocks but the index level doesn't, which slightly favours
the portfolio), and a ticker that stops being scanned is held at its last
known price until the next rebalance.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from followup import market_for, with_version, _bench_series, _asof  # noqa: E402

ANNUALIZE = 252  # one scanner run per trading day


def _cost_rate(ticker: str) -> float:
    return config.PAPER_COST_BPS.get(market_for(ticker), 0) / 10_000


def simulate(history: pd.DataFrame, benchmarks: pd.DataFrame, start_date: str | None = None) -> dict | None:
    """
    Runs the strategy over history.csv (optionally from start_date).
    Returns None if there are fewer than two run dates to work with.
    """
    if history is None or history.empty:
        return None
    h = with_version(history).dropna(subset=["close"])
    h["date"] = h["date"].astype(str)
    if start_date:
        h = h[h["date"] >= start_date]
    dates = sorted(h["date"].unique())
    if len(dates) < 2:
        return None

    closes = h.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").reindex(dates).ffill()
    scores = h.pivot_table(index="date", columns="ticker", values="trend_score", aggfunc="last").reindex(dates)
    version_by_date = h.groupby("date")["model_version"].agg(lambda s: s.mode().iloc[0])
    bench = _bench_series(benchmarks)
    bench_lvl = {m: [_asof(bench.get(m), pd.Timestamp(d)) for d in dates] for m in ("US", "Sweden")}

    n, every = config.PAPER_TOP_N, max(1, config.PAPER_REBALANCE_EVERY)
    value, bvalue = 1.0, 1.0
    w: dict[str, float] = {}
    bw: dict[str, float] = {}
    entry: dict[str, float] = {}
    curve, port_r, bench_r = [], [], []
    period_p, period_b = [], []          # returns per rebalance period
    p_start, b_start = 1.0, 1.0
    total_cost, turnovers, n_rebal = 0.0, [], 0

    for i, d in enumerate(dates):
        if i > 0:
            prev = dates[i - 1]
            rets = {}
            for t in w:
                c0, c1 = closes.at[prev, t], closes.at[d, t]
                rets[t] = (c1 / c0 - 1) if (pd.notna(c0) and pd.notna(c1) and c0) else 0.0
            rp = sum(w[t] * rets[t] for t in w)
            value *= 1 + rp
            if 1 + rp:
                w = {t: w[t] * (1 + rets[t]) / (1 + rp) for t in w}
            rb = 0.0
            new_bw = {}
            for m, wt in bw.items():
                b0, b1 = bench_lvl[m][i - 1], bench_lvl[m][i]
                r = (b1 / b0 - 1) if (b0 and b1 and not math.isnan(b0) and not math.isnan(b1)) else 0.0
                rb += wt * r
                new_bw[m] = wt * (1 + r)
            bvalue *= 1 + rb
            tot = sum(new_bw.values())
            bw = {m: v / tot for m, v in new_bw.items()} if tot else bw
            port_r.append(rp)
            bench_r.append(rb)

        rebalance = i % every == 0 and i < len(dates) - 1 or (i == 0)
        if rebalance:
            day_scores = scores.loc[d].dropna()
            day_scores = day_scores[closes.loc[d, day_scores.index].notna()]
            picks = list(day_scores.sort_values(ascending=False).head(n).index)
            if picks:
                if n_rebal > 0:
                    period_p.append(value / p_start - 1)
                    period_b.append(bvalue / b_start - 1)
                target = {t: 1.0 / len(picks) for t in picks}
                tickers = set(target) | set(w)
                trades = {t: abs(target.get(t, 0.0) - w.get(t, 0.0)) for t in tickers}
                cost = sum(trades[t] * _cost_rate(t) for t in tickers)
                value *= 1 - cost
                total_cost += cost
                turnovers.append(sum(trades.values()) / 2)
                for t in picks:
                    if t not in w:
                        entry[t] = float(closes.at[d, t])
                entry = {t: entry[t] for t in picks if t in entry}
                w = target
                us = sum(v for t, v in w.items() if market_for(t) == "US")
                bw = {m: x for m, x in (("US", us), ("Sweden", 1 - us)) if x > 0}
                p_start, b_start = value, bvalue
                n_rebal += 1

        curve.append({"date": d, "port": round(value, 5), "bench": round(bvalue, 5),
                      "version": str(version_by_date.get(d, config.LEGACY_MODEL_VERSION)), "rebalance": bool(rebalance)})

    # the still-open period counts once it has run at least half its length
    runs_in_open = (len(dates) - 1) % every
    if n_rebal and runs_in_open >= every / 2:
        period_p.append(value / p_start - 1)
        period_b.append(bvalue / b_start - 1)

    last = dates[-1]
    holdings = []
    for t, wt in sorted(w.items(), key=lambda x: -x[1]):
        c = closes.at[last, t]
        e = entry.get(t)
        holdings.append({
            "ticker": t,
            "market": market_for(t),
            "weight": round(wt * 100, 1),
            "since_entry": _pct((c / e - 1) if (e and pd.notna(c)) else None),
        })

    def stats(r: list[float]) -> tuple[float | None, float | None]:
        if len(r) < 2:
            return None, None
        a = np.array(r)
        sd = a.std(ddof=1)
        vol = sd * math.sqrt(ANNUALIZE)
        sharpe = (a.mean() / sd * math.sqrt(ANNUALIZE)) if sd > 0 else None
        return vol, sharpe

    def max_dd(series: list[float]) -> float:
        peak, dd = -math.inf, 0.0
        for v in series:
            peak = max(peak, v)
            dd = min(dd, v / peak - 1)
        return dd

    p_vol, p_sharpe = stats(port_r)
    b_vol, b_sharpe = stats(bench_r)
    runs = len(dates) - 1
    beat = [p > b for p, b in zip(period_p, period_b)]
    return {
        "start": dates[0],
        "end": last,
        "runs": runs,
        "rebalances": n_rebal,
        "total_return": _pct(value - 1),
        "bench_return": _pct(bvalue - 1),
        "excess": _pct((value - 1) - (bvalue - 1)),
        "vol": _pct(p_vol), "bench_vol": _pct(b_vol),
        "sharpe": None if p_sharpe is None else round(p_sharpe, 2),
        "bench_sharpe": None if b_sharpe is None else round(b_sharpe, 2),
        "max_drawdown": _pct(max_dd([c["port"] for c in curve])),
        "bench_max_drawdown": _pct(max_dd([c["bench"] for c in curve])),
        "total_costs": _pct(total_cost),
        "avg_turnover": _pct(float(np.mean(turnovers[1:])) if len(turnovers) > 1 else None),
        "periods": len(beat),
        "hit_rate": _pct(sum(beat) / len(beat)) if beat else None,
        "holdings": holdings,
        "curve": curve,
    }


def validation_status(result: dict | None) -> dict:
    """
    The pre-agreed bar (config.PAPER_VALIDATION_DAYS / PAPER_MIN_HIT_RATE):
    after that many runs, the strategy must beat the index after costs and
    win at least PAPER_MIN_HIT_RATE % of rebalance periods. Until then it's
    "validating" -- and no conclusions should be drawn either way.
    """
    need = config.PAPER_VALIDATION_DAYS
    if not result:
        return {"state": "validating", "runs": 0, "required_runs": need}
    out = {"runs": result["runs"], "required_runs": need, "min_hit_rate": config.PAPER_MIN_HIT_RATE}
    if result["runs"] < need:
        out["state"] = "validating"
        return out
    ok = (result["excess"] or 0) > 0 and (result["hit_rate"] or 0) >= config.PAPER_MIN_HIT_RATE
    out["state"] = "passed" if ok else "failed"
    return out


def _pct(x) -> float | None:
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) or math.isinf(x) else round(x * 100, 2)


def build_paper_payload(history: pd.DataFrame, benchmarks: pd.DataFrame, current_version: str,
                        version_start: str | None) -> dict:
    """Full history and, separately, only since the current model version started."""
    full = simulate(history, benchmarks)
    current = simulate(history, benchmarks, start_date=version_start) if version_start else full
    version_changes = []
    if full:
        prev = None
        for c in full["curve"]:
            if c["version"] != prev:
                version_changes.append({"date": c["date"], "version": c["version"]})
                prev = c["version"]
    return {
        "top_n": config.PAPER_TOP_N,
        "rebalance_every": config.PAPER_REBALANCE_EVERY,
        "cost_bps": config.PAPER_COST_BPS,
        "current_version": current_version,
        "version_start": version_start,
        "full": full,
        "current": current,
        "version_changes": version_changes,
        "status": validation_status(current),
    }
