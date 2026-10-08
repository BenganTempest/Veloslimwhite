"""
Follow-up: did the scanner's picks actually do better than the market?

Two groups of "events" are tracked, both from files the pipeline already
commits every day:

  * alerts      -- every Telegram threshold-cross (data/alert_log.csv)
  * daily top N -- the config.FOLLOWUP_TOP_N highest scores of each run
                   (from data/history.csv), which produces data much sooner
                   than the much rarer alerts

For each event the stock's return N scanner runs later (config.
FOLLOWUP_HORIZONS, roughly trading days) is compared with its own market's
index over exactly the same span (S&P 500 for US tickers, OMXS30 for
Stockholm). "Excess" = stock return minus index return, in percentage
points; "hit rate" = share of events that beat the index.

Everything here is pure pandas on local CSVs -- no network -- so the daily
pipeline, the weekly summary and the tests can all call it.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

ALERT_LOG_COLUMNS = ["date", "ticker", "name", "score", "prev_score", "close", "model_version"]


def market_for(ticker: str) -> str:
    return "Sweden" if str(ticker).endswith(".ST") else "US"


# --- loading / saving ----------------------------------------------------------

def with_version(df: pd.DataFrame) -> pd.DataFrame:
    """Rows written before versioning existed get config.LEGACY_MODEL_VERSION."""
    df = df.copy()
    if "model_version" not in df.columns:
        df["model_version"] = config.LEGACY_MODEL_VERSION
    df["model_version"] = df["model_version"].fillna(config.LEGACY_MODEL_VERSION).astype(str)
    return df


def load_history() -> pd.DataFrame:
    if config.SNAPSHOT_HISTORY_CSV.exists():
        return with_version(pd.read_csv(config.SNAPSHOT_HISTORY_CSV))
    return pd.DataFrame(columns=["date", "ticker", "close", "trend_score", "model_version"])


def load_alert_log() -> pd.DataFrame:
    if config.ALERT_LOG_CSV.exists():
        return with_version(pd.read_csv(config.ALERT_LOG_CSV))
    return pd.DataFrame(columns=ALERT_LOG_COLUMNS)


def append_alert_log(date_str: str, crossers: list[dict], closes: dict[str, float],
                     model_version: str | None = None) -> None:
    if not crossers:
        return
    existing = load_alert_log()
    existing = existing[existing["date"] != date_str]  # a re-run on the same day replaces that day's rows
    new = pd.DataFrame([
        {
            "date": date_str,
            "ticker": c["ticker"],
            "name": c.get("name", c["ticker"]),
            "score": round(float(c["score"]), 1),
            "prev_score": round(float(c["prev"]), 1) if c.get("prev") is not None else None,
            "close": closes.get(c["ticker"]),
            "model_version": model_version or config.MODEL_VERSION,
        }
        for c in crossers
    ], columns=ALERT_LOG_COLUMNS)
    out = pd.concat([existing, new], ignore_index=True) if not existing.empty else new
    config.ALERT_LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.ALERT_LOG_CSV, index=False)


def load_benchmarks() -> pd.DataFrame:
    """Long format: date (trading day), market, close."""
    if config.BENCHMARK_HISTORY_CSV.exists():
        return pd.read_csv(config.BENCHMARK_HISTORY_CSV)
    return pd.DataFrame(columns=["date", "market", "close"])


def update_benchmark_history(bench_prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    bench_prices: {market: OHLCV DataFrame}. Merges the freshly downloaded
    series (about 2 years each run) into data/benchmarks.csv, so the file
    keeps growing past the download window.
    """
    rows = []
    for market, df in bench_prices.items():
        closes = df["Close"].dropna()
        for d, c in closes.items():
            rows.append({"date": pd.Timestamp(d).strftime("%Y-%m-%d"), "market": market, "close": float(c)})
    fresh = pd.DataFrame(rows, columns=["date", "market", "close"])
    combined = pd.concat([load_benchmarks(), fresh], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "market"], keep="last").sort_values(["market", "date"])
    config.BENCHMARK_HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(config.BENCHMARK_HISTORY_CSV, index=False)
    return combined


def benchmark_mom(benchmarks: pd.DataFrame, days: int = 20) -> dict[str, float]:
    """Latest `days`-trading-day change of each market index (as a fraction)."""
    out = {}
    for market, g in benchmarks.groupby("market"):
        closes = g.sort_values("date")["close"].astype(float)
        if len(closes) > days:
            out[market] = float(closes.iloc[-1] / closes.iloc[-1 - days] - 1)
    return out


# --- core computation -------------------------------------------------------------

def _bench_series(benchmarks: pd.DataFrame) -> dict[str, pd.Series]:
    out = {}
    if benchmarks is None or benchmarks.empty:
        return out
    for market, g in benchmarks.groupby("market"):
        s = g.assign(date=pd.to_datetime(g["date"])).set_index("date")["close"].astype(float).sort_index()
        out[market] = s
    return out


def _asof(series: pd.Series | None, date: pd.Timestamp) -> float:
    """Last index close on or before `date` -- matches what the scanner could see when it ran."""
    if series is None or series.empty:
        return math.nan
    s = series.loc[:date]
    return float(s.iloc[-1]) if len(s) else math.nan


def event_returns(events: pd.DataFrame, history: pd.DataFrame, benchmarks: pd.DataFrame,
                  horizons: list[int] | None = None) -> pd.DataFrame:
    """
    events: columns date, ticker (date = a scanner run date in history.csv).
    Adds, per horizon h: ret_h, bench_h, excess_h (fractions, NaN when the
    horizon hasn't passed yet or a price is missing), plus ret_now /
    bench_now / excess_now / runs_since measured to the latest run.
    """
    horizons = horizons or config.FOLLOWUP_HORIZONS
    out = events.copy().reset_index(drop=True)
    if out.empty or history.empty:
        for h in horizons:
            out[f"ret_{h}"] = out[f"bench_{h}"] = out[f"excess_{h}"] = pd.Series(dtype=float)
        for k in ("ret_now", "bench_now", "excess_now", "runs_since"):
            out[k] = pd.Series(dtype=float)
        return out

    run_dates = sorted(history["date"].astype(str).unique())
    pos = {d: i for i, d in enumerate(run_dates)}
    close = {(str(d), t): c for d, t, c in zip(history["date"], history["ticker"], history["close"])}
    last_seen: dict[str, str] = {}
    for d, t in sorted(zip(history["date"].astype(str), history["ticker"])):
        last_seen[t] = d
    bench = _bench_series(benchmarks)

    def span(ticker: str, d0: str, d1: str) -> tuple[float, float]:
        c0, c1 = close.get((d0, ticker)), close.get((d1, ticker))
        r = (c1 / c0 - 1) if (c0 and c1 and pd.notna(c0) and pd.notna(c1)) else math.nan
        s = bench.get(market_for(ticker))
        b0, b1 = _asof(s, pd.Timestamp(d0)), _asof(s, pd.Timestamp(d1))
        br = (b1 / b0 - 1) if (b0 and b1 and not math.isnan(b0) and not math.isnan(b1)) else math.nan
        return r, br

    cols: dict[str, list] = {f"{k}_{h}": [] for h in horizons for k in ("ret", "bench", "excess")}
    for k in ("ret_now", "bench_now", "excess_now", "runs_since"):
        cols[k] = []
    for d, t in zip(out["date"].astype(str), out["ticker"]):
        i = pos.get(d)
        for h in horizons:
            if i is None or i + h >= len(run_dates):
                r = br = math.nan
            else:
                r, br = span(t, d, run_dates[i + h])
            cols[f"ret_{h}"].append(r)
            cols[f"bench_{h}"].append(br)
            cols[f"excess_{h}"].append(r - br if not (math.isnan(r) or math.isnan(br)) else math.nan)
        end = last_seen.get(t)
        if i is None or end is None or end <= d:
            r = br = math.nan
            runs = 0 if i is None else len(run_dates) - 1 - i
        else:
            r, br = span(t, d, end)
            runs = pos[end] - i
        cols["ret_now"].append(r)
        cols["bench_now"].append(br)
        cols["excess_now"].append(r - br if not (math.isnan(r) or math.isnan(br)) else math.nan)
        cols["runs_since"].append(runs)
    for k, v in cols.items():
        out[k] = v
    return out


def top_n_events(history: pd.DataFrame, n: int | None = None) -> pd.DataFrame:
    n = n or config.FOLLOWUP_TOP_N
    if history.empty:
        return pd.DataFrame(columns=["date", "ticker", "score", "model_version"])
    h = with_version(history).dropna(subset=["trend_score"])
    top = h.sort_values("trend_score", ascending=False).groupby("date", sort=False).head(n)
    return (top.rename(columns={"trend_score": "score"})[["date", "ticker", "score", "model_version"]]
            .sort_values("date"))


def summarize(ev: pd.DataFrame, horizons: list[int] | None = None) -> dict:
    horizons = horizons or config.FOLLOWUP_HORIZONS
    out = {}
    for h in horizons:
        col = f"excess_{h}"
        done = ev.dropna(subset=[col]) if col in ev.columns else ev.iloc[0:0]
        if done.empty:
            out[str(h)] = {"n": 0, "avg_return": None, "avg_excess": None, "hit_rate": None}
            continue
        out[str(h)] = {
            "n": int(len(done)),
            "avg_return": _pct(done[f"ret_{h}"].mean()),
            "avg_excess": _pct(done[col].mean()),
            "hit_rate": _pct((done[col] > 0).mean()),
        }
    return out


def _pct(x) -> float | None:
    """Fraction -> percent, rounded; NaN -> None (data.json is written with allow_nan=False)."""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else round(x * 100, 2)


def build_followup_payload(history: pd.DataFrame, alert_log: pd.DataFrame, benchmarks: pd.DataFrame,
                           recent_n: int = 30) -> dict:
    """
    Results for all versions together plus, under "by_version", separately
    per scoring-model version -- each event belongs to the version that
    produced the score it was based on.
    """
    horizons = config.FOLLOWUP_HORIZONS
    history = with_version(history)
    alert_log = with_version(alert_log) if not alert_log.empty else pd.DataFrame(columns=ALERT_LOG_COLUMNS)
    alerts_ev = event_returns(alert_log[["date", "ticker", "name", "score", "model_version"]],
                              history, benchmarks, horizons)
    top_ev = event_returns(top_n_events(history), history, benchmarks, horizons)

    recent = alerts_ev.sort_values("date", ascending=False).head(recent_n)
    recent_rows = []
    for _, r in recent.iterrows():
        row = {
            "date": str(r["date"]),
            "ticker": r["ticker"],
            "name": str(r.get("name", r["ticker"])),
            "market": market_for(r["ticker"]),
            "model_version": str(r.get("model_version", config.LEGACY_MODEL_VERSION)),
            "score": None if pd.isna(r.get("score")) else round(float(r["score"]), 1),
            "runs_since": int(r["runs_since"]) if pd.notna(r["runs_since"]) else 0,
            "ret_now": _pct(r["ret_now"]),
            "bench_now": _pct(r["bench_now"]),
            "excess_now": _pct(r["excess_now"]),
        }
        for h in horizons:
            row[f"excess_{h}"] = _pct(r[f"excess_{h}"])
        recent_rows.append(row)

    versions = sorted(set(history["model_version"]) | set(alert_log["model_version"].astype(str)))
    by_version = {}
    for v in versions:
        a_v = alerts_ev[alerts_ev["model_version"].astype(str) == v] if not alerts_ev.empty else alerts_ev
        t_v = top_ev[top_ev["model_version"].astype(str) == v] if not top_ev.empty else top_ev
        by_version[v] = {"n_alerts": int(len(a_v)), "alerts": summarize(a_v, horizons), "top": summarize(t_v, horizons)}

    return {
        "horizons": horizons,
        "top_n": config.FOLLOWUP_TOP_N,
        "benchmark_names": config.BENCHMARK_NAMES,
        "n_alerts": int(len(alert_log)),
        "alerts": summarize(alerts_ev, horizons),
        "top": summarize(top_ev, horizons),
        "by_version": by_version,
        "recent_alerts": recent_rows,
    }
