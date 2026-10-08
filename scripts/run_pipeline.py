"""
Entry point run daily by GitHub Actions (see .github/workflows/daily.yml).
Orchestrates: universe -> prices/news -> features -> pattern model ->
composite score -> append to history.csv -> write dashboard data.json.

Designed to degrade gracefully: if news fetch fails entirely, sentiment
just goes neutral for that run instead of crashing; if the pattern model
can't train (too little history), pattern_score is dropped from the blend
and the remaining weights are renormalized, with that noted in the output
so the dashboard can say so honestly rather than silently substituting.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from universe import build_universe  # noqa: E402
from collect import fetch_prices, fetch_news  # noqa: E402
from features import compute_features, FEATURE_COLUMNS  # noqa: E402
import pattern_model  # noqa: E402
import score as scoring  # noqa: E402
from backtest import run_backtests  # noqa: E402
from build_dashboard import build_data_json, write_data_json, ensure_html_shell, build_details  # noqa: E402
from earnings import check_upcoming_earnings, tickers_with_earnings_soon  # noqa: E402
import notify  # noqa: E402
import followup  # noqa: E402
import versioning  # noqa: E402
import paper_portfolio  # noqa: E402
from features import avg_turnover  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("run_pipeline")


def _currency(ticker: str) -> str:
    return "SEK" if ticker.endswith(".ST") else "USD"


def apply_liquidity_filter(prices: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict[str, float], list[str]]:
    """Drops tickers below config.MIN_AVG_TURNOVER (watchlist tickers are always kept)."""
    turnover = {t: avg_turnover(df) for t, df in prices.items()}
    keep, dropped = {}, []
    watch = set(config.WATCHLIST)
    for t, df in prices.items():
        floor = config.MIN_AVG_TURNOVER.get(_currency(t), 0)
        if turnover[t] >= floor or t in watch:
            keep[t] = df
        else:
            dropped.append(t)
    log.info("Liquidity filter: kept %d, left out %d thinly traded ticker(s)", len(keep), len(dropped))
    return keep, turnover, dropped


def append_run_log(row: dict) -> None:
    path = config.RUN_LOG_CSV
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not existing.empty:
        existing = existing[existing["date"] != row["date"]]
    out = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def main() -> None:
    today_str = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    model_version, version_register = versioning.resolve_version(today_str)
    log.info("Scoring model version %s", model_version)

    universe = build_universe()
    # watchlist tickers are always scanned, even if they aren't in any index list
    extra = [t for t in config.WATCHLIST if t not in set(universe["ticker"])]
    if extra:
        universe = pd.concat(
            [universe, pd.DataFrame({"ticker": extra, "name": extra, "sector": "Unknown", "index": "Watchlist"})],
            ignore_index=True,
        )
    tickers = universe["ticker"].tolist()

    # build_universe() may have just made a burst of individual Yahoo
    # requests (per-ticker sector backfill for Stockholm names). Give its
    # short-term rate limit a moment to reset before the much larger bulk
    # price download right below -- observed to matter in practice.
    if config.PAUSE_AFTER_SECTOR_ENRICHMENT_SECONDS:
        log.info("Pausing %ds before the bulk price download...", config.PAUSE_AFTER_SECTOR_ENRICHMENT_SECONDS)
        time.sleep(config.PAUSE_AFTER_SECTOR_ENRICHMENT_SECONDS)

    raw_prices = fetch_prices(tickers)
    if not raw_prices:
        log.error("No price data at all -- aborting run without touching dashboard/history")
        sys.exit(1)
    missing = [t for t in tickers if t not in raw_prices]
    missing_pct = 100.0 * len(missing) / max(1, len(tickers))
    if missing_pct > config.DATA_QUALITY_WARN_MISSING_PCT:
        print(f"::warning title=Datakvalitet::{len(missing)} av {len(tickers)} aktier ({missing_pct:.1f} %) "
              f"kunde inte hämtas från Yahoo i denna körning.", flush=True)

    # --- market indexes: for market-relative momentum and the follow-up panel
    benchmarks = followup.load_benchmarks()
    try:
        bench_raw = fetch_prices(list(config.BENCHMARKS.values()))
        bench_prices = {m: bench_raw[sym] for m, sym in config.BENCHMARKS.items() if sym in bench_raw}
        if bench_prices:
            benchmarks = followup.update_benchmark_history(bench_prices)
        else:
            log.warning("Could not download any benchmark index this run -- using the stored history")
    except Exception as exc:  # noqa: BLE001
        log.warning("Benchmark download failed (non-fatal): %s", exc)
    bench_mom_20d = followup.benchmark_mom(benchmarks, 20)

    prices, turnover, illiquid = apply_liquidity_filter(raw_prices)

    news = fetch_news(universe[universe["ticker"].isin(prices.keys())])

    # --- features for every ticker's full history (training) + today (scoring)
    today_features = {}
    for t, df in prices.items():
        feats = compute_features(df)
        if feats.dropna(subset=FEATURE_COLUMNS).empty:
            continue
        today_features[t] = feats.dropna(subset=FEATURE_COLUMNS).iloc[-1]
    today_features_df = pd.DataFrame(today_features).T
    for c in FEATURE_COLUMNS:
        today_features_df[c] = pd.to_numeric(today_features_df[c], errors="coerce")

    panel = pattern_model.build_training_panel(prices)
    model = pattern_model.train(panel)

    weights = {
        "pattern": config.WEIGHT_PATTERN,
        "momentum": config.WEIGHT_MOMENTUM,
        "sentiment": config.WEIGHT_SENTIMENT,
    }
    model_meta = {"trained": False}

    def _r(x, d=3):
        return None if x is None or pd.isna(x) else round(float(x), d)

    if model is not None:
        pattern_df = pattern_model.score_today(model, today_features_df)
        model_meta = {
            "trained": True,
            "validation_auc": _r(model.validation_auc),
            "validation_precision_at_top_decile": _r(model.validation_precision_at_top_decile),
            "n_train_rows": model.n_train_rows,
            "n_positive_train": model.n_positive_train,
            "n_validation_rows": model.n_validation_rows,
            "base_rate": _r(model.base_rate, 4),
            "val_mean_pred": _r(model.val_mean_pred, 4),
            "val_actual_rate": _r(model.val_actual_rate, 4),
            "val_top_decile_mean_pred": _r(model.val_top_decile_mean_pred, 4),
        }
    else:
        log.warning("Pattern model could not be trained this run -- falling back to momentum+sentiment only")
        pattern_df = pd.DataFrame(columns=["pattern_score", "breakout_prob"], dtype=float)
        total = weights["momentum"] + weights["sentiment"]
        weights = {"pattern": 0.0, "momentum": weights["momentum"] / total, "sentiment": weights["sentiment"] / total}

    sectors = universe.drop_duplicates("ticker").set_index("ticker")["sector"]
    momentum_scores = scoring.score_momentum(today_features_df, sectors)
    sentiment_df = scoring.score_sentiment(news, list(today_features_df.index))

    scored = scoring.composite_score(
        pattern_df["pattern_score"], momentum_scores, sentiment_df["sentiment_score"], weights
    )
    scored = scored.join(pattern_df[["breakout_prob"]], how="left")
    scored = scored.join(today_features_df[FEATURE_COLUMNS], how="left")
    scored = scored.join(sentiment_df[["sentiment_compound", "headline_count"]], how="left")
    scored = scored.join(scoring.excess_vs_benchmark(today_features_df, bench_mom_20d), how="left")
    scored["avg_turnover"] = pd.Series(turnover).reindex(scored.index)
    scored["tags"] = scored.apply(lambda r: scoring.build_tags(r), axis=1)
    scored = scored.sort_values("trend_score", ascending=False)

    log.info("Scored %d tickers. Top 5: %s", len(scored), list(scored.head(5).index))

    # --- earnings-date awareness, scoped to just the top scorers (see
    # config.py / earnings.py for why this can't safely run for everyone)
    if config.CHECK_EARNINGS_DATES:
        if config.PAUSE_BEFORE_EARNINGS_CHECK_SECONDS:
            log.info("Pausing %ds before the earnings-date check...", config.PAUSE_BEFORE_EARNINGS_CHECK_SECONDS)
            time.sleep(config.PAUSE_BEFORE_EARNINGS_CHECK_SECONDS)
        top_tickers = list(scored.head(config.EARNINGS_CHECK_TOP_N).index)
        earnings_by_ticker = check_upcoming_earnings(top_tickers)
        soon = tickers_with_earnings_soon(earnings_by_ticker)
        if soon:
            log.info("%d ticker(s) have earnings within %d days", len(soon), config.EARNINGS_SOON_DAYS)
            for ticker in soon:
                scored.at[ticker, "tags"] = list(scored.at[ticker, "tags"]) + ["earnings soon"]

    # --- read YESTERDAY's scores (if any) before we overwrite history.csv
    # with today's snapshot -- powers the "movers" leaderboard, the
    # Telegram alert and the follow-up panel.
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    prev_scores: dict[str, float] = {}
    if config.SNAPSHOT_HISTORY_CSV.exists():
        existing_history = followup.with_version(pd.read_csv(config.SNAPSHOT_HISTORY_CSV))
        existing_history = existing_history[existing_history["date"] != today_str]
        if not existing_history.empty:
            last_prev_date = existing_history["date"].max()
            prev_rows = existing_history[existing_history["date"] == last_prev_date]
            prev_scores = dict(zip(prev_rows["ticker"], prev_rows["trend_score"]))
    else:
        existing_history = pd.DataFrame(columns=["date", "ticker", "close", "trend_score", "model_version"])

    score_change: dict[str, float] = {}
    for ticker in scored.index:
        prev = prev_scores.get(ticker)
        if prev is not None and pd.notna(prev):
            score_change[ticker] = round(float(scored.at[ticker, "trend_score"]) - float(prev), 1)

    # --- today's snapshot (also gives the closes the alert log needs)
    snapshot_rows = []
    closes: dict[str, float] = {}
    for ticker, r in scored.iterrows():
        close_series = prices[ticker]["Close"] if ticker in prices else pd.Series(dtype=float)
        close = close_series.iloc[-1] if not close_series.empty else None
        # A halted/delisted/gapped ticker can have a non-empty price series
        # whose LAST value is still NaN -- `close is None` alone doesn't
        # catch that (NaN is not None), and writing a NaN into history.csv
        # would silently corrupt the honesty log's backtest math later.
        if close is None or pd.isna(close):
            continue
        closes[ticker] = float(close)
        snapshot_rows.append(
            {"date": today_str, "ticker": ticker, "close": float(close), "trend_score": float(r["trend_score"]),
             "model_version": model_version}
        )

    # --- Telegram alert for new threshold crossers + big watchlist moves
    universe_idx = universe.drop_duplicates("ticker").set_index("ticker")
    try:
        crossers = notify.maybe_send_threshold_alert(scored, prev_scores, universe_idx)
        followup.append_alert_log(today_str, crossers, closes, model_version)
    except Exception as exc:  # noqa: BLE001
        log.warning("Telegram alert step failed (non-fatal): %s", exc)

    snapshot_df = pd.DataFrame(snapshot_rows)
    history = pd.concat([existing_history, snapshot_df], ignore_index=True)
    history.to_csv(config.SNAPSHOT_HISTORY_CSV, index=False)

    data_quality = {
        "universe_size": len(tickers),
        "priced": len(raw_prices),
        "missing": len(missing),
        "missing_pct": round(missing_pct, 2),
        "missing_sample": missing[:20],
        "illiquid_excluded": len(illiquid),
        "scored": int(len(scored)),
        "warn_missing_pct": config.DATA_QUALITY_WARN_MISSING_PCT,
        "model_trained": model is not None,
        "benchmarks_ok": sorted(bench_mom_20d.keys()),
    }
    append_run_log({
        "date": today_str,
        "universe_size": len(tickers),
        "priced": len(raw_prices),
        "missing": len(missing),
        "missing_pct": round(missing_pct, 2),
        "illiquid_excluded": len(illiquid),
        "scored": int(len(scored)),
        "model_trained": model is not None,
        "model_version": model_version,
        "alerts": len(followup.load_alert_log().query("date == @today_str")),
    })

    backtests = run_backtests(history)
    followup_payload = followup.build_followup_payload(history, followup.load_alert_log(), benchmarks)
    version_start = next((v["first_date"] for v in version_register if v["version"] == model_version), today_str)
    try:
        paper = paper_portfolio.build_paper_payload(history, benchmarks, model_version, version_start)
    except Exception as exc:  # noqa: BLE001 -- a bookkeeping bug must never stop the daily scan
        log.warning("Paper portfolio failed (non-fatal): %s", exc)
        paper = {}
    model_versions = versioning.with_legacy(version_register, sorted(set(history["model_version"].astype(str))))

    detail_tickers = list(scored.head(config.DETAIL_TOP_N).index)
    detail_tickers += [t for t in config.WATCHLIST if t in scored.index and t not in detail_tickers]
    details = build_details(detail_tickers, prices, history, news)

    payload = build_data_json(
        scored, universe, prices, model_meta, backtests, score_change,
        details=details, followup=followup_payload, data_quality=data_quality, weights=weights,
        paper=paper, model_version=model_version, model_versions=model_versions,
    )
    ensure_html_shell()
    write_data_json(payload)

    log.info("Pipeline complete. Wrote docs/data.json and appended history.csv (%d total snapshot rows).", len(history))


if __name__ == "__main__":
    main()
