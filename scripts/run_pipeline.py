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
from build_dashboard import build_data_json, write_data_json, ensure_html_shell  # noqa: E402
from earnings import check_upcoming_earnings, tickers_with_earnings_soon  # noqa: E402
import notify  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("run_pipeline")


def main() -> None:
    universe = build_universe()
    tickers = universe["ticker"].tolist()

    # build_universe() may have just made a burst of individual Yahoo
    # requests (per-ticker sector backfill for Stockholm names). Give its
    # short-term rate limit a moment to reset before the much larger bulk
    # price download right below -- observed to matter in practice.
    if config.PAUSE_AFTER_SECTOR_ENRICHMENT_SECONDS:
        log.info("Pausing %ds before the bulk price download...", config.PAUSE_AFTER_SECTOR_ENRICHMENT_SECONDS)
        time.sleep(config.PAUSE_AFTER_SECTOR_ENRICHMENT_SECONDS)

    prices = fetch_prices(tickers)
    if not prices:
        log.error("No price data at all -- aborting run without touching dashboard/history")
        sys.exit(1)

    news = fetch_news(universe[universe["ticker"].isin(prices.keys())])

    # --- features for every ticker's full history (training) + today (scoring)
    today_features = {}
    for t, df in prices.items():
        feats = compute_features(df)
        if feats.dropna(subset=FEATURE_COLUMNS).empty:
            continue
        today_features[t] = feats.dropna(subset=FEATURE_COLUMNS).iloc[-1]
    today_features_df = pd.DataFrame(today_features).T

    panel = pattern_model.build_training_panel(prices)
    model = pattern_model.train(panel)

    weight_pattern = config.WEIGHT_PATTERN
    weight_momentum = config.WEIGHT_MOMENTUM
    weight_sentiment = config.WEIGHT_SENTIMENT
    model_meta = {"trained": False}

    if model is not None:
        pattern_scores = pattern_model.score_today(model, today_features_df)
        model_meta = {
            "trained": True,
            "validation_auc": None if pd.isna(model.validation_auc) else round(model.validation_auc, 3),
            "validation_precision_at_top_decile": (
                None if pd.isna(model.validation_precision_at_top_decile)
                else round(model.validation_precision_at_top_decile, 3)
            ),
            "n_train_rows": model.n_train_rows,
            "n_positive_train": model.n_positive_train,
            "n_validation_rows": model.n_validation_rows,
        }
    else:
        log.warning("Pattern model could not be trained this run -- falling back to momentum+sentiment only")
        pattern_scores = pd.Series(dtype=float)
        total = weight_momentum + weight_sentiment
        weight_momentum, weight_sentiment = weight_momentum / total, weight_sentiment / total
        weight_pattern = 0.0
        config.WEIGHT_PATTERN, config.WEIGHT_MOMENTUM, config.WEIGHT_SENTIMENT = (
            weight_pattern, weight_momentum, weight_sentiment,
        )

    momentum_scores = scoring.score_momentum(today_features_df)
    sentiment_df = scoring.score_sentiment(news, list(today_features_df.index))

    scored = scoring.composite_score(pattern_scores, momentum_scores, sentiment_df["sentiment_score"])
    scored = scored.join(today_features_df[FEATURE_COLUMNS], how="left")
    scored = scored.join(sentiment_df[["sentiment_compound", "headline_count"]], how="left")
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
    # with today's snapshot -- powers both the "movers" leaderboard and the
    # Telegram threshold-cross alert, at no extra data-collection cost since
    # this is the same history.csv already kept for the honesty panel.
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    today_str = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    prev_scores: dict[str, float] = {}
    if config.SNAPSHOT_HISTORY_CSV.exists():
        existing_history = pd.read_csv(config.SNAPSHOT_HISTORY_CSV)
        existing_history = existing_history[existing_history["date"] != today_str]
        if not existing_history.empty:
            last_prev_date = existing_history["date"].max()
            prev_rows = existing_history[existing_history["date"] == last_prev_date]
            prev_scores = dict(zip(prev_rows["ticker"], prev_rows["trend_score"]))
    else:
        existing_history = pd.DataFrame(columns=["date", "ticker", "close", "trend_score"])

    score_change: dict[str, float] = {}
    for ticker in scored.index:
        prev = prev_scores.get(ticker)
        if prev is not None:
            score_change[ticker] = round(float(scored.at[ticker, "trend_score"]) - float(prev), 1)

    # --- Telegram alert for new threshold crossers (silently skipped if
    # TELEGRAM_ALERTS_ENABLED is off or the secrets aren't configured)
    universe_idx = universe.set_index("ticker")
    try:
        notify.maybe_send_threshold_alert(scored, prev_scores, universe_idx)
    except Exception as exc:  # noqa: BLE001
        log.warning("Telegram alert step failed (non-fatal): %s", exc)

    # --- append today's snapshot to the accumulating honesty log
    snapshot_rows = []
    for ticker, r in scored.iterrows():
        close = prices[ticker]["Close"].iloc[-1] if ticker in prices else None
        if close is None:
            continue
        snapshot_rows.append(
            {"date": today_str, "ticker": ticker, "close": float(close), "trend_score": float(r["trend_score"])}
        )
    snapshot_df = pd.DataFrame(snapshot_rows)
    history = pd.concat([existing_history, snapshot_df], ignore_index=True)
    history.to_csv(config.SNAPSHOT_HISTORY_CSV, index=False)

    backtests = run_backtests(history)

    payload = build_data_json(scored, universe, prices, model_meta, backtests, score_change)
    ensure_html_shell()
    write_data_json(payload)

    log.info("Pipeline complete. Wrote docs/data.json and appended history.csv (%d total snapshot rows).", len(history))


if __name__ == "__main__":
    main()
