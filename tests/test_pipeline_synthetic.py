"""
End-to-end smoke test using synthetic OHLCV data, because this development
sandbox's network policy blocks both the real data hosts (Yahoo Finance)
and installing the yfinance/feedparser/vaderSentiment packages from PyPI --
see the project README for why. Everything exercised here uses only
numpy/pandas/scikit-learn, which ARE available, so this validates the real
feature engineering, pattern-model training, scoring, and dashboard-JSON
logic without needing live data. Run on GitHub Actions with the real
dependencies installed, the exact same functions run against real data.

Run: python3 tests/test_pipeline_synthetic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from features import compute_features, FEATURE_COLUMNS  # noqa: E402
import pattern_model  # noqa: E402
import score as scoring  # noqa: E402
from backtest import run_backtests  # noqa: E402
from build_dashboard import build_data_json, write_data_json  # noqa: E402
from earnings import check_upcoming_earnings, tickers_with_earnings_soon  # noqa: E402
import notify  # noqa: E402
import staleness_check  # noqa: E402
import paper_portfolio  # noqa: E402
import versioning  # noqa: E402
import followup  # noqa: E402

rng = np.random.default_rng(42)


def make_synthetic_ohlcv(n_days=500, drift=0.0003, vol=0.02, breakout_at=None, seed=0) -> pd.DataFrame:
    local_rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    rets = local_rng.normal(drift, vol, n_days)
    if breakout_at is not None:
        # inject a genuine run-up starting at breakout_at so the "positive"
        # class actually exists in the synthetic training data
        rets[breakout_at:breakout_at + 10] += 0.03
    close = 50 * np.cumprod(1 + rets)
    high = close * (1 + local_rng.uniform(0, 0.01, n_days))
    low = close * (1 - local_rng.uniform(0, 0.01, n_days))
    open_ = close * (1 + local_rng.normal(0, 0.002, n_days))
    volume = local_rng.integers(1_000_000, 5_000_000, n_days).astype(float)
    if breakout_at is not None:
        volume[breakout_at:breakout_at + 10] *= 3  # volume spike alongside the breakout
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates
    )


def build_synthetic_universe(n=40, n_swedish=6) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    sectors = ["Technology", "Healthcare", "Energy", "Financials", "Industrials"]
    rows = []
    prices = {}
    for i in range(n):
        ticker = f"SYN{i:03d}"
        rows.append({"ticker": ticker, "name": f"Synthetic Co {i}", "sector": sectors[i % len(sectors)], "index": "SYN"})
        # every 5th ticker gets an engineered breakout ~40 trading days ago so
        # the pattern model has positive examples to learn from and score against
        breakout_at = 460 if i % 5 == 0 else None
        prices[ticker] = make_synthetic_ohlcv(breakout_at=breakout_at, seed=i)
    # a handful of .ST tickers to exercise the SEK/USD currency-labeling path
    for i in range(n_swedish):
        ticker = f"SYNSE{i}.ST"
        rows.append({"ticker": ticker, "name": f"Svenska Bolag {i}", "sector": "Unknown", "index": "OMX Stockholm"})
        breakout_at = 460 if i == 0 else None
        prices[ticker] = make_synthetic_ohlcv(breakout_at=breakout_at, seed=1000 + i)
    return pd.DataFrame(rows), prices


def run():
    print("1. Building synthetic universe + prices...")
    universe, prices = build_synthetic_universe()
    assert len(universe) == len(prices) == 46

    # Regression case for a real bug found on the live site: a ticker whose
    # latest Close came back NaN from Yahoo (halted/delisted/just-listed --
    # a non-empty price series with a NaN *last* value, which `close is
    # None` alone does not catch). Unguarded, this produced a literal `NaN`
    # token in data.json, which is not valid JSON -- JS's JSON.parse()
    # rejected the ENTIRE file, not just that one ticker, taking down the
    # whole dashboard. NAN_PRICE_TICKER is asserted on explicitly below.
    NAN_PRICE_TICKER = "SYN001"
    nan_idx = prices[NAN_PRICE_TICKER].index[-1]
    prices[NAN_PRICE_TICKER].at[nan_idx, "Close"] = float("nan")

    print("2. compute_features on every ticker...")
    today_features = {}
    for t, df in prices.items():
        feats = compute_features(df)
        assert set(FEATURE_COLUMNS).issubset(feats.columns)
        clean = feats.dropna(subset=FEATURE_COLUMNS)
        assert not clean.empty, f"{t} produced no usable feature rows"
        today_features[t] = clean.iloc[-1]
    today_features_df = pd.DataFrame(today_features).T
    print(f"   OK -- {len(today_features_df)} tickers have usable features")

    print("3. build_training_panel + train (pattern model)...")
    panel = pattern_model.build_training_panel(prices)
    assert not panel.empty
    assert panel["label"].sum() > 0, "synthetic data has no positive breakout examples -- test data bug"
    model = pattern_model.train(panel)
    assert model is not None, "model failed to train on synthetic data"
    print(f"   OK -- trained on {model.n_train_rows} rows ({model.n_positive_train} positive), "
          f"val AUC={model.validation_auc:.3f}" if not np.isnan(model.validation_auc) else "   OK (val AUC n/a, small sample)")

    pattern_df = pattern_model.score_today(model, today_features_df)
    pattern_scores = pattern_df["pattern_score"]
    assert not pattern_scores.empty
    assert pattern_scores.between(0, 100).all()
    assert pattern_df["breakout_prob"].between(0, 100).all()
    assert 0 < model.base_rate < 1
    # prior correction must lower probabilities when positives are rarer than 50%
    # and must not change the ranking
    raw = model.clf.predict_proba(model.scaler.transform(today_features_df[FEATURE_COLUMNS].values.astype(float)))[:, 1]
    if model.base_rate < 0.5:
        assert (pattern_df["breakout_prob"].values <= raw * 100 + 1e-9).all()
    assert pd.Series(raw).rank().tolist() == pattern_df["breakout_prob"].rank().tolist()
    print(f"   OK -- scored {len(pattern_scores)} tickers, calibrated chance range "
          f"[{pattern_df['breakout_prob'].min():.2f}%, {pattern_df['breakout_prob'].max():.2f}%], base rate {model.base_rate:.3f}")

    print("4. score_momentum (relative to market and sector)...")
    sectors = universe.set_index("ticker")["sector"]
    momentum_scores = scoring.score_momentum(today_features_df, sectors)
    assert momentum_scores.between(0, 100).all()
    # market-relative: the best Swedish ticker must rank at the top of its own market
    # even if every US ticker had higher raw momentum
    shifted = today_features_df.copy()
    for c in ("mom_5d", "mom_20d", "mom_60d"):
        shifted.loc[~shifted.index.str.endswith(".ST"), c] += 5.0  # huge US-only rally
    m2 = scoring.score_momentum(shifted)
    se = m2[m2.index.str.endswith(".ST")]
    assert se.max() == 100.0, "top Swedish ticker should rank 100 within its own market"
    print("   OK -- ranks within market; a US-only rally doesn't push Swedish names down")

    print("4b. composite_score with no pattern model (bug fix: used to drop every row)...")
    empty_pattern = pd.Series(dtype=float)
    fallback = scoring.composite_score(empty_pattern, momentum_scores, pd.Series(dtype=float),
                                       {"pattern": 0.0, "momentum": 0.555, "sentiment": 0.445})
    assert len(fallback) == len(momentum_scores), "no-model fallback must still score every ticker"
    print(f"   OK -- {len(fallback)} tickers still scored without the pattern model")

    print("5. sentiment (stubbed -- vaderSentiment isn't installable in this sandbox, see README)...")
    fake_news = pd.DataFrame(columns=["ticker", "title", "published", "link"])  # empty -> neutral path
    sentiment_df = scoring.score_sentiment(fake_news, list(today_features_df.index))
    assert (sentiment_df["sentiment_score"] == 50.0).all()
    print("   OK -- neutral-sentiment fallback path verified")

    print("6. composite_score + tags...")
    scored = scoring.composite_score(pattern_scores, momentum_scores, sentiment_df["sentiment_score"])
    scored = scored.join(pattern_df[["breakout_prob"]], how="left")
    scored = scored.join(today_features_df[FEATURE_COLUMNS], how="left")
    scored = scored.join(sentiment_df[["sentiment_compound", "headline_count"]], how="left")
    scored["tags"] = scored.apply(lambda r: scoring.build_tags(r), axis=1)
    assert scored["trend_score"].between(0, 100).all()
    scored = scored.sort_values("trend_score", ascending=False)
    print(f"   OK -- top 3: {list(scored.head(3).index)}")
    print(f"   Engineered-breakout tickers (SYN000, SYN005, ...) rank: "
          f"{[scored.index.get_loc(t) for t in scored.index if t in ('SYN000','SYN005','SYN010') ]}")

    print("7. multi-day synthetic history + backtest...")
    history_rows = []
    for day_offset in range(30):
        for t in scored.index:
            history_rows.append({
                "date": (pd.Timestamp.today() - pd.Timedelta(days=30 - day_offset)).strftime("%Y-%m-%d"),
                "ticker": t,
                "close": float(prices[t]["Close"].iloc[-(30 - day_offset)]),
                "trend_score": float(scored.loc[t, "trend_score"]) + rng.normal(0, 5),
            })
    history = pd.DataFrame(history_rows)
    backtests = run_backtests(history)
    assert len(backtests) == 2
    print(f"   OK -- {[(b.horizon_label, b.ready) for b in backtests]}")

    print("8. build_data_json + write_data_json...")
    model_meta = {
        "trained": True,
        "validation_auc": None if np.isnan(model.validation_auc) else round(model.validation_auc, 3),
        "validation_precision_at_top_decile": None,
        "n_train_rows": model.n_train_rows,
        "n_positive_train": model.n_positive_train,
        "n_validation_rows": model.n_validation_rows,
    }
    # a synthetic "yesterday" comparison for a handful of tickers, to
    # exercise the day-over-day movers feature end to end
    sample_tickers = list(scored.index[:5])
    synthetic_score_change = {
        t: round(float(scored.at[t, "trend_score"]) - 40.0, 1) for t in sample_tickers
    }
    payload = build_data_json(scored, universe, prices, model_meta, backtests, synthetic_score_change)
    assert payload["universe_size"] == len(scored)
    assert len(payload["rows"]) == len(scored)
    for key in ("ticker", "trend_score", "sparkline", "tags", "currency", "market", "score_change"):
        assert key in payload["rows"][0], f"missing key {key} in dashboard row payload"
    assert "movers_top_n" in payload and "earnings_check_top_n" in payload and "telegram_score_threshold" in payload
    payload_by_ticker = {r["ticker"]: r for r in payload["rows"]}
    assert payload_by_ticker[NAN_PRICE_TICKER]["price"] is None, \
        "a NaN last close must become JSON null, not a raw NaN token that breaks JSON.parse() for every ticker"
    # write_data_json uses allow_nan=False -- this call itself is the real
    # regression check: it would raise ValueError here if the NaN price (or
    # any other stray NaN) leaked into the payload unguarded, exactly the
    # class of bug that broke the live site.
    write_data_json(payload)
    import json
    reloaded = json.load(open(config.DOCS_DIR / "data.json"))
    assert reloaded["universe_size"] == payload["universe_size"]
    print(f"   OK -- wrote {config.DOCS_DIR / 'data.json'} ({(config.DOCS_DIR / 'data.json').stat().st_size} bytes), round-trips through JSON")

    print("9. currency/market labeling for .ST tickers...")
    by_ticker = {r["ticker"]: r for r in payload["rows"]}
    se_rows = [r for t, r in by_ticker.items() if t.endswith(".ST")]
    us_rows = [r for t, r in by_ticker.items() if not t.endswith(".ST")]
    assert se_rows, "no Swedish rows made it into the payload"
    assert all(r["currency"] == "SEK" and r["market"] == "Sweden" for r in se_rows)
    assert all(r["currency"] == "USD" and r["market"] == "US" for r in us_rows)
    print(f"   OK -- {len(se_rows)} SEK rows, {len(us_rows)} USD rows, all labeled correctly")

    print("10. day-over-day score_change made it into the payload correctly...")
    by_ticker2 = {r["ticker"]: r for r in payload["rows"]}
    for t in sample_tickers:
        assert by_ticker2[t]["score_change"] == synthetic_score_change[t], f"score_change mismatch for {t}"
    untouched = [t for t in scored.index if t not in sample_tickers]
    assert all(by_ticker2[t]["score_change"] is None for t in untouched), \
        "tickers with no prior score should show score_change=None (\"new\"), not 0.0"
    print(f"   OK -- {len(sample_tickers)} tickers carry a real change, "
          f"{len(untouched)} correctly show None (no prior-day score)")

    print("11. earnings-date check fails soft without yfinance/network (this sandbox has neither)...")
    earnings_result = check_upcoming_earnings(list(scored.index[:3]))
    assert earnings_result == {}, "expected an empty (fail-soft) result with no yfinance/network available"
    assert tickers_with_earnings_soon(earnings_result) == set()
    print("   OK -- returned {} instead of raising, exactly the fail-soft behavior GitHub Actions relies on "
          "when a ticker's calendar data is missing")

    print("12. Telegram threshold-crossing logic (pure function, no network)...")
    universe_idx = universe.set_index("ticker")
    threshold = config.TELEGRAM_SCORE_THRESHOLD
    reset = config.TELEGRAM_RESET_THRESHOLD
    fake_scored = scored.copy()
    fake_scored["trend_score"] = reset - 10  # everyone else well below
    t_cross, t_above, t_new, t_seen, t_rearm = list(fake_scored.index[:5])
    fake_scored.loc[[t_cross, t_above, t_new, t_seen], "trend_score"] = threshold + 5
    prev_scores = {t_cross: threshold - 10, t_above: threshold + 1, t_seen: threshold - 10, t_rearm: threshold + 2}
    # t_new has no prev score; t_seen already alerted and never reset; t_rearm alerted then fell below reset
    crossers, alerted = notify.find_threshold_crossers(
        fake_scored, prev_scores, universe_idx, alerted={t_seen, t_rearm}
    )
    crossed_tickers = {c["ticker"] for c in crossers}
    assert crossed_tickers == {t_cross}, f"only the genuine below->above cross should alert, got {crossed_tickers}"
    assert {t_cross, t_above, t_new, t_seen} <= alerted, "everything above threshold should be remembered"
    assert t_rearm not in alerted, "a ticker that fell below the reset level should be re-armed"
    # next run: t_cross stays above -> no repeat alert
    again, _ = notify.find_threshold_crossers(fake_scored, {t_cross: threshold + 5}, universe_idx, alerted)
    assert not again, "an already-alerted ticker still above threshold must not re-alert"
    msg = notify.format_alert_message(crossers)
    assert t_cross in msg and str(threshold) in msg
    many = [{"ticker": f"T{i}", "name": f"N{i}", "score": 90.0, "prev": 80.0} for i in range(8)]
    long_msg = notify.format_alert_message(many)
    assert "T4" in long_msg and "T5" not in long_msg and "+ 3 till" in long_msg, "message should cap the list"
    print(f"   OK -- only real crosses alert, no-prev/hysteresis/cap rules all hold")

    print("13. Telegram send is a no-op (not an error) when secrets aren't configured...")
    import os
    os.environ.pop("TELEGRAM_BOT_TOKEN", None)
    os.environ.pop("TELEGRAM_CHAT_ID", None)
    sent = notify.send_telegram_message("test message -- should not actually send")
    assert sent is False, "expected send to report False (skipped) with no token/chat id configured"
    print("   OK -- skipped cleanly, no exception, no network attempted")

    print("14. Staleness-alert decision logic (pure function, no network)...")
    assert staleness_check.decide_alert(10, None) is None, "well under 30 days -- nothing to alert"
    assert staleness_check.decide_alert(35, None) == "30", "first crossing of the 30-day mark should alert '30'"
    assert staleness_check.decide_alert(35, "30") is None, "already alerted '30' for this streak -- no repeat"
    assert staleness_check.decide_alert(50, "30") == "45", "still stale past 45 days -- escalate to '45'"
    assert staleness_check.decide_alert(50, None) == "45", \
        "a streak already this stale should jump straight to '45', not a now-pointless '30'"
    assert staleness_check.decide_alert(50, "45") is None, "already alerted '45' -- no further escalation configured"
    print("   OK -- 30/45-day thresholds and no-repeat-alert logic all correct")

    print("15. Staleness check end-to-end is a no-op (not an error) without Telegram secrets configured...")
    # os.environ still has TELEGRAM_BOT_TOKEN/CHAT_ID popped from step 13 above.
    forty_days_ago_epoch = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=40)).timestamp()
    level = staleness_check.check_and_alert(forty_days_ago_epoch)
    assert level is None, "send should fail soft (no secrets) -- nothing should be recorded as sent"
    print("   OK -- fails soft exactly like the threshold-alert path above")

    print("16. Paper portfolio arithmetic on a hand-checkable case...")
    old_n, old_every, old_cost = config.PAPER_TOP_N, config.PAPER_REBALANCE_EVERY, config.PAPER_COST_BPS
    config.PAPER_TOP_N, config.PAPER_REBALANCE_EVERY, config.PAPER_COST_BPS = 1, 5, {"US": 15, "Sweden": 30}
    days = [f"2026-01-{d:02d}" for d in range(5, 16)]  # 11 runs
    rows = []
    for i, d in enumerate(days):
        a_price = 100.0 if i == 0 else 110.0       # A: +10% on day 1, then flat
        rows.append({"date": d, "ticker": "A", "close": a_price, "trend_score": 90 if i < 5 else 10})
        rows.append({"date": d, "ticker": "B", "close": 50.0, "trend_score": 50})  # B flat, top from day 5
    hist = pd.DataFrame(rows)
    bench = pd.DataFrame([{"date": d, "market": "US", "close": 1000.0} for d in days])
    res = paper_portfolio.simulate(hist, bench)
    # buy A (cost 0.15%), A +10%, switch A->B on day 5 (sell+buy = 2 x 0.15% of value), B flat
    expected = (1 - 0.0015) * 1.10 * (1 - 0.003)
    assert abs(res["total_return"] - round((expected - 1) * 100, 2)) < 0.011, (res["total_return"], expected)
    assert res["bench_return"] == 0.0 and res["rebalances"] == 2
    assert res["holdings"][0]["ticker"] == "B"
    assert abs(res["total_costs"] - 0.45) < 0.011
    config.PAPER_TOP_N, config.PAPER_REBALANCE_EVERY, config.PAPER_COST_BPS = old_n, old_every, old_cost
    print(f"   OK -- {res['total_return']}% matches the hand calculation {(expected - 1) * 100:.2f}%, costs {res['total_costs']}%")

    print("17. Model versioning: fingerprint + automatic sub-version...")
    reg: list = []
    v1, reg = versioning.resolve_version("2026-01-01", reg, persist=False)
    v1b, reg = versioning.resolve_version("2026-01-02", reg, persist=False)
    assert v1 == v1b == config.MODEL_VERSION and len(reg) == 1, "same settings must keep the same version"
    old_w = config.WEIGHT_SENTIMENT
    config.WEIGHT_SENTIMENT = old_w + 0.01
    v2, reg = versioning.resolve_version("2026-01-03", reg, persist=False)
    config.WEIGHT_SENTIMENT = old_w
    assert v2 != v1 and v2.startswith(config.MODEL_VERSION + "+"), "changed settings must get a new version"
    legacy = followup.with_version(pd.DataFrame({"date": ["2026-01-01"], "ticker": ["X"]}))
    assert legacy["model_version"].iloc[0] == config.LEGACY_MODEL_VERSION
    print(f"   OK -- {v1} kept for identical settings, {v2} registered after a change, old rows tagged "
          f"{config.LEGACY_MODEL_VERSION}")

    print("\nALL SYNTHETIC PIPELINE CHECKS PASSED")


if __name__ == "__main__":
    run()
