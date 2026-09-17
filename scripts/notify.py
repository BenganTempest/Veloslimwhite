"""
Optional, free Telegram alert when a ticker's Trend Score crosses
config.TELEGRAM_SCORE_THRESHOLD for the first time (i.e. it was below the
threshold yesterday -- or there was no "yesterday" data -- and is at or
above it today).

Telegram's Bot API has no billing at all for this kind of use: a bot token
from @BotFather and your own numeric chat ID, both read from environment
variables (populated from GitHub Actions secrets, never committed to the
repo) rather than from config.py. If those two environment variables
aren't set, alerting is silently skipped -- this feature is fully optional
and the rest of the pipeline doesn't know or care whether it's configured.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

log = logging.getLogger("notify")

TELEGRAM_API_BASE = "https://api.telegram.org"


def find_threshold_crossers(
    scored: pd.DataFrame, prev_scores: dict[str, float], universe_idx: pd.DataFrame
) -> list[dict]:
    """
    A ticker "crosses" if today's trend_score >= threshold and yesterday's
    score (if we have one) was below it. No prior score at all (new ticker,
    or first-ever run) also counts as a cross -- there's nothing to compare
    against, and a brand-new high scorer is exactly the kind of thing worth
    flagging rather than silently skipping.
    """
    threshold = config.TELEGRAM_SCORE_THRESHOLD
    crossers = []
    for ticker, r in scored.iterrows():
        today_score = r.get("trend_score")
        if today_score is None or pd.isna(today_score) or today_score < threshold:
            continue
        prev = prev_scores.get(ticker)
        if prev is not None and prev >= threshold:
            continue  # was already above threshold yesterday -- not a new cross
        name = str(universe_idx.loc[ticker, "name"]) if ticker in universe_idx.index else ticker
        crossers.append({"ticker": ticker, "name": name, "score": float(today_score)})
    crossers.sort(key=lambda c: c["score"], reverse=True)
    return crossers


def format_alert_message(crossers: list[dict]) -> str:
    threshold = config.TELEGRAM_SCORE_THRESHOLD
    lines = [f"\U0001F4C8 {len(crossers)} ticker(s) crossed above {threshold} today:"]
    for c in crossers:
        lines.append(f"• {c['ticker']} ({c['name']}) — {c['score']:.1f}")
    if config.DASHBOARD_URL:
        lines.append("")
        lines.append(config.DASHBOARD_URL)
    return "\n".join(lines)


def send_telegram_message(text: str) -> bool:
    """
    Fails soft on purpose: a network hiccup or bad token should never take
    down the rest of the pipeline (docs/data.json still needs to get
    written either way). Returns True/False only for logging/testing.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        log.info("Telegram not configured (TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not set) -- skipping alert")
        return False

    import requests  # lazy import, same reasoning as elsewhere: keep this module testable without it installed

    try:
        resp = requests.post(
            f"{TELEGRAM_API_BASE}/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
            timeout=10,
        )
        if resp.status_code != 200:
            log.warning("Telegram API returned %s: %s", resp.status_code, resp.text[:200])
            return False
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("Telegram send failed: %s", exc)
        return False


def maybe_send_threshold_alert(
    scored: pd.DataFrame, prev_scores: dict[str, float], universe_idx: pd.DataFrame
) -> None:
    if not config.TELEGRAM_ALERTS_ENABLED:
        return
    crossers = find_threshold_crossers(scored, prev_scores, universe_idx)
    if not crossers:
        log.info("No new threshold crossers today -- no Telegram alert to send")
        return
    message = format_alert_message(crossers)
    sent = send_telegram_message(message)
    if sent:
        log.info("Sent Telegram alert for %d ticker(s)", len(crossers))
