"""
Optional, free Telegram alert when a ticker's Trend Score crosses
config.TELEGRAM_SCORE_THRESHOLD: it was below the threshold in the previous
run and is at or above it today. Tickers with no previous score don't count,
and a ticker that has alerted must drop below config.TELEGRAM_RESET_THRESHOLD
before it can alert again (memory kept in config.TELEGRAM_ALERT_STATE_FILE).

Telegram's Bot API has no billing at all for this kind of use: a bot token
from @BotFather and your own numeric chat ID, both read from environment
variables (populated from GitHub Actions secrets, never committed to the
repo) rather than from config.py. If those two environment variables
aren't set, alerting is silently skipped -- this feature is fully optional
and the rest of the pipeline doesn't know or care whether it's configured.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

log = logging.getLogger("notify")

TELEGRAM_API_BASE = "https://api.telegram.org"


def load_alerted() -> set[str]:
    """Tickers that have already alerted and haven't fallen below the reset level since."""
    path = config.TELEGRAM_ALERT_STATE_FILE
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text()).get("alerted", []))
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read %s (%s) -- starting with an empty alert memory", path, exc)
        return set()


def save_alerted(alerted: set[str]) -> None:
    path = config.TELEGRAM_ALERT_STATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"alerted": sorted(alerted)}, indent=2) + "\n")


def find_threshold_crossers(
    scored: pd.DataFrame,
    prev_scores: dict[str, float],
    universe_idx: pd.DataFrame,
    alerted: set[str] | None = None,
) -> tuple[list[dict], set[str]]:
    """
    Returns (crossers, updated_alerted_set).

    A ticker is a real "crosser" only if ALL of these hold:
      * today's score >= TELEGRAM_SCORE_THRESHOLD
      * we have a score for it from the previous run, and that was below
        the threshold (no prior score = we can't tell whether it actually
        crossed -- typically it was just missing from yesterday's download
        because of Yahoo rate-limiting -- so it is NOT treated as new)
      * it hasn't already alerted without dropping below
        TELEGRAM_RESET_THRESHOLD since (hysteresis)

    Tickers above the threshold that don't qualify are silently added to the
    alerted set, so they won't fire later just because, say, a missing
    prior score shows up again.
    """
    threshold = config.TELEGRAM_SCORE_THRESHOLD
    reset = config.TELEGRAM_RESET_THRESHOLD
    alerted = set(alerted or ())
    crossers = []
    for ticker, r in scored.iterrows():
        today_score = r.get("trend_score")
        if today_score is None or pd.isna(today_score):
            continue  # no reading today -- leave its alert memory untouched
        if today_score < reset:
            alerted.discard(ticker)  # fell far enough: re-armed for a future alert
            continue
        if today_score < threshold or ticker in alerted:
            continue
        alerted.add(ticker)
        prev = prev_scores.get(ticker)
        if prev is None or pd.isna(prev) or prev >= threshold:
            continue  # above, but not an observed cross -- remember silently
        name = str(universe_idx.loc[ticker, "name"]) if ticker in universe_idx.index else ticker
        crossers.append({"ticker": ticker, "name": name, "score": float(today_score), "prev": float(prev)})
    crossers.sort(key=lambda c: c["score"], reverse=True)
    return crossers, alerted


def format_alert_message(crossers: list[dict]) -> str:
    threshold = config.TELEGRAM_SCORE_THRESHOLD
    max_n = config.TELEGRAM_MAX_TICKERS_IN_MESSAGE
    n = len(crossers)
    head = "1 aktie" if n == 1 else f"{n} aktier"
    lines = [f"\U0001F4C8 {head} gick över {threshold} idag:"]
    for c in crossers[:max_n]:
        lines.append(f"• {c['ticker']} ({c['name']}) — {c['prev']:.1f} → {c['score']:.1f}")
    if n > max_n:
        lines.append(f"+ {n - max_n} till, se sidan")
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
    crossers, alerted = find_threshold_crossers(scored, prev_scores, universe_idx, load_alerted())
    save_alerted(alerted)
    if not crossers:
        log.info("No new threshold crossers today -- no Telegram alert to send")
        return
    message = format_alert_message(crossers)
    sent = send_telegram_message(message)
    if sent:
        log.info("Sent Telegram alert for %d ticker(s)", len(crossers))
