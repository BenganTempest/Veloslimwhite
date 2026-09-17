"""
Warns via Telegram if the repo has gone quiet for a while.

This exists specifically because of GitHub's own behavior: a PUBLIC repo's
scheduled Actions workflows are automatically disabled after 60 days with
no repository activity (commits). Under normal operation this pipeline
resets that clock on its own every day it commits an updated docs/data.json
-- so in practice this only ever fires when something has been broken for a
sustained stretch (a crash before the commit step, a persistent rate-limit
or upstream-source problem, etc.), well before the 60-day cutoff actually
arrives:

  - config.STALENESS_WARN_DAYS   (default 30): a low-key first heads-up
  - config.STALENESS_URGENT_DAYS (default 45): a louder nudge if it's
    still unresolved two weeks later

Invoked from the workflow as a standalone step -- deliberately BEFORE
run_pipeline.py and with `if: always()` in the workflow, so it still runs
(and can still alert) even on a day the main pipeline step fails outright.
It reads the repo's last commit timestamp (passed in as a plain argument
computed via `git log` in the workflow, before this run makes any commit of
its own) rather than doing its own git calls, which keeps this module
trivially testable without a real git repo.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import notify  # noqa: E402

# Ascending severity order. Checked highest-first when deciding what level
# today's gap has reached, so a long-stale streak jumps straight to "45"
# rather than working its way up through "30" first.
THRESHOLDS = [
    (config.STALENESS_WARN_DAYS, "30"),
    (config.STALENESS_URGENT_DAYS, "45"),
]
SEVERITY = {level: rank for rank, (_, level) in enumerate(THRESHOLDS, start=1)}


def load_state() -> dict:
    path = config.STALENESS_STATE_FILE
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            return {}
    return {}


def save_state(state: dict) -> None:
    path = config.STALENESS_STATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state))


def decide_alert(days_since_commit: float, last_sent_level: str | None) -> str | None:
    """
    Pure decision function: given how long it's been since the repo's last
    commit and which level (if any) was already alerted for the CURRENT
    stale streak, returns the level to alert now ("30"/"45"), or None if
    there's nothing new to send. Compares by severity rank rather than
    equality, so once "45" has been sent, a later check that's still past
    30 days (but not past 45) doesn't re-trigger a now-redundant "30".
    """
    current = None
    for days, level in reversed(THRESHOLDS):  # highest threshold first
        if days_since_commit >= days:
            current = level
            break
    if current is None:
        return None
    last_severity = SEVERITY.get(last_sent_level, 0)
    if SEVERITY[current] > last_severity:
        return current
    return None


def format_message(level: str, days_since_commit: int) -> str:
    dashboard_note = f"\n{config.DASHBOARD_URL}" if config.DASHBOARD_URL else ""
    if level == "45":
        return (
            f"⚠️ Trend Scanner: still no update to the repo in {days_since_commit} days. "
            f"GitHub auto-disables a public repo's scheduled Actions after 60 days with no activity "
            f"-- worth checking the Actions tab soon, or the daily scan may stop running "
            f"entirely.{dashboard_note}"
        )
    return (
        f"ℹ️ Trend Scanner: no update to the repo in {days_since_commit} days. "
        f"Probably nothing to worry about yet, but GitHub disables a public repo's scheduled Actions "
        f"after 60 days of inactivity -- worth a quick look at the Actions tab if this keeps "
        f"climbing.{dashboard_note}"
    )


def check_and_alert(last_commit_epoch: float, now: datetime | None = None) -> str | None:
    """
    Returns the alert level actually sent ("30"/"45"), or None. Fails soft
    by design (see main()) -- a broken staleness check must never block or
    break the actual daily scan.
    """
    if not config.TELEGRAM_ALERTS_ENABLED:
        return None
    now = now or datetime.now(timezone.utc)
    last_commit = datetime.fromtimestamp(last_commit_epoch, tz=timezone.utc)
    days_since = (now - last_commit).total_seconds() / 86400.0

    state = load_state()
    last_sent_level = state.get("level")

    if days_since < config.STALENESS_WARN_DAYS:
        # A recent commit means any prior stale streak is over -- clear the
        # state so a FUTURE stale streak starts fresh at "30", not stuck
        # thinking "45" already went out.
        if state:
            save_state({})
        return None

    level = decide_alert(days_since, last_sent_level)
    if level is None:
        return None

    sent = notify.send_telegram_message(format_message(level, int(days_since)))
    if sent:
        save_state({"level": level, "days_since_commit": int(days_since)})
        return level
    return None


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: staleness_check.py <last_commit_epoch_seconds>", file=sys.stderr)
        return
    try:
        last_commit_epoch = float(sys.argv[1])
    except ValueError:
        print("Could not parse last-commit epoch seconds -- skipping staleness check.", file=sys.stderr)
        return
    try:
        level = check_and_alert(last_commit_epoch)
        if level:
            print(f"Staleness check: sent a level-{level} Telegram alert.")
        else:
            print("Staleness check: nothing to send.")
    except Exception as exc:  # noqa: BLE001
        # Same fail-soft principle as the rest of the pipeline -- this must
        # never be the reason a scheduled run shows a red X.
        print(f"Staleness check failed (non-fatal): {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
