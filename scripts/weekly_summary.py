"""
Weekly Telegram summary, run by .github/workflows/weekly.yml on Sunday
morning. Reads only the files the daily pipeline already commits -- no
Yahoo/news requests -- and sends one message with:

  * the week's biggest Trend Score climbers
  * how recent alerts have done since they fired, versus their index
  * the running hit rate for alerts and for each day's top N
  * how the week's runs went (number of runs, tickers missing)

Fails soft like the rest of the alerting: no secrets -> nothing sent.

Run locally:  python scripts/weekly_summary.py --dry-run   (prints instead of sending)
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import followup  # noqa: E402
import notify  # noqa: E402
import paper_portfolio  # noqa: E402
import versioning  # noqa: E402

log = logging.getLogger("weekly_summary")


def _fmt_pct(x: float | None, signed: bool = True) -> str:
    if x is None:
        return "–"
    s = f"{x:+.1f}" if signed else f"{x:.1f}"
    return s.replace(".", ",") + " %"


def _fmt_pp(x: float | None) -> str:
    return "–" if x is None else f"{x:+.1f}".replace(".", ",") + " procentenheter"


def _names() -> dict[str, str]:
    if config.UNIVERSE_CACHE.exists():
        u = pd.read_csv(config.UNIVERSE_CACHE)
        return dict(zip(u["ticker"], u["name"]))
    return {}


def build_message(now: datetime | None = None) -> str | None:
    now = now or datetime.now(timezone.utc)
    week_ago = (now - timedelta(days=7)).strftime("%Y-%m-%d")
    history = followup.load_history()
    if history.empty:
        return None
    names = _names()
    lines = [f"\U0001F4CA Trendscanner – vecka {now.isocalendar().week}"]

    # --- biggest climbers: last score this week vs the last score before the week
    dates = sorted(history["date"].astype(str).unique())
    week_dates = [d for d in dates if d > week_ago]
    before = [d for d in dates if d <= week_ago]
    if week_dates:
        end = history[history["date"] == week_dates[-1]].set_index("ticker")["trend_score"]
        start_date = before[-1] if before else week_dates[0]
        start = history[history["date"] == start_date].set_index("ticker")["trend_score"]
        change = (end - start.reindex(end.index)).dropna().sort_values(ascending=False)
        top = change.head(config.WEEKLY_TOP_CLIMBERS)
        if len(top):
            lines += ["", "Veckans största klättrare:"]
            for t, ch in top.items():
                lines.append(f"• {t} ({names.get(t, t)}) — {start[t]:.1f} → {end[t]:.1f} ({ch:+.1f})")

    # --- alerts from the last 4 weeks and how they've done so far
    alert_log = followup.load_alert_log()
    benchmarks = followup.load_benchmarks()
    cutoff = (now - timedelta(days=28)).strftime("%Y-%m-%d")
    recent = alert_log[alert_log["date"].astype(str) >= cutoff] if not alert_log.empty else alert_log
    if not recent.empty:
        ev = followup.event_returns(recent[["date", "ticker", "name"]], history, benchmarks)
        lines += ["", "Larm senaste 4 veckorna – utveckling sedan larmet (index samma period):"]
        for _, r in ev.sort_values("date", ascending=False).head(8).iterrows():
            ret, bench, ex = (followup._pct(r[k]) for k in ("ret_now", "bench_now", "excess_now"))
            if ret is None:
                lines.append(f"• {r['ticker']} ({r['date'][5:]}): för nytt att mäta")
                continue
            mark = "" if ex is None else (" ✅" if ex > 0 else " ❌")
            lines.append(f"• {r['ticker']} ({r['date'][5:]}): {_fmt_pct(ret)} (index {_fmt_pct(bench)}){mark}")
    else:
        lines += ["", "Inga larm de senaste 4 veckorna."]

    # --- running hit rates
    payload = followup.build_followup_payload(history, alert_log, benchmarks)
    register = versioning.load_register()
    current = register[-1] if register else None
    version = current["version"] if current else None
    scope = payload["by_version"].get(version) if version else None
    scope = scope or {"alerts": payload["alerts"], "top": payload["top"]}
    h = str(10 if 10 in config.FOLLOWUP_HORIZONS else config.FOLLOWUP_HORIZONS[0])
    a, t = scope["alerts"].get(h, {}), scope["top"].get(h, {})
    lines += ["", f"Träffsäkerhet efter {h} handelsdagar" + (f" (modell {version}):" if version else ":")]
    if a.get("n"):
        lines.append(f"• Larm: {a['hit_rate']:.0f} % slog index, i snitt {_fmt_pp(a['avg_excess'])} mot index (n={a['n']})")
    else:
        lines.append("• Larm: inte tillräckligt med data än")
    if t.get("n"):
        lines.append(f"• Dagens topp {payload['top_n']}: {t['hit_rate']:.0f} % slog index, "
                     f"i snitt {_fmt_pp(t['avg_excess'])} mot index (n={t['n']})")
    else:
        lines.append(f"• Dagens topp {payload['top_n']}: inte tillräckligt med data än")

    # --- paper portfolio since the current model version started
    try:
        pp = paper_portfolio.build_paper_payload(history, benchmarks, version or config.MODEL_VERSION,
                                                 current["first_date"] if current else None)
        r, st = pp.get("current"), pp.get("status", {})
        lines += ["", "Pappersportfölj" + (f" (modell {version})" if version else "") + ":"]
        if r:
            lines.append(f"• {_fmt_pct(r['total_return'])} efter kostnader, index {_fmt_pct(r['bench_return'])} "
                         f"({_fmt_pp(r['excess'])})")
            lines.append(f"• Största nedgång {_fmt_pct(r['max_drawdown'])}, kostnader hittills {_fmt_pct(r['total_costs'], False)}")
        else:
            lines.append("• för få körningar än")
        state = {"validating": "under validering", "passed": "GODKÄND", "failed": "UNDERKÄND"}.get(st.get("state"), "under validering")
        lines.append(f"• Status: {state} ({st.get('runs', 0)} av {st.get('required_runs', config.PAPER_VALIDATION_DAYS)} körningar)")
    except Exception as exc:  # noqa: BLE001
        log.warning("Paper portfolio section failed: %s", exc)

    # --- run health
    if config.RUN_LOG_CSV.exists():
        runs = pd.read_csv(config.RUN_LOG_CSV)
        runs = runs[runs["date"].astype(str) > week_ago]
        lines += ["", "Drift:"]
        if runs.empty:
            lines.append("⚠️ Inga körningar loggade den här veckan – kolla Actions-fliken.")
        else:
            bad = runs[runs["missing_pct"] > config.DATA_QUALITY_WARN_MISSING_PCT]
            lines.append(f"• {len(runs)} körningar, i snitt saknades {runs['missing_pct'].mean():.1f} % av aktierna".replace(".", ",", 1))
            if len(bad):
                lines.append(f"⚠️ {len(bad)} körning(ar) saknade mer än {config.DATA_QUALITY_WARN_MISSING_PCT:.0f} % av aktierna")
            if "model_trained" in runs and not runs["model_trained"].astype(str).eq("True").all():
                lines.append("⚠️ Mönstermodellen kunde inte tränas i minst en körning")

    if config.DASHBOARD_URL:
        lines += ["", config.DASHBOARD_URL]
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    dry = "--dry-run" in sys.argv
    if not config.WEEKLY_SUMMARY_ENABLED and not dry:
        print("Weekly summary disabled in config.py")
        return
    try:
        msg = build_message()
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not build weekly summary: %s", exc)
        return
    if not msg:
        print("No history yet -- nothing to summarize.")
        return
    if dry:
        print(msg)
        return
    sent = notify.send_telegram_message(msg)
    print("Weekly summary sent." if sent else "Weekly summary NOT sent (see log above).")


if __name__ == "__main__":
    main()
