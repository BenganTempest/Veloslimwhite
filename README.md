# Trend Scanner

A daily, automated dashboard that scans the S&P 500, Nasdaq-100, and the Nasdaq
Stockholm all-share list (including smaller First North names) for stocks whose
recent price/volume behavior resembles stocks that had big moves in the past, blended
with current momentum and news sentiment, into one 0–100 "Trend Score" per ticker.

**This is a research toy, not investment advice.** Read [Limitations](#limitations)
before you trust anything it shows you. The dashboard itself repeats this and shows
an honesty panel that tracks, in arrears, whether its own past scores actually meant
anything.

It costs nothing to run: GitHub Actions' free tier does the daily compute, GitHub
Pages hosts the dashboard, and every data source (Yahoo Finance via `yfinance`,
Wikipedia and stockanalysis.com for the ticker lists, Google News RSS for headlines)
is free and keyless.

Prices are shown in each ticker's own currency (USD for US names, SEK for
Stockholm-listed names) — the dashboard has a Market filter to isolate one or the
other.

## How it works

```
GitHub Actions (daily, scheduled)
  -> scripts/staleness_check.py  runs FIRST, even if the pipeline below fails: optional
                                  free Telegram nudge if the repo hasn't been updated in
                                  30/45 days (see "Optional: free Telegram alerts" below)
  -> scripts/universe.py     pull S&P 500 + Nasdaq-100 tickers from Wikipedia, plus
                              the Nasdaq Stockholm all-share list from stockanalysis.com
  -> scripts/collect.py      pull 2y OHLCV per ticker (yfinance) + recent headlines (Google News RSS)
  -> scripts/features.py     momentum, volatility, relative volume, RSI, distance from 52w high
  -> scripts/pattern_model.py  train a small logistic regression: does today's feature
                                shape resemble stocks' feature shape before a past
                                >=20% / 10-trading-day move? (time-split validated, so
                                the reported accuracy isn't inflated by lookahead)
  -> scripts/score.py        blend pattern score + momentum + news sentiment (VADER) -> Trend Score
  -> scripts/earnings.py     best-effort "earnings soon" tag, top-scoring tickers only
  -> scripts/notify.py       optional free Telegram alert on a new threshold crossing
  -> scripts/backtest.py     check the scanner's OWN accumulated daily snapshots: did
                              high-scored stocks actually outperform low-scored ones?
  -> scripts/build_dashboard.py  write docs/data.json (incl. day-over-day score change)
  -> commit docs/data.json + data/history.csv back to the repo
  -> GitHub Pages serves docs/index.html, which fetches data.json client-side
```

Nothing runs continuously — it's a scheduled batch job, once a day, which is enough
for a slower-moving research signal like this. `docs/index.html` is a static page
with no backend; it just re-fetches `data.json` each time you load it.

## One-time setup (about 10 minutes)

1. **Create a free GitHub account** at github.com if you don't have one.

2. **Create a new repository.** On github.com, click "New repository". Any name
   (e.g. `trend-scanner`). Public is simplest for free GitHub Pages; private repos
   also get free Pages on current GitHub plans, but public is the well-trodden path.
   Don't initialize it with a README (you already have one).

3. **Push this project to it.** From a terminal, inside this folder:

   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```

4. **Enable GitHub Pages.** In your repo on github.com: Settings -> Pages ->
   under "Build and deployment", Source = "Deploy from a branch", Branch = `main`,
   folder = `/docs`. Save. GitHub will give you a URL like
   `https://<your-username>.github.io/<your-repo>/` — that's your dashboard, once
   it's populated (next step).

5. **Run the pipeline once manually** instead of waiting for the schedule: repo ->
   Actions tab -> "Daily trend scan" (left sidebar) -> "Run workflow" button -> Run
   workflow. It takes a few minutes (it's downloading price history for ~600
   tickers). Watch it under the Actions tab; a green check means it worked and
   committed `docs/data.json`.

6. **Visit your Pages URL.** It can take a minute or two after the first successful
   run for Pages to pick up the new commit. Reload if it still shows the "could not
   load data.json yet" message.

After that, it runs on its own: the schedule in `.github/workflows/daily.yml` fires
at 21:30 UTC on weekdays (after the US close for most of the year — edit the cron
line if you want a different time), and each run appends to the honesty panel's
history.

## Optional: free Telegram alerts

The pipeline can ping you on Telegram the first time a ticker's Trend Score crosses
`config.TELEGRAM_SCORE_THRESHOLD` (default 75). This is entirely free — no billing
account, no credit card, ever — but it does need two values only you can create,
stored as GitHub Actions secrets (never in the repo):

1. **Create a bot.** In Telegram, message **@BotFather**, send `/newbot`, and follow
   the prompts (pick any name). It replies with a bot token that looks like
   `123456789:AAAbc-Def...`. That's `TELEGRAM_BOT_TOKEN`.
2. **Get your chat ID.** Message your new bot anything first (bots can't message you
   until you've messaged them), then message **@userinfobot** and it will reply with
   your numeric ID. That's `TELEGRAM_CHAT_ID`.
3. **Add both as repo secrets.** On github.com: your repo -> Settings -> Secrets and
   variables -> Actions -> "New repository secret". Add `TELEGRAM_BOT_TOKEN` and
   `TELEGRAM_CHAT_ID` with the values above.

That's it — the next run will alert you automatically. Skip all of this entirely and
the pipeline just won't send alerts; nothing else changes or breaks. Set
`TELEGRAM_ALERTS_ENABLED = False` in `config.py` to turn it off explicitly, and change
`TELEGRAM_SCORE_THRESHOLD` (and, if you want the message to link back, `DASHBOARD_URL`)
to taste.

The same bot also covers a second, unrelated case: GitHub automatically disables a
**public** repo's *scheduled* Actions workflows after 60 days with no repository
activity ([GitHub Docs](https://docs.github.com/actions/managing-workflow-runs/disabling-and-enabling-a-workflow)) —
after that, this whole pipeline would just silently stop running. Under normal
operation this is a non-issue: the daily run itself commits `docs/data.json`
whenever the numbers change, which resets that 60-day clock every day on its own.
It only becomes a real risk if something breaks badly enough that the pipeline
stops reaching its own commit step for a long stretch — so `scripts/staleness_check.py`
runs at the very start of every scheduled run (before the main pipeline, and even
on a day the main pipeline step fails) and sends a Telegram heads-up at 30 days
since the repo's last commit, then a more urgent one at 45 if it's still stale two
weeks after that — well before GitHub's 60-day cutoff actually arrives. Tune
`STALENESS_WARN_DAYS` / `STALENESS_URGENT_DAYS` in `config.py` to taste; it's
silently skipped if `TELEGRAM_ALERTS_ENABLED = False` or the two Telegram secrets
above aren't set, same as the threshold alert.

## Customizing

Everything tunable lives in `config.py` with comments: which indices to scan,
the momentum/volatility windows, the breakout definition used to train the pattern
model (default: +20% in 10 trading days), the score's weighting between pattern
similarity / momentum / sentiment, and news lookback window. Change a value, commit,
push — the next scheduled or manual run picks it up.

## Limitations (read this)

- **Not a forecast.** The pattern-similarity score measures resemblance to *past*
  run-ups using price/volume shape alone. Markets change regime, the training set is
  modest, and this exact setup (small labeled dataset, many correlated features) is
  where overfitting is easy. The dashboard shows the model's honest out-of-sample AUC
  every run — expect it to hover not far from 0.5 (no skill) much of the time. That's
  not a bug to fix by tuning until AUC looks better; a much higher number on this kind
  of label would be a red flag for leakage, not a breakthrough.
- **The honesty panel needs time.** It checks the scanner's own live scores against
  what actually happened, using its own accumulated daily snapshots — so it reports
  "not enough data yet" until it's been running for the full horizon (1 month, then 3
  months). Don't trust it, or distrust it, before then.
- **Sentiment is a rough signal.** VADER is a general-purpose lexicon, not tuned for
  financial language ("shares fell on strong earnings" reads oddly to a plain lexicon).
  Treat the sentiment column as buzz/tone, not precision.
- **Yahoo/yfinance rate limiting will happen sometimes, and the tickers-scanned
  count will wobble day to day because of it.** At ~1,500 tickers this is normal,
  not a sign anything's broken. `scripts/collect.py` retries once after a pause if
  a chunk of tickers fails; `config.PRICE_DOWNLOAD_MAX_RETRIES` /
  `PRICE_DOWNLOAD_RETRY_PAUSE_SECONDS` control that. If the scanned count trends
  down hard over several days rather than just wobbling, check the Actions log for
  `YFRateLimitError` or `Failed downloads` — that's the tell.
- **Google News RSS is not an official, documented API.** It's a commonly used free
  endpoint, but it can change or rate-limit without notice. `scripts/collect.py` is
  written to fail soft (a broken news fetch just zeroes out sentiment for that run,
  it won't crash the pipeline) — check the Actions logs occasionally. If it breaks
  for good, the cleanest fix is swapping in a proper news API (e.g. NewsAPI.org's
  free tier) and storing the key as a GitHub Actions secret rather than in the repo.
- **The Nasdaq Stockholm list is confirmed capped at ~500 of ~752 listed names.**
  stockanalysis.com's page reports 752 Nasdaq Stockholm stocks, but the plain HTML
  our scraper reads only ever contains the first 500 — the rest loads later via
  client-side JavaScript (infinite scroll), which a simple `requests.get()` can't
  trigger. This isn't a header-matching bug like the earlier Wikipedia ones and
  isn't something `scripts/universe.py` can fix by parsing more carefully. There's
  an undocumented endpoint the site's own JS uses to fetch the rest, but it needs
  auth for anything beyond the free tier and its exact data format wasn't something
  that could be safely reverse-engineered and verified from this project's
  development environment — a fix that turned out wrong would silently corrupt
  ticker data rather than fail loudly, which is worse than the current honest
  500-name cap. Check the Actions log line `OMX Stockholm: got N tickers` after a
  run; it'll keep reading ~500 until this is revisited with a proper fix (a
  different data source, or a headless-browser fetch). It also has no sector data, so `scripts/universe.py`
  backfills sector/industry for any ticker still marked "Unknown" with a per-ticker
  `yfinance` lookup — a much heavier call than the bulk price download, so it's
  capped (`SECTOR_ENRICH_MAX_PER_RUN` in `config.py`), threaded gently, best-effort,
  and cached in `data/universe.csv` so a ticker only gets looked up once, not every
  day. A few names may still end up "Unknown" if Yahoo doesn't have sector data for
  them either (common for very small/thin names) — filter by Market if that's
  annoying rather than Sector.
- **Built and tested against synthetic data.** The environment this project was
  developed in has a network policy that blocks financial data hosts and can't
  install `yfinance`/`feedparser`/`vaderSentiment` from PyPI (see below) — so the full
  pipeline was validated end-to-end against synthetic price data
  (`tests/test_pipeline_synthetic.py`) and the dashboard was screenshot-tested with
  Playwright, but not against a real live run. Watch the logs on your first real
  Actions run in case real yfinance/Wikipedia output has a quirk the synthetic data
  didn't exercise (a changed Wikipedia table column name is the most likely one —
  `scripts/universe.py` already has some defensive handling for that).
- **Point-in-time snapshot, not intraday.** Prices/scores update once a day. This
  isn't built for day trading or anything time-sensitive.
- **"Earnings soon" only checks the top-scoring tickers, and dates can be wrong.**
  Checking every ticker's earnings calendar daily would mean 1000+ more individual
  Yahoo requests on top of the ones that already caused rate-limiting once (see
  above) — so `scripts/earnings.py` only checks the top `EARNINGS_CHECK_TOP_N`
  (default 100) tickers by today's score, fails soft per-ticker, and Yahoo's
  calendar data is its own estimate that can shift or be missing, especially for
  smaller names.
- **Day-over-day movers need a "yesterday" to compare against.** A ticker shows
  `score_change: null` ("new" on the dashboard) on the scanner's first-ever run, or
  the first day a given ticker appears in the universe — nothing wrong, just nothing
  to diff against yet. It fills in from the second run onward.
- **A ticker with no valid last-close price (halted, delisted, just listed) shows a
  "–" for price instead of breaking the page.** This was a real bug found on the
  live site: Yahoo returned `NaN` for one ticker's latest close, and Python's `json`
  module writes that out as a bare `NaN` token by default, which isn't valid JSON —
  the browser's `JSON.parse()` then rejected the *entire* file, not just that one
  row, so nothing loaded at all. Fixed in `scripts/build_dashboard.py` (a NaN close
  now becomes `null`, same as any other missing value) and hardened generally:
  `write_data_json()` now writes with `allow_nan=False`, so if any *other* field
  ever leaks a stray NaN/Infinity in the future, the pipeline fails loudly in the
  Actions log instead of silently shipping a `data.json` that breaks the dashboard
  for everyone until someone notices.

## Development notes

`tests/test_pipeline_synthetic.py` runs the real feature engineering, pattern-model
training/validation, scoring, backtest, and dashboard-JSON logic against synthetic
OHLCV data (no network needed) — useful for checking a change to `config.py` or the
scoring math before pushing:

```bash
pip install -r requirements.txt
python3 tests/test_pipeline_synthetic.py
```

`docs/index.html` is the dashboard shell (HTML/CSS/vanilla JS, no build step, no
external dependencies) and rarely needs to change; each pipeline run only rewrites
`docs/data.json` and appends a few rows to `data/history.csv`, which keeps git
history small and diffs readable.
