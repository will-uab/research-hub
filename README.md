# research-hub

A personal research-surveillance hub that runs unattended on a small always-on machine (it was built on a Raspberry Pi 5)
and publishes one phone-friendly page. Every morning it collects new work relevant to *your* interests from public sources,
triages it with Claude, and shows it by domain, by stream or by person:

* **Papers and preprints** (PubMed, medRxiv, arXiv, major journals) and **papers by people you follow** (OpenAlex)
* **Clinical trials** (ClinicalTrials.gov), **grant announcements** (Grants.gov, NIH Guide notices) and **new NIH awards** (RePORTER)
* A weekly **journal club** pick with a discussion guide, a daily and weekly **briefing** (sectioned prose with linked sources)
* An on-demand **two-host audio briefing** with a journal-club style deep dive on one open-access paper, voiced locally with Piper
* A **learning loop**: 👍/👎 on cards (with reasons) become *proposed* edits to your preferences, which you approve or reject
* A hard **budget guard**: every model call is priced before it is made and refused if it would exceed your daily cap

It is a single-user tool. The page is served on loopback and meant to be exposed only to your own network (for example with
`tailscale serve`); there is no login, so do not put it on the public internet.

```
sources ─► per-stream fetch ─► dedup (SQLite) ─► keyword score ─► Claude triage (budget-guarded) ─► site/ (static page + JSON API)
                                                                          └─► briefing ─► on-demand podcast (Claude script + Piper voices)
```

> **Not medical or professional advice.** Triage scores, briefings and audio are produced by an AI from titles, abstracts and
> registry entries (and, for the podcast deep dive, the open full text of one paper). They can be wrong or incomplete: always read the
> source. Data come from public APIs (NCBI E-utilities, medRxiv, arXiv, Grants.gov, NIH RePORTER, ClinicalTrials.gov, OpenAlex,
> Europe PMC); please follow each service's terms of use and rate limits.

## Quick start

```bash
git clone https://github.com/<you>/research-hub.git ~/research-hub && cd ~/research-hub
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env && chmod 600 .env          # add ANTHROPIC_API_KEY, model names and prices; never commit .env
cp config/interests.example.yaml config/interests.yaml   # then edit the profile, domains and (optionally) people_to_watch
.venv/bin/pip install -r requirements-dev.txt && .venv/bin/python -m pytest -q    # optional
bin/lit --dry-run                                # no API calls, no spend: writes a preview to out/preview/
bin/lit                                          # a real literature run
```

`.env` and `config/interests.yaml` hold your secrets and personal choices and are git-ignored; keep them out of version control.
For the podcast you also need `ffmpeg` and Piper voice files (see *Audio briefing* below). `lit gen-systemd` writes the systemd
service and timer files for your username and install path and prints the install commands.

### Commands

| Command | What it does |
|---|---|
| `lit` | Normal run: retrieve, triage, publish/email, record. This is what systemd runs. |
| `lit --dry-run` | Writes `out/digest-DATE.html` (+ `.txt`) and prints a cost estimate. **No email, no Anthropic calls, nothing recorded as seen.** Works with no `.env` (prices then come from `.env.example`). Triage is a keyword-score stub. |
| `lit --backfill N` | Look back N days instead of 2 (for initial tuning). Combine with `--dry-run`. |
| `lit --max-candidates K` | Override the 60-paper LLM cap for this run. |
| `lit build-site` | Regenerate `site/` data from the database without running a stream. |
| `lit gen-systemd` | Write a service + timer per enabled stream from the `schedule:` fields and print the install commands. |
| `lit migrate-config` | Convert an old (v1) `interests.yaml` to v2; the original is kept as `interests.v1.yaml.bak`. |
| `lit feedback +12345678 -23456789` | Like / dislike papers by the code shown under each item (`+PMID`; arXiv: `+arxiv:2411.01234`; medRxiv: `+doi:10.1101/...`). Up to 10 recent verdicts are fed to the triage prompt as calibration examples, so cost stays flat. |

Exit code is non-zero if retrieval fails entirely (all three sources) or delivery fails; in that case a red
"FAILED" entry appears on the web index (and a short email is sent if email delivery is on). Partial source failures
still produce the digest, with the errors listed in its footer.

## Editing your interests and journals

Everything lives in `config/interests.yaml` and is re-read on each run, so there's nothing to restart.

* **`profile`** is the plain-English description the LLM sees. Edit it as your interests shift.
* **`groups`** are the keyword queries. A group matches when *every* clause in `all` matches (AND); inside a clause,
  any `mesh` heading or `tiab` phrase counts (OR). `weight` ranks candidates before the LLM sees them; the AI/LLM
  methods groups are weighted highest (9-10) and the HIV/OUD groups lowest (4). The same groups generate the PubMed
  query and score medRxiv/arXiv items locally. `*` is a wildcard; short ALL-CAPS acronyms (HIV, LLM, AIDS) are matched
  case-sensitively.
* **`major.journals`** is the major-papers list. `ta` is the PubMed journal abbreviation (find it on the journal's
  PubMed page or in the NLM Catalog). `filter: net` restricts that journal to the broad topic net in `major.net`;
  `filter: none` takes every new item.
* **`medrxiv.categories`** and **`arxiv.categories` / `arxiv.health_terms`** control the local pre-filters.
  arXiv items are dropped unless the title/abstract contains one of the health terms. Set
  `require_health_terms: false` to disable that gate (expect many more candidates).
* **`limits`**: look-back window, the 60-candidate LLM cap, the share reserved for major-journal items, batch size,
  abstract truncation (250 words), and the number of feedback examples.
* **`digest`**: relevance thresholds (0-10), section sizes (major max 8), and `send_if_empty: false` to stay silent
  on empty days instead of sending "nothing notable today".

Check what a change does without spending anything: `lit --dry-run --backfill 3` and open `out/digest-DATE.html`.
To see the exact PubMed strings: `.venv/bin/python -c "import yaml,datetime as d; from litsearch import queries; \
c=yaml.safe_load(open('config/interests.yaml')); print(queries.build_all(c, d.date.today()-d.timedelta(2), d.date.today()))"`.

## Getting an NCBI API key (optional)

Without a key PubMed allows 3 requests/second; with one, 10. This pipeline makes only a handful of calls a day,
so the key is a nicety. Sign in at <https://account.ncbi.nlm.nih.gov/>, open *Account settings*, *API Key
Management*, create a key, and put it in `.env` as `NCBI_API_KEY=`.

## Research hub: five streams, one page

Nine independent streams feed one page, all driven by `config/interests.yaml` (version 2: three domains, plus a block per stream):

| Stream | Schedule (America/Chicago) | Source | Notes |
|---|---|---|---|
| `journal_club` | Mon 04:35 | stored literature items | 1 primary + 2 alternates with discussion guide (abstract-based); rotates the primary pick across domains; stronger model; weekly allowance |
| `people` | daily 04:30 | OpenAlex | new papers and preprints by the people on your watch list (a paper already in the Literature feed is flagged 👤 there instead of repeated); everything they publish is shown, whatever the LLM score |
| `briefing` | daily 04:45 (+ weekly on Mondays) | the other streams' stored items | a curated paragraph-form summary written by Sonnet; pops up at the top of the page until you tap **Got it** (synced across devices); items it cites are linked; skipped on quiet days (`min_items`); the weekly piece compares volumes with the previous week to describe trends |
| `podcast` | on demand (🎙 button on the page) | today's daily briefing and the items behind it | a ~10-minute two-host audio news briefing (Mondays ~12 min, adding the weekly trends and journal club): Sonnet writes the script from the stored items only, then **Piper** (local, no cloud TTS) voices it and `ffmpeg` encodes `site/audio/YYYY-MM-DD.mp3`; the page gets a player (speed control, lock-screen metadata) and a transcript with links; skipped when there is no briefing; the last 30 days of audio are kept |
| `tuning` | daily 05:00 | your ratings | proposes edits to your learned preferences (first report at 10 ratings, then every 20 new ones); you approve each on the page; see below |
| `literature` | daily 04:00 | PubMed, medRxiv, arXiv, major journals | the original digest; also still publishes the per-day digest archive (`digests.html`) |
| `trials` | daily 04:10 | ClinicalTrials.gov v2 | new registrations, newly posted results, status changes of trials already shown; AI/LLM interventions flagged |
| `grants` | daily 04:20 (+ Friday summary) | Grants.gov (NIH + other HHS) + NIH Guide notices | **NIH stopped posting NOFOs in the Guide in FY2026 (NOT-OD-25-143)**, so NOFOs come from Grants.gov and the Guide RSS is used only to catch modification/expiration notices; PCORI is not available (its pages return 403 to scripted access and it is not on Grants.gov) |
| `reporter` | Sun 04:15 | NIH RePORTER v2 | newly awarded projects (new/competing by default) in your domains + anything new for `people_to_watch` |

Run one: `lit --stream grants` (add `--dry-run` for stubbed triage and a preview in `out/preview/`, `--backfill N` for a longer look-back). With no `--stream`, `lit` runs `literature`. Each stream regenerates the page data when it finishes, whether it succeeded or failed (failures show in the status banner). One failing source never blocks the others: they are separate timers and separate processes.

**The page** (`/`): *By domain* (LLM/AI, Infectious Diseases, HIV + Addiction, All) or *By stream*, with the other dimension as chips; date range, minimum relevance, sort (relevance / newest / deadline) and search (titles, people, sponsors, abstracts). State lives in the URL hash (bookmarkable) and `localStorage`. Grants with a deadline in the future always show regardless of the date range. The "What's new" panel shows what arrived since your last visit, grants due within 60 days, new awards for watched PIs, trials with new results, and this week's journal club pick. The banner shows per-stream health and API spend against the daily cap and weekly allowance. Data is JSON in `site/data/` (rolling 90 days; older items stay in SQLite); abstracts load on tap.

### Rating, saving, and the watch list on the page

The page is served by a small server (`litsearch.web`, unit `litsearch-web.service`) that also exposes a JSON API; it listens on `127.0.0.1` and `tailscale serve` is unchanged. Each card has 👍 👎 🔖 buttons: ratings feed that stream's triage prompt (a stream only sees ratings of its own items), 👎 items are dimmed and never featured, saves keep an item on the page past the 90-day window, and opening an item marks it read. "🔖 Saved only" and "Hide read" are in the filters. Marks live in SQLite (`marks`, `feedback`), so your phone and desktop agree. Without the API (a static preview) the buttons simply don't appear.

**Security of the write endpoints:** the Host header must be `localhost`, `127.0.0.1` or a `*.ts.net` name (or listed in `LIT_ALLOWED_HOSTS`); writes need `Content-Type: application/json` *and* an `X-LH` header (which makes browsers send a CORS preflight the server never approves, so other websites can't trigger them); an `Origin`, if present, must be your own host; bodies are limited to 8 KB.

**Following people (By person):** the third view, next to By domain and By stream, lists everyone you follow with item counts; tap a name to see just them, with **Unfollow** on their header. Open **＋ Follow someone new** to search, then search a name, pick the RePORTER profile (for NIH awards) and tick every OpenAlex record that is that person (OpenAlex sometimes splits one person into several), then **Follow**. Funded-project cards also have a **＋ Follow** button next to each PI. From a terminal: `lit watch list | search "Name" | add "Last, First" [--profile-id N] [--openalex-id A…] | remove "Last, First" | resolve` (`resolve` finds OpenAlex ids for anyone lacking them, only when the person's institution confirms the match). The list is the `people_to_watch` block in `interests.yaml` (written atomically under a lock, so hand edits are fine too). Entries may carry `profile_id` (RePORTER) and `openalex_id` (one id or a list); pinned entries match exactly, name-only entries match by name.

### Learning from your ratings

Taps feed three kinds of memory, from shortest to longest:
1. **Short-term examples:** each stream's triage prompt shows your latest likes and dislikes (balanced: up to half of each).
2. **Reasons and snapshots:** after 👍/👎 an optional one-tap reason appears (👎: not my topic, already knew it, weak study, wrong population, too basic; 👍: on topic, great method, useful for a project, follow this person). Each rating stores a **snapshot** of the item as it looked then, including the pipeline model's score, so "the model said 9 and you disliked it" can be learned from.
3. **Learned preferences in `interests.yaml`:** the `learned:` block holds a short `preferences` paragraph (added to every triage prompt as soft guidance; your hand-written `profile` takes precedence) plus `prefer` / `deprioritize` rules (kinds: term, journal, author, institute, mechanism, sponsor). Rules only **re-rank**: each item gets an adjustment clipped to -3..+2 on the 0-10 scale, used for the candidate cap, the page's ranking and Featured. Nothing is hidden (term rules match titles only), and cards show a tag like *↓ veterinary*.

**The tune-up** (`tuning` stream, 05:00 daily) runs when 10 ratings exist, then after every 20 new ones (`first_after` / `every`; `lit --stream tuning --force` to run now, `--dry-run` for a preview). A deterministic step finds candidate rules (at least 3 rated items, at least 70% consistent, older ratings fading with a 180-day half-life; "already knew it" and "too basic" count for nothing, "weak study" only for sources). One Sonnet call then picks from those candidates (it cannot invent rules or people; its weights are capped by the evidence), rewrites the paragraph, suggests people to follow from authors of papers you liked, and notes where its scores and your taps disagree. Each rule comes with a 30-day **backtest**. Open **🎛 Tuning** on the page to **Accept / Reject** each proposal (nothing changes until you do; accepted changes can be **Undone**). Needs `LLM_MODEL_TUNING` and `PRICE_TUNING_*` in `.env` (see `.env.example`).

### Common edits

* **People to watch:** use the page's watch panel or `lit watch` (above). Their new NIH awards (new, competing and supplement awards; set `watch_award_types` in the `reporter` block to add `"5"` for annual continuations) are always shown in Funded Projects, and their new papers in People.
* **Add a domain:** add a block under `domains:` with `label`, `weight`, `keywords`, `mesh` and (for PubMed precision) `literature_groups`. It is tagged automatically, appears as a tab and chip on the page, and is included in the journal-club rotation, the RePORTER/trials queries and the LLM's domain list. Add `institute_domains` entries under `streams.grants` if an NIH institute implies it.
* **Add a stream:** one module in `litsearch/streams/` with a `Stream` subclass (copy `trials.py`; for flat item lists extend `ItemStream` and override `fetch`, `score`, `triage_spec`), register it in `streams/__init__.py`, add a `streams:` block, and add its label/cadence to `STREAMS` in `core/pagedata.py` plus a card renderer in `web/app.js`. Then `lit gen-systemd` creates its timer.
* **Change a schedule:** edit `schedule:` (`05:30` or `sun 06:15`), then `lit gen-systemd` and the printed `sudo` commands.
* **Different model for a stream:** set `model_env:` in its block, then `LLM_MODEL_<STREAM>=...` **and** `PRICE_<STREAM>_*` in `.env`. The program refuses to run a stream on a different model without its prices, because the budget guard would silently under-count.

### Reading the spend log

Combined cap `DAILY_BUDGET_USD` (all streams), optional `daily_cap_usd` per stream in `interests.yaml`, and `WEEKLY_EXTRA_USD` for the journal club (which does not use the daily cap).

```bash
sqlite3 data/seen.db "select day, stream, calls, round(cost_usd,5) from spend order by day desc, stream limit 30"
sqlite3 data/seen.db "select day, round(sum(cost_usd),4) from spend group by day order by day desc limit 14"
sqlite3 data/seen.db "select stream, finished, ok, error from runs order by id desc limit 15"
```
The same numbers are in the page's status banner. If a cap trips, that stream shows its items keyword-ranked without LLM summaries, and the banner says so.

## Delivery: web page over Tailscale (default)

`DELIVERY=web` (the default in `.env`) writes each day's digest to `site/YYYY-MM-DD.html` (a second run the same
day becomes `-2`, so nothing is overwritten), refreshes `site/index.html` (newest first, failures in red), and keeps
90 days. A tiny static server on `127.0.0.1:8081` serves `site/`, and `tailscale serve` exposes it to your tailnet
only. Nothing is reachable from the LAN or the internet.

One-time setup (needs sudo):

```bash
sudo cp ~/litsearch/systemd/litsearch-web.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now litsearch-web.service
sudo tailscale serve --bg 8081
tailscale serve status                         # prints the URL
```

`tailscale serve` uses HTTPS on port 443 and may ask you to enable HTTPS certificates for your tailnet in the
Tailscale admin console (DNS page); it prints a link if so. To skip that, use plain HTTP, which is still carried
inside the encrypted tailnet: `sudo tailscale serve --bg --http=80 8081`, then open `http://<this-pi's-tailscale-name>/`.

Open the URL on your phone (Tailscale on) and add it to your home screen. This Pi's Tailscale name is shown by
`tailscale status` (first line). To also get an email, set `DELIVERY=both` and fill in the SMTP block.

## Email delivery (optional): SMTP

Only needed for `DELIVERY=email` or `both`. Gmail app passwords may not be available on every account (they need
2-Step Verification and can be disabled by a Workspace admin); any SMTP relay (Brevo, SMTP2GO, institutional SMTP) works
the same way with different `.env` values.

**Gmail:**

1. Turn on 2-Step Verification for the Google account (<https://myaccount.google.com/security>).
2. Go to <https://myaccount.google.com/apppasswords>, create an app password named "litsearch", and copy the
   16-character code.
3. In `.env`: `SMTP_HOST=smtp.gmail.com`, `SMTP_PORT=587`, `SMTP_SECURITY=starttls`, `SMTP_USER=you@gmail.com`,
   `SMTP_PASSWORD=<the 16 characters>`, `MAIL_FROM=you@gmail.com`, `MAIL_TO=you@gmail.com`.

For institutional SMTP, use the host/port your IT office gives you (port 465 with `SMTP_SECURITY=ssl`, or 587 with
`starttls`). Collapsible abstracts use `<details>`: Apple Mail, Outlook for Mac and most webmail honor it; some
clients (including Gmail) show every abstract expanded. The plain-text part always has the full abstracts.

## Anthropic: API key and a workspace spend limit (second safety net)

The in-program guard stops LLM calls before a call that could push the day past `DAILY_BUDGET_USD`, but a bug or a
leaked key shouldn't be able to run up a bill, so also set a limit on Anthropic's side:

1. In the [Anthropic Console](https://console.anthropic.com/), create a **dedicated workspace** (Settings, Workspaces)
   named e.g. "litsearch".
2. In that workspace's settings set a **monthly spend limit** (for example $3-5, about 30-100x expected use).
3. Create the API key *inside that workspace* and put it in `.env` as `ANTHROPIC_API_KEY`.

## How cost is controlled

* **Model**: `LLM_MODEL` (default `claude-haiku-5-5`) with `LLM_EFFORT=low`.
* **Prices are in `.env`** (`PRICE_*_PER_MTOK`), never in code. Cost is computed from `usage` on every response
  (input, output, cache-write, cache-read tokens) and logged per day in `data/seen.db` (`spend` table).
* **Guard**: before each call, worst-case cost (input estimate plus the full `max_tokens` output) is added to
  today's spend; if that would exceed the budget, no call is made. Remaining items are emailed keyword-ranked with a
  note. The guard counts spend across all runs in a day, including retries and `--backfill`.
* **Caching**: the profile and feedback examples are a cached system prompt shared by the day's batches. It engages
  (about 1K tokens cached; calls 2-4 show `cache_r=995`), though at this size the saving is tiny.
* Measured first real run (60 candidates, 4 calls): 35K input and 8K output tokens, $0.0077. The dry-run estimate
  ($0.006) runs a little low because real output is about 120 tokens per paper, not 90. Output tokens dominate the
  cost; the daily budget is about 13x what a normal day uses. Check `grep "LLM call" logs/litsearch.log`.

Inspect spend any time:
`sqlite3 data/seen.db "select day,calls,input_tokens,output_tokens,round(cost_usd,5) from spend order by day desc limit 14"`

## Install the timers

`lit gen-systemd` writes one service + timer per enabled, scheduled stream into `systemd/` (using your username and this folder),
plus the web server unit, and prints the exact `sudo cp ... && systemctl enable --now ...` commands. Schedules come from the
`schedule:` field of each stream in `config/interests.yaml` (`HH:MM`, `ddd HH:MM` or `ddd-ddd HH:MM`, in the timezone set in
`systemd.py`). Streams marked `on_demand: true` (the podcast) get no timer.

* `Persistent=true`: if the machine was off at the scheduled time, the run happens as soon as it is back up.
* Trigger a run now: `sudo systemctl start litsearch-<stream>.service`; follow it with `journalctl -u litsearch-<stream> -f`.
* Logs: `logs/litsearch.log` (rotates at 1 MB, 5 backups) and the journal.

## When Anthropic retires the model

Deprecations are announced on <https://platform.claude.com/docs/en/about-claude/model-deprecations>. When
`claude-haiku-5-5` is retired (or you want a newer model):

1. Pick the replacement ID from <https://platform.claude.com/docs/en/about-claude/models/overview>.
2. Edit `.env`: set `LLM_MODEL=<new id>` and update all four `PRICE_*_PER_MTOK` values to the new model's prices.
3. `lit --dry-run` to confirm the estimate, then let the next scheduled run (or `lit`) confirm the live call. If the
   API rejects `LLM_EFFORT` for the new model, clear it (`LLM_EFFORT=`).

An unknown or retired model makes the API call fail; the run then falls back to keyword-ranked items and the footer
lists the API error, so you'll notice the same morning.

## How dedup works

`data/seen.db` records every paper emailed (`sent`), every paper the LLM scored below threshold (`screened`, so
you're not charged twice for it during the 2-day overlap), and errata/retractions about papers you were sent
(`notice`). A paper matches on PMID, DOI, arXiv ID, or normalised title, and preprints are linked to their published
versions through the DOI that medRxiv/arXiv report (`published`) and through title matching. A published version
of a preprint you were already sent is not emailed again. Papers are recorded only after the email is sent.
Items that were not triaged (over the cap, parse failures, budget cut-off) are not recorded, so they can come back
within the overlap window.

## Layout

```
config/interests.yaml   v2: profile, domains, people_to_watch, per-stream settings  |  .env  secrets (600)
litsearch/core/         config, db (unified items), budget, triage, domains, pipeline runner, pagedata, systemd
litsearch/streams/      literature, trials, grants, reporter, journal_club (one plugin each)
litsearch/sources/      pubmed, medrxiv, arxiv, clinicaltrials, grantsgov, nihguide, reporter
web/                    the static page (copied to site/ on every run)
templates/              digest/index pages, text digest           |  systemd/  digest service + timer, web service
tests/                  pytest suite (queries, dedup, budget guard, JSON parsing, pipeline)
data/seen.db  logs/  out/ (dry runs)  site/ (served pages)   created at runtime (gitignored)
```


## Audio briefing (podcast)

* **Setup:** `pip install piper-tts` (in `requirements.txt`), `ffmpeg`, and the two voice files in `data/voices/` (`python -m piper.download_voices en_GB-alan-medium en_US-hfc_female-medium` run from that folder). Add the `LLM_MODEL_PODCAST` / `PRICE_PODCAST_*` lines from `.env.example`. No API key is needed for the voices.
* **Hosts and voices** are in the `podcast:` block of `config/interests.yaml` (`hosts: A/B` with `name`, `model`, `length_scale`; a smaller `length_scale` speaks faster). Both voices must be the same quality tier (same sample rate).
* **Length:** `target_minutes` (10) and `monday_extra_minutes` (2) times `words_per_minute` (175, measured for these voices). A thin day gives a shorter episode.
* **Pronunciation:** `config/pronunciation.yaml` maps written forms to spoken ones (whole words, case-sensitive); it affects audio only. Add a line whenever something sounds wrong.
* **On demand, with a button:** there is no timer. Tap **🎙 Generate today's episode** at the top of the page (or **↻ Regenerate** if today's already exists); it asks for confirmation because each episode costs real money, runs in the background (about 3 to 4 minutes; you can close the page) and the player appears when it finishes. It needs today's daily briefing, which the 04:45 timer still writes. The web server starts the run (`POST /api/podcast/generate`, API version 3); a second tap while one is running does nothing.
* **Its own budget:** the podcast has `separate_budget: true` and `daily_cap_usd` (0.50): its spend is limited by that alone, does not count against `DAILY_BUDGET_USD`, and the hub's other streams do not count against it. The status panel shows it separately. Each run is priced worst-case first, so a run is refused when less than about $0.22 of the podcast's day remains.
* **Commands:** `lit --stream podcast --dry-run` (stub script, no model call, no audio); `lit --stream podcast` (real run; does nothing if today's episode exists); `lit --stream podcast --redo` replaces today's episode; `lit --stream podcast --force` re-renders today's stored script if only the audio step failed (no new model call).
* **Playing on a phone:** the page player works with the screen locked; the web server answers byte-range requests (needed by iOS).

## License

MIT, see `LICENSE`. Third-party software is installed separately and keeps its own licenses (notably Piper, ffmpeg and the Piper voice
files, whose licenses vary by voice: check the one you download).
