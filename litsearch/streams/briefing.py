"""Briefing stream: a curated, paragraph-form daily summary (and a weekly trends piece on Mondays), written by Claude.

Runs after the other streams. It reads what they stored, never the web, so every claim must trace to an item it was
given: the model cites items as [iID] and anything it cites that it was not given is stripped.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from datetime import datetime, timedelta, timezone

from ..core import learned, triage
from ..core.models import Item
from ..core.triage import ParseError, feedback_block, truncate_words
from .base import Stream

log = logging.getLogger(__name__)
MAX_TOKENS = 3500
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
EVENT_KINDS = ("nofo_modified", "notice", "reissued", "status_change", "results_posted")
STREAM_NAMES = {"literature": "literature", "people": "people you follow", "grants": "grant", "reporter": "award",
                "trials": "trial", "journal_club": "journal club"}
CITE = re.compile(r"\[i(\d+)\]")


def system_prompt(profile: str, domains: dict[str, str], period: str, feedback) -> str:
    dom = "; ".join(f"{k} ({v})" for k, v in domains.items())
    shape = ("3 to 5 sections: 'At a glance' (one or two sentences on the overall shape of the day), then one section per theme actually "
             "present in the items (headings you choose, e.g. 'LLM methods', 'HIV and addiction', 'Trials and funding'), and a closing "
             "'Worth a look first' section whose `bullets` name 2 to 4 specific items, one line each" if period == "daily" else
             "4 to 6 sections: 'The week in brief', 'Trends' (compare VOLUME with the previous week and say where several streams point the "
             "same way, e.g. new trials, new funding and new papers on one topic), 'People you follow', 'Funding and deadlines', and a "
             "closing 'What to watch' section with `bullets`; omit any section the items do not support")
    return f"""You write a {period} research briefing for one researcher: organized into short titled sections of readable prose, with an occasional bullet list where a list genuinely helps.

RESEARCHER PROFILE
{profile.strip()}

WHAT YOU RECEIVE
Items collected by an automated pipeline (papers, preprints, trials, grant announcements, NIH awards, papers by people the researcher follows), each with an id like [i123], plus event notices and volume counts. Domains: {dom}.

TASK
Write the briefing: {shape}. Each section has a short heading (at most 6 words), up to 2 paragraphs of at most about 90 words each (lead with the point), and optionally `bullets` (one-line items). Cite the items you rely on by writing their ids in square brackets right after the claim, e.g. "...a cohort study of PrEP uptake [i123]". You may wrap a few key terms in **double asterisks** for emphasis (at most 3 per section); no other markup.
Rules:
- Use ONLY the supplied items, events and counts. Never add outside facts, numbers, results or items. If something is thin, say so.
- Each item's date is its publication, registration or award date, not the day the pipeline found it: items can be days or weeks old, so never call an older item "today's" or "new today"; say how recent it is when that matters.
- Items are screened from abstracts or registry entries, not full papers: describe findings as "reports" or "finds" and mark preprints as preprints. No clinical advice.
- Give the AI/LLM-methods interests the most attention, but do not ignore HIV, addiction and infectious-disease items that matter.
- Prefer synthesis (what the items together suggest) over a list of titles. Name people the researcher follows when they appear.{feedback_block(feedback)}

OUTPUT
Respond with ONLY a JSON object, no code fences:
{{"headline": "at most 12 words", "sections": [{{"heading": "At a glance", "paragraphs": ["..."], "bullets": []}}, {{"heading": "...", "paragraphs": ["...", "..."]}}, {{"heading": "Worth a look first", "paragraphs": [], "bullets": ["... [i123]", "... [i456]"]}}]}}"""


def parse_briefing(text: str, valid_ids: set[int]) -> dict:
    """Validate the model's JSON and keep only citations to items it was actually given.

    Accepts the structured shape {"sections": [{"heading", "paragraphs", "bullets"}]} and the older flat {"paragraphs": [...]}.
    """
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    data = None
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        i, j = s.find("{"), s.rfind("}")
        if i != -1 and j > i:
            try:
                data = json.loads(s[i:j + 1])
            except json.JSONDecodeError:
                pass
    if not isinstance(data, dict):
        raise ParseError("no JSON object")
    raw = data.get("sections")
    if raw is None and isinstance(data.get("paragraphs"), list):          # older flat shape -> one untitled section
        raw = [{"heading": "", "paragraphs": data["paragraphs"]}]
    if not isinstance(raw, list):
        raise ParseError("no sections or paragraphs")
    refs: list[int] = []

    def keep(m):
        n = int(m.group(1))
        if n not in valid_ids:
            return ""                                           # a citation to something it was not given
        if n not in refs:
            refs.append(n)
        return m.group(0)

    def clean(t: str) -> str:
        return re.sub(r"\s{2,}", " ", CITE.sub(keep, str(t))).replace(" .", ".").replace(" ,", ",").strip()

    sections = []
    for sec in raw:
        if not isinstance(sec, dict):
            continue
        paras = [clean(p) for p in (sec.get("paragraphs") or []) if str(p).strip()] if isinstance(sec.get("paragraphs"), list) else []
        bullets = [clean(b) for b in (sec.get("bullets") or []) if str(b).strip()] if isinstance(sec.get("bullets"), list) else []
        if not paras and not bullets:
            continue
        sections.append({"heading": clean(sec.get("heading", ""))[:70], "paragraphs": paras[:3], "bullets": bullets[:6]})
    body = sum(len(p) for sec in sections for p in sec["paragraphs"]) + sum(len(b) for sec in sections for b in sec["bullets"])
    if not 1 <= len(sections) <= 8 or body < 80:
        raise ParseError(f"expected 1-8 substantial sections, got {len(sections)} ({body} chars)")
    flat = [p for sec in sections for p in sec["paragraphs"]]       # kept for consumers that only know paragraphs
    return {"headline": str(data.get("headline", "")).strip()[:140], "sections": sections, "paragraphs": flat, "refs": refs}


class BriefingStream(Stream):
    name = "briefing"

    # ---------------------------------------------------------------- gathering
    def _rows(self, since: datetime, until: datetime | None = None) -> list[dict]:
        db = self.ctx.db
        out = []
        for r in db.conn.execute("SELECT * FROM items WHERE status IN ('sent','shown') AND stream NOT IN ('briefing','podcast') AND first_seen >= ?",
                                 (since.isoformat(),)):
            if until and r["first_seen"] >= until.isoformat():
                continue
            ss = json.loads(r["stream_specific"])
            if ss.get("migrated_from_v1") or ss.get("kind") == "weekly_summary":
                continue
            if r["stream"] == "journal_club" and ss.get("role") != "primary":
                continue
            out.append({"id": r["id"], "stream": r["stream"], "title": r["title"], "domains": json.loads(r["domains"]),
                        "relevance": r["relevance"], "takeaway": r["takeaway"] or "", "ss": ss, "date": r["date"] or ""})
        return out

    def _worth(self, r: dict) -> float:
        ss, rel = r["ss"], r["relevance"]
        s = rel if rel is not None else 3
        if ss.get("watched") or ss.get("watched_authors") or r["stream"] in ("people", "journal_club"):
            s += 3
        if ss.get("has_results") or ss.get("ai_intervention"):
            s += 1
        return s

    def select(self, rows: list[dict]) -> list[dict]:
        cfg = self.cfg
        min_rel, cap = cfg.get("min_relevance", 5), cfg.get("max_items", 60)
        keep = [r for r in rows if r["relevance"] is None or r["relevance"] >= min_rel or r["stream"] in ("people", "journal_club")
                or r["ss"].get("watched")]
        keep.sort(key=lambda r: -self._worth(r))
        per_stream, out = Counter(), []
        for r in keep:                                              # no single stream may crowd out the rest
            if per_stream[r["stream"]] >= max(6, cap // 3):
                continue
            per_stream[r["stream"]] += 1
            out.append(r)
        return out[:cap]

    @staticmethod
    def line(r: dict, labels: dict[str, str]) -> str:
        ss, st = r["ss"], r["stream"]
        extra = ""
        if st == "grants":
            extra = f" ({ss.get('number') or 'n/a'}; {', '.join(ss.get('institutes', []))}; {ss.get('mechanism') or 'n/a'}; due {ss.get('close_date') or 'n/a'})"
        elif st == "reporter":
            extra = f" ({ss.get('contact_pi')}, {ss.get('institution')}; ${ss.get('amount') or 0:,.0f}{'; FOLLOWED: ' + '/'.join(ss['watched']) if ss.get('watched') else ''})"
        elif st == "trials":
            extra = f" ({ss.get('status')}{'; AI intervention' if ss.get('ai_intervention') else ''}{'; results posted' if ss.get('has_results') else ''}; {ss.get('sponsor')})"
        elif st == "people":
            extra = f" (by {', '.join(ss.get('watched', []))}; {ss.get('journal') or ss.get('type')}{'; preprint' if ss.get('is_preprint') else ''})"
        elif st == "literature":
            extra = f" ({ss.get('source_label') or ss.get('journal') or ''}{'; preprint' if ss.get('is_preprint') else ''}{'; major journal' if ss.get('is_major') else ''}{'; by followed: ' + '/'.join(ss['watched_authors']) if ss.get('watched_authors') else ''})"
        doms = "/".join(labels.get(d, d) for d in r["domains"]) or "-"
        rel = f"rel {r['relevance']:.0f}" if r["relevance"] is not None else "unscored"
        take = truncate_words(r["takeaway"], 40)
        when = f" | {r['date'][:10]}" if r.get("date") else ""
        return f"[i{r['id']}] {STREAM_NAMES.get(st, st)} | {doms} | {rel}{when} | {r['title']}{extra}{' - ' + take if take else ''}"

    def events(self, since: datetime) -> list[str]:
        db, out = self.ctx.db, []
        titles = {r["id"]: r for r in db.conn.execute("SELECT id, title, stream FROM items")}
        for e in db.events(since.isoformat(), EVENT_KINDS):
            it = titles.get(e["item_id"])
            if not it:
                continue
            d = json.loads(e["detail"])
            what = {"status_change": f"status {d.get('from')} -> {d.get('to')}", "results_posted": "results posted",
                    "nofo_modified": "announcement modified" + (f": {d['comments']}" if d.get("comments") else ""),
                    "notice": f"NIH notice {d.get('notice')}", "reissued": f"reissued as {d.get('new_number')}"}[e["kind"]]
            out.append(f"EVENT [i{e['item_id']}] {it['title'][:90]}: {what}")
        return out[:15]

    def volume(self, rows: list[dict], prev: list[dict], labels: dict[str, str]) -> str:
        sc, sp = Counter(r["stream"] for r in rows), Counter(r["stream"] for r in prev)
        dc, dp = Counter(d for r in rows for d in r["domains"]), Counter(d for r in prev for d in r["domains"])
        a = "; ".join(f"{STREAM_NAMES.get(k, k)} {sc[k]} (before: {sp[k]})" for k in sorted(set(sc) | set(sp)))
        b = "; ".join(f"{labels.get(k, k)} {dc[k]} (before: {dp[k]})" for k in sorted(set(dc) | set(dp)))
        return f"VOLUME this period vs previous period - by stream: {a or 'none'}. By domain: {b or 'none'}."

    # ---------------------------------------------------------------- generation
    def period_start(self, period: str, now: datetime) -> datetime:
        """Start of the window: when the previous briefing of this kind was written (so consecutive briefings never
        repeat items), but never further back than twice the nominal length; the nominal length on the first run."""
        hours = self.cfg.get("lookback_hours", 26) if period == "daily" else 24 * 7
        row = self.ctx.db.conn.execute("SELECT first_seen FROM items WHERE stream='briefing' AND source_id LIKE ? ORDER BY id DESC LIMIT 1",
                                       (f"{period}:%",)).fetchone()
        if row:
            try:
                return max(datetime.fromisoformat(row["first_seen"]), now - timedelta(hours=2 * hours))
            except ValueError:
                pass
        return now - timedelta(hours=hours)

    def build(self, period: str, key: str, now: datetime) -> tuple[str, set[int], dict]:
        since = self.period_start(period, now)
        hours = max(1, round((now - since).total_seconds() / 3600))
        prev_since = since - (now - since)                       # an equally long stretch before it, for "vs previous"
        labels = self.ctx.tagger.labels
        rows = self._rows(since)
        prev = self._rows(prev_since, since)
        chosen = self.select(rows)
        user = "\n".join([f"PERIOD: {period} briefing {key}, covering the {hours} hours since the previous {period} briefing.", self.volume(rows, prev, labels), "",
                          "ITEMS:", *[self.line(r, labels) for r in chosen], "", *self.events(since)])
        counts = {"items_total": len(rows), "items_given": len(chosen), "events": len(self.events(since)), "hours": hours,
                  "by_stream": dict(Counter(r["stream"] for r in rows))}
        return user, {r["id"] for r in chosen} | {int(x) for x in re.findall(r"EVENT \[i(\d+)\]", user)}, counts

    def stub(self, chosen_ids: set[int], period: str) -> dict:
        ids = sorted(chosen_ids)[:4]
        cite = lambda i: f" [i{ids[i]}]" if i < len(ids) else ""
        sections = [{"heading": "At a glance", "paragraphs": [f"[dry-run stub] One or two sentences on the overall shape of the {period}{cite(0)}."], "bullets": []},
                    {"heading": "A theme (stub)", "paragraphs": [f"[dry-run stub] A paragraph pulling out a theme that several items share{cite(1)}, with a **key term** emphasised.",
                                                                  f"[dry-run stub] A second, shorter paragraph{cite(2)}."], "bullets": []},
                    {"heading": "Worth a look first", "paragraphs": [], "bullets": [f"[dry-run stub] One-line pointer{cite(0)}", f"[dry-run stub] Another pointer{cite(3)}"]}]
        return {"headline": f"[dry-run stub] {period.title()} briefing would be written by Claude here", "sections": sections,
                "paragraphs": [p for sec in sections for p in sec["paragraphs"]], "refs": ids}

    def generate(self, period: str, key: str, now) -> tuple[dict | None, dict, str]:
        ctx = self.ctx
        user, valid, counts = self.build(period, key, now)
        if counts["items_given"] < self.cfg.get("min_items", 3):
            return None, counts, f"quiet {period}: only {counts['items_given']} item(s) to summarise"
        if ctx.opts.dry_run:
            return self.stub(valid, period), counts, ""
        import anthropic
        client = anthropic.Anthropic(api_key=ctx.settings.get("ANTHROPIC_API_KEY"), max_retries=3, timeout=180)
        system = learned.insert_into_prompt(system_prompt(ctx.cfg["profile"], ctx.tagger.labels, period, ctx.db.recent_feedback(10)), self.learned)
        model, effort = ctx.settings.model_for(self.cfg), ctx.settings.effort_for(self.name)
        for attempt in (1, 2):
            try:
                got = triage.complete(client, model, effort, ctx.budget, system, user, MAX_TOKENS)
            except Exception as e:  # noqa: BLE001
                log.exception("briefing call failed")
                return None, counts, f"Anthropic API: {type(e).__name__}: {e}"
            if got is None:
                return None, counts, f"{period} briefing skipped: budget guard ({ctx.budget.trip_reason})"
            text, stop = got
            try:
                if stop in ("refusal", "max_tokens"):
                    raise ParseError(f"stop_reason={stop}")
                return parse_briefing(text, valid), counts, ""
            except ParseError as e:
                log.warning("briefing attempt %d unparseable: %s", attempt, e)
        return None, counts, f"{period} briefing skipped: model output was not valid after one retry"

    def execute(self) -> int:
        ctx, db = self.ctx, self.ctx.db
        today, now = ctx.today, datetime.now(timezone.utc)
        iso = today.isocalendar()
        periods = [("daily", f"daily:{today.isoformat()}")]
        if WEEKDAYS[today.weekday()] == str(self.cfg.get("weekly_day", "monday")).lower():
            periods.append(("weekly", f"weekly:{iso.year}-W{iso.week:02d}"))
        model = ctx.settings.model_for(self.cfg)
        for period, key in periods:
            if db.find(Item("briefing", "claude", key, "")) is not None:
                if ctx.opts.dry_run:
                    print(f"\nDry run [briefing] {key}: already written; nothing to preview.")
                else:
                    ctx.notes.append(f"briefing: {key} already exists")
                continue
            out, counts, note = self.generate(period, key, now)
            if note:
                ctx.notes.append(f"briefing: {note}")
            ctx.counts[period] = counts["items_given"]
            if out is None:
                if ctx.opts.dry_run:
                    print(f"\nDry run [briefing] {key}: {note}")
                continue
            label = today.strftime("%a %b %-d") if period == "daily" else f"week of {(today - timedelta(days=today.weekday())).strftime('%b %-d')}"
            item = Item(stream="briefing", source="claude", source_id=key, title=f"{'Daily briefing' if period == 'daily' else 'Weekly trends'} · {label}",
                        date=today.isoformat(), summary="\n\n".join((f"## {sec['heading']}\n" if sec["heading"] else "") + "\n".join(sec["paragraphs"] + ["- " + b for b in sec["bullets"]]) for sec in out["sections"]), takeaway=out["headline"],
                        stream_specific={"kind": "briefing", "period": period, "date": today.isoformat(), "headline": out["headline"],
                                         "paragraphs": out["paragraphs"], "sections": out.get("sections", []), "refs": out["refs"], "model": model if not ctx.opts.dry_run else "dry-run stub",
                                         "covers": {"hours": counts.get("hours")},
                                         "counts": counts})
            db.record(item, "shown")
            if ctx.opts.dry_run:
                print(f"\nDry run [briefing] {key}: {counts['items_given']} of {counts['items_total']} items given to the model\n  {out['headline']}")
                for sec in out["sections"]:
                    print(f"   ## {sec['heading']}")
                    for p in sec["paragraphs"]:
                        print("      " + p[:140])
                    for b in sec["bullets"]:
                        print("      * " + b[:140])
            else:
                log.info("[briefing] %s written (%d items, %d citations); cost so far $%.5f", key, counts["items_given"], len(out["refs"]), ctx.budget.run_cost)
        return 0
