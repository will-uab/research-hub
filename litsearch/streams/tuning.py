"""Tuning stream: every N new ratings, propose edits to your learned preferences (you approve each one on the page)."""
from __future__ import annotations

import json
import logging
import re

from ..core import learned, triage, tuning
from ..core.triage import ParseError
from .base import Stream

log = logging.getLogger(__name__)
MAX_TOKENS = 3500
MAX_PREF_CHARS = 1400
SIGN = {1: "liked", -1: "disliked"}


def system_prompt(profile: str) -> str:
    return f"""You maintain a short, durable preference profile for one researcher, from their likes and dislikes of items an automated pipeline showed them.

RESEARCHER'S OWN PROFILE (hand-written; it always takes precedence)
{profile.strip()}

YOU RECEIVE
The current learned paragraph and rules; the rated items (their tap, an optional reason, and the pipeline model's 0-10 relevance score at the time); candidate rules found by counting (each with id c1, c2...); candidate people to follow (f1, f2...); and the places where the model's scores and the researcher's taps disagree.

TASK
1. preferences: rewrite the learned paragraph (at most 150 words, plain prose, describing tendencies: topics, kinds of studies and sources they value or skip). Keep what still holds, revise what the new evidence changes, never describe individual papers, never contradict the hand-written profile. Return an empty string if there is not enough evidence to say anything new.
2. rules: choose which candidate rules are worth proposing, by id, with a weight (an integer; positive = prefer, negative = deprioritize; never stronger than the candidate's suggested weight) and a one-sentence why. Skip candidates that look coincidental. The reasons matter: "not my topic" is a topic signal, "weak study" is about quality or source, "already knew it" and "too basic" say nothing about topic.
3. follow: choose which candidate people are worth suggesting, by id, with a one-sentence why.
4. notes: 2 to 4 sentences on the patterns you see and where the model's scores and the researcher's taps disagree most.
Use ONLY the supplied evidence.

OUTPUT
Respond with ONLY a JSON object, no code fences:
{{"preferences": "...", "rules": [{{"id": "c3", "weight": -2, "why": "..."}}], "follow": [{{"id": "f1", "why": "..."}}], "notes": "..."}}"""


def parse_response(text: str, cands: dict[str, dict], follows: dict[str, dict], current_pref: str) -> dict:
    """Validate against what was actually offered: ids must exist, weights are capped by the evidence, text is bounded."""
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
    pref = str(data.get("preferences") or "").strip()
    if len(pref) > MAX_PREF_CHARS:
        pref = ""                                                         # too long to be a distilled paragraph: ignore rather than truncate
    rules, seen = [], set()
    for r in (data.get("rules") or [])[:12] if isinstance(data.get("rules"), list) else []:
        c = cands.get(str(r.get("id", ""))) if isinstance(r, dict) else None
        if not c or c["id"] in seen:
            continue
        try:
            w = float(r.get("weight"))
        except (TypeError, ValueError):
            w = c["weight"]
        cap = abs(c["weight"])
        w = max(-cap, min(cap, w)) if w * c["weight"] > 0 else c["weight"]       # sign and strength follow the evidence, not the model
        seen.add(c["id"])
        rules.append({"cand": c, "weight": w, "why": str(r.get("why", "")).strip()[:220]})
    follow, fseen = [], set()
    for f in (data.get("follow") or [])[:6] if isinstance(data.get("follow"), list) else []:
        c = follows.get(str(f.get("id", ""))) if isinstance(f, dict) else None
        if c and c["id"] not in fseen:
            fseen.add(c["id"])
            follow.append({"cand": c, "why": str(f.get("why", "")).strip()[:220]})
    return {"preferences": pref if pref and pref != current_pref.strip() else "", "rules": rules, "follow": follow,
            "notes": str(data.get("notes", "")).strip()[:700]}


class TuningStream(Stream):
    name = "tuning"

    def user_prompt(self, rated, cands, follows, dis, l) -> str:
        fmt = lambda r: (f"[{r['item_id']}] {SIGN[r['sign']]} | reason: {r.get('reason') or '-'} | model score then: "
                         f"{'n/a' if r.get('relevance') is None else round(r['relevance'])} | {r['stream']} | {r['title'][:90]}"
                         f"{' - ' + r['takeaway'][:110] if r.get('takeaway') else ''}")
        rule = lambda x: f"{x['kind']}: {x['value']} ({x['weight']:+g})"
        lines = ["CURRENT LEARNED PARAGRAPH:", l["preferences"] or "(none yet)", "",
                 "CURRENT RULES: " + (", ".join(rule(x) for x in l["prefer"] + l["deprioritize"]) or "(none)"), "",
                 f"RATED ITEMS (newest last; {len(rated)} total, showing up to 60):", *[fmt(r) for r in rated[-60:]], "", "CANDIDATE RULES:"]
        lines += [f"{c['id']} | {c['kind']} | \"{c['value']}\" | {c['likes']} liked / {c['dislikes']} disliked | like-rate {c['share']} | suggested weight {c['weight']:+g}"
                  for c in cands.values()] or ["(none met the evidence thresholds)"]
        lines += ["", "CANDIDATE PEOPLE TO FOLLOW:"]
        lines += [f"{c['id']} | {c['name']} | on {c['likes']} liked paper(s){' | tagged follow-this-person' if c['tagged'] else ''}" for c in follows.values()] or ["(none)"]
        lines += ["", "WHERE THE MODEL'S SCORE AND THE TAP DISAGREE:"]
        lines += [f"[{d['item_id']}] {SIGN[d['sign']]} but model scored {round(d['llm_score'])} | reason: {d['reason'] or '-'} | {d['title']}" for d in dis] or ["(none)"]
        return "\n".join(lines)

    def execute(self) -> int:
        ctx, db, cfg = self.ctx, self.ctx.db, self.cfg
        rated = db.rated()
        last = db.latest_report()
        new = [r for r in rated if r["fid"] > (last["last_feedback_id"] if last else 0)]
        need = cfg.get("first_after", 10) if not last else cfg.get("every", 20)
        ctx.counts.update({"ratings": len(rated), "new_ratings": len(new), "needed": need})
        if len(new) < need and not (ctx.opts.force and rated):
            msg = f"{len(new)} of {need} new ratings since the {'last' if last else 'first'} tune-up; nothing to do yet"
            log.info("[tuning] %s", msg)
            ctx.notes.append(f"tuning: {msg}") if ctx.opts.dry_run else None
            if ctx.opts.dry_run:
                print(f"\nDry run [tuning]: {msg}")
            return 0
        l = learned.normalise(ctx.cfg.get("learned"))
        followed = [p["name"] for p in (ctx.cfg.get("people_to_watch") or [])]
        raw = tuning.candidate_rules(rated, l, cfg.get("min_support", 3), cfg.get("min_share", 0.7))
        cands = {f"c{i + 1}": {**c, "id": f"c{i + 1}"} for i, c in enumerate(raw)}
        follows = {f"f{i + 1}": {**c, "id": f"f{i + 1}"} for i, c in enumerate(tuning.follow_candidates(rated, followed))}
        dis = tuning.disagreements(rated)
        if ctx.opts.dry_run:
            out = {"preferences": "", "rules": [{"cand": c, "weight": c["weight"], "why": "[dry-run stub] chosen by evidence thresholds only"} for c in list(cands.values())[:4]],
                   "follow": [{"cand": c, "why": "[dry-run stub]"} for c in list(follows.values())[:2]],
                   "notes": "[dry-run stub] The tune-up's notes would appear here."}
            model = "dry-run stub"
        else:
            out, model = self.ask(rated, cands, follows, dis, l)
            if out is None:
                return 0
        self.store(db, rated, out, l, model)
        return 0

    def ask(self, rated, cands, follows, dis, l):
        ctx = self.ctx
        import anthropic
        client = anthropic.Anthropic(api_key=ctx.settings.get("ANTHROPIC_API_KEY"), max_retries=3, timeout=180)
        model, effort = ctx.settings.model_for(self.cfg), ctx.settings.effort_for(self.name)
        system, user = system_prompt(ctx.cfg["profile"]), self.user_prompt(rated, cands, follows, dis, l)
        for attempt in (1, 2):
            try:
                got = triage.complete(client, model, effort, ctx.budget, system, user, MAX_TOKENS)
            except Exception as e:  # noqa: BLE001
                log.exception("tuning call failed")
                ctx.notes.append(f"tuning: Anthropic API: {type(e).__name__}: {e}")
                return None, model
            if got is None:
                ctx.notes.append(f"tuning: skipped, budget guard ({ctx.budget.trip_reason})")
                return None, model
            text, stop = got
            try:
                if stop in ("refusal", "max_tokens"):
                    raise ParseError(f"stop_reason={stop}")
                return parse_response(text, cands, follows, l["preferences"]), model
            except ParseError as e:
                log.warning("tuning attempt %d unparseable: %s", attempt, e)
        ctx.notes.append("tuning: skipped, model output was not valid after one retry")
        return None, model

    def store(self, db, rated, out, l, model) -> None:
        ctx = self.ctx
        ratings = {r["item_id"]: r["sign"] for r in rated}
        rid = db.add_report(len(rated), max(r["fid"] for r in rated), out["notes"], model)
        n = 0
        if out["preferences"]:
            db.add_proposal(rid, "preferences", {"old": l["preferences"], "new": out["preferences"]}, "Rewritten from all your ratings.", [], {}); n += 1
        for r in out["rules"]:
            c = r["cand"]
            rule = {"kind": c["kind"], "value": c["value"], "weight": r["weight"]}
            bt = tuning.backtest(db, rule, ratings)
            db.add_proposal(rid, "rule", {**rule, "likes": c["likes"], "dislikes": c["dislikes"], "share": c["share"]}, r["why"], c["evidence"], bt); n += 1
        for f in out["follow"]:
            c = f["cand"]
            db.add_proposal(rid, "follow", {"name": c["name"], "likes": c["likes"]}, f["why"], c["evidence"], {}); n += 1
        ctx.counts.update({"proposals": n})
        log.info("[tuning] report %d: %d proposal(s) from %d ratings", rid, n, len(rated))
        if ctx.opts.dry_run:
            print(f"\nDry run [tuning]: {n} proposal(s) from {len(rated)} ratings\n  notes: {out['notes'][:160]}")
            for p in db.proposals(("pending",), 20)[::-1]:
                pl = p["payload"]
                print(f"   - {p['kind']:<11} " + (f"{pl.get('kind')}: {pl.get('value')} ({pl.get('weight'):+g}) | backtest {p['backtest']}" if p["kind"] == "rule" else str(pl)[:110]))
