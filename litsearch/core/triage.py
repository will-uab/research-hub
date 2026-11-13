"""LLM triage shared by all streams: batches, strict-JSON parse, retry once, guarded by the budget."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .budget import Budget, Prices

log = logging.getLogger(__name__)

MAX_TOKENS = 4096
EXPECTED_OUT_TOKENS_PER_ITEM = 120  # measured on the first real run (incl. low-effort thinking)


class ParseError(ValueError):
    pass


@dataclass
class TriageSpec:
    """What a stream supplies: its system prompt and how each item is shown to the model."""
    system_prompt: str
    format_item: Callable[[Any, str], str]      # (item, "p3") -> text block
    header: str = "Items:"
    tracks: frozenset[str] | None = None        # allowed `track` values; None = stream has no tracks
    domain_ids: tuple[str, ...] = ()
    expected_out_tokens: int = EXPECTED_OUT_TOKENS_PER_ITEM


def truncate_words(text: str, n: int) -> str:
    words = (text or "").split()
    return " ".join(words) if len(words) <= n else " ".join(words[:n]) + " …"


def feedback_block(feedback: list[tuple[int, str]]) -> str:
    if not feedback:
        return ""
    liked = [t for s, t in feedback if s > 0]
    disliked = [t for s, t in feedback if s < 0]
    fb = "\n\nRecent reader feedback (calibrate to these):\n"
    if liked:
        fb += "Liked:\n" + "\n".join(f"- {t}" for t in liked) + "\n"
    if disliked:
        fb += "Disliked:\n" + "\n".join(f"- {t}" for t in disliked) + "\n"
    return fb


def format_batch(items: list, ids: list[str], spec: TriageSpec) -> str:
    return spec.header + "\n\n" + "\n\n".join(spec.format_item(it, i) for i, it in zip(ids, items))


# ------------------------------------------------------------------ response parsing
def parse_response(text: str, expected_ids: list[str], tracks: frozenset[str] | None = None,
                   domain_ids: tuple[str, ...] = ()) -> dict[str, dict]:
    """Return {id: validated item}. Raises ParseError if no usable JSON array is found.

    Items for unknown ids are ignored; invalid items are dropped (caller sees them as missing).
    `domains` is optional and filtered to known ids (the model confirms/adds domains in the same call).
    """
    s = (text or "").strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    data = None
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        i, j = s.find("["), s.rfind("]")
        if i != -1 and j > i:
            try:
                data = json.loads(s[i:j + 1])
            except json.JSONDecodeError:
                pass
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                data = v
                break
    if not isinstance(data, list):
        raise ParseError("no JSON array in model output")

    out: dict[str, dict] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        pid = str(item.get("id", "")).strip()
        if pid not in expected_ids:
            continue
        try:
            rel = float(item["relevance"])
        except (KeyError, TypeError, ValueError):
            continue
        track = None
        if tracks is not None:
            track = str(item.get("track", "")).strip().lower()
            if track not in tracks:
                continue
        raw_domains = item.get("domains") or []
        doms = [d for d in domain_ids if d in {str(x).strip() for x in raw_domains}] if isinstance(raw_domains, list) else []
        out[pid] = {
            "relevance": max(0.0, min(10.0, rel)),
            "track": track,
            "takeaway": str(item.get("one_sentence_takeaway", "")).strip(),
            "fit_note": str(item.get("why_relevant", "")).strip(),
            "domains": doms,
        }
    return out


# ------------------------------------------------------------------ triage run
@dataclass
class TriageResult:
    scored: list = field(default_factory=list)
    parse_failed: list = field(default_factory=list)   # skipped after one retry
    unscored: list = field(default_factory=list)       # budget guard / API outage
    errors: list[str] = field(default_factory=list)
    calls: int = 0


def _text_of(resp) -> str:
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


def usage_dict(resp) -> dict:
    u = resp.usage
    return {
        "input_tokens": u.input_tokens or 0,
        "output_tokens": u.output_tokens or 0,
        "cache_write_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
        "cache_read_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
    }


def complete(client, model: str, effort: str, budget: Budget, system: str, user_text: str,
             max_tokens: int = MAX_TOKENS) -> tuple[str, str] | None:
    """One budget-guarded Messages call. Returns (text, stop_reason), or None if the guard blocked it."""
    worst = budget.prices.estimate(len(system) + len(user_text), max_tokens)
    if not budget.allow(worst):
        return None
    kwargs = {}
    if effort:
        kwargs["output_config"] = {"effort": effort}
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_text}],
        **kwargs,
    )
    u = usage_dict(resp)
    cost = budget.record(u)
    log.info("LLM call [%s]: in=%d out=%d cache_w=%d cache_r=%d cost=$%.5f stop=%s",
             budget.stream, u["input_tokens"], u["output_tokens"], u["cache_write_tokens"],
             u["cache_read_tokens"], cost, resp.stop_reason)
    return _text_of(resp), resp.stop_reason


class Triager:
    def __init__(self, client, model: str, effort: str, budget: Budget, spec: TriageSpec, batch_size: int):
        self.client, self.model, self.effort = client, model, effort
        self.budget, self.spec, self.batch_size = budget, spec, batch_size

    def _call(self, user_text: str) -> tuple[str, str] | None:
        return complete(self.client, self.model, self.effort, self.budget, self.spec.system_prompt, user_text)

    def _score_batch(self, items: list, ids: list[str], res: TriageResult) -> bool:
        """Returns False if the guard/outage stopped us (items left in res.unscored)."""
        pending = dict(zip(ids, items))
        for attempt in (1, 2):                      # one retry on malformed/partial output
            sub_ids = list(pending)
            user = format_batch(list(pending.values()), sub_ids, self.spec)
            try:
                got = self._call(user)
            except Exception as e:  # noqa: BLE001 - anthropic.APIError and friends
                log.exception("Anthropic API call failed")
                res.errors.append(f"Anthropic API: {type(e).__name__}: {e}")
                res.unscored.extend(pending.values())
                return False
            res.calls += 1 if got else 0
            if got is None:
                res.unscored.extend(pending.values())
                return False
            text, stop = got
            try:
                if stop in ("refusal", "max_tokens"):
                    raise ParseError(f"stop_reason={stop}")
                parsed = parse_response(text, sub_ids, self.spec.tracks, self.spec.domain_ids)
            except ParseError as e:
                log.warning("triage attempt %d unparseable: %s", attempt, e)
                parsed = {}
            for pid, r in parsed.items():
                it = pending.pop(pid)
                it.relevance, it.track = r["relevance"], r["track"]
                it.takeaway, it.fit_note, it.llm_scored = r["takeaway"], r["fit_note"], True
                if r["domains"]:                    # LLM confirms/adds; empty -> keep keyword tags
                    it.domains = r["domains"]
                res.scored.append(it)
            if not pending:
                return True
        res.parse_failed.extend(pending.values())
        res.errors.append(f"{len(pending)} item(s) skipped: model output not valid JSON after retry")
        return True

    def run(self, candidates: list) -> TriageResult:
        res = TriageResult()
        ordered = sorted(candidates, key=lambda p: p.kw_score, reverse=True)
        for i in range(0, len(ordered), self.batch_size):
            chunk = ordered[i:i + self.batch_size]
            ids = [f"p{n + 1}" for n in range(len(chunk))]
            if not self._score_batch(chunk, ids, res):
                res.unscored.extend(ordered[i + self.batch_size:])
                break
        return res


# ------------------------------------------------------------------ dry-run stub + estimate
def stub_triage(candidates: list, track_fn: Callable[[Any], str | None] | None = None) -> TriageResult:
    """Offline stand-in: keyword score -> relevance. No API call."""
    res = TriageResult()
    for p in candidates:
        p.track = track_fn(p) if track_fn else None
        p.relevance = min(10.0, round(p.kw_score / 2, 1))
        if p.track == "major":
            p.relevance = max(p.relevance, 6.0)
        p.takeaway = "[dry-run stub] Takeaway would be generated by the LLM."
        if hasattr(p, "matched"):           # literature papers
            why = ", ".join(p.matched) or "major journal"
        else:                               # other streams
            why = ", ".join(p.domains) or "domain keywords"
        p.fit_note = "keyword match: " + why
        p.llm_scored = True
        res.scored.append(p)
    return res


def estimate_cost(spec: TriageSpec, candidates: list, prices: Prices, batch_size: int,
                  chars_per_token: float = 3.5) -> dict:
    """Expected (not worst-case) cost for triaging these candidates."""
    ordered = sorted(candidates, key=lambda p: p.kw_score, reverse=True)
    in_tok = out_tok = 0.0
    calls = 0
    for i in range(0, len(ordered), batch_size):
        chunk = ordered[i:i + batch_size]
        user = format_batch(chunk, [f"p{n + 1}" for n in range(len(chunk))], spec)
        in_tok += (len(spec.system_prompt) + len(user)) / chars_per_token
        out_tok += spec.expected_out_tokens * len(chunk)
        calls += 1
    usage = {"input_tokens": int(in_tok), "output_tokens": int(out_tok)}
    return {"calls": calls, "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"], "cost_usd": prices.cost(usage)}
