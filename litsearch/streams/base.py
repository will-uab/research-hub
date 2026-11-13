"""Stream plugin base: fetch -> normalize -> dedup -> prefilter -> LLM triage -> (stream) finalize.

A new stream is one module with a Stream subclass (registered in streams/__init__.py) plus a config block under
`streams:` in interests.yaml. Override only the hooks you need; journal_club overrides `execute` entirely.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from ..core import learned, triage
from ..core.context import RunContext

log = logging.getLogger(__name__)


class RetrievalFailed(RuntimeError):
    pass


@dataclass
class RunState:
    """Everything `finalize` needs."""
    start: date
    retrieved: int = 0
    by_source: str = ""
    counts: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    fresh: list = field(default_factory=list)
    cands: list = field(default_factory=list)
    notices: list = field(default_factory=list)
    est: dict = field(default_factory=dict)
    res: triage.TriageResult = field(default_factory=triage.TriageResult)


class Stream:
    name = ""

    def __init__(self, ctx: RunContext):
        self.ctx = ctx
        self.cfg = ctx.stream_cfg

    # ------------------------------------------------------------------ hooks
    def window(self) -> tuple[date, date]:
        days = self.ctx.opts.backfill or self.cfg.get("window_days", 2)
        return self.ctx.today - timedelta(days=days), self.ctx.today

    def fetch(self, start: date, end: date) -> tuple[list, list[str], dict]:
        raise NotImplementedError

    def normalize(self, items: list) -> list:
        return items

    def split_notices(self, items: list) -> tuple[list, list]:
        return items, []

    def is_new(self, item) -> bool:
        return self.ctx.db.find(item) is None

    def score(self, items: list) -> None:
        """Set kw_score (and initial domains) on each item."""

    @property
    def max_candidates(self) -> int:
        return self.ctx.opts.max_candidates or self.cfg.get("max_candidates", 60)

    def prefilter(self, items: list) -> list:
        ranked = sorted((i for i in items if i.kw_score > 0), key=lambda i: i.kw_score, reverse=True)
        return ranked[:self.max_candidates]

    @property
    def learned(self) -> dict:
        return learned.normalise(self.ctx.cfg.get("learned"))

    def apply_learned(self, items: list) -> None:
        """Soft re-ranking from your approved preferences: scales a candidate's keyword score (never to zero, never filtering),
        so it matters when the candidate cap bites."""
        l = self.learned
        if not (l["prefer"] or l["deprioritize"]):
            return
        for it in items:
            if it.kw_score > 0:
                adj, _ = learned.adjust(learned.features_of(it), l)
                it.kw_score *= max(0.4, 1 + adj / 5)

    def triage_spec(self, feedback: list[tuple[int, str]]) -> triage.TriageSpec:
        raise NotImplementedError

    def stub_track(self, item) -> str | None:
        return None

    def finalize(self, st: RunState) -> int:
        raise NotImplementedError

    # --------------------------------------------------------------- template
    def execute(self) -> int:
        ctx = self.ctx
        db = ctx.db
        start, end = self.window()
        items, errors, counts = self.fetch(start, end)
        st = RunState(start=start, retrieved=len(items), counts=counts, errors=list(errors),
                      by_source=", ".join(f"{k} {v}" for k, v in counts.items()))
        log.info("[%s] retrieved %d (%s)", self.name, st.retrieved, st.by_source)

        items = self.normalize(items)
        items, notices = self.split_notices(items)
        st.notices = [p for p in notices if db.find(p) is None]
        st.fresh = [p for p in items if self.is_new(p)]
        log.info("[%s] %d new after dedup/exclusions (%d notices)", self.name, len(st.fresh), len(st.notices))

        self.score(st.fresh)
        self.apply_learned(st.fresh)
        st.cands = self.prefilter(st.fresh)
        log.info("[%s] %d candidates for triage (cap %d)", self.name, len(st.cands), self.max_candidates)

        spec = self.triage_spec(db.recent_feedback(self.cfg.get("feedback_examples", 10), self.name))
        spec.system_prompt = learned.insert_into_prompt(spec.system_prompt, self.learned)
        batch_size = self.cfg.get("batch_size", 15)
        st.est = triage.estimate_cost(spec, st.cands, ctx.prices, batch_size)

        if ctx.opts.dry_run:
            st.res = triage.stub_triage(st.cands, self.stub_track)
        elif st.cands:
            import anthropic
            client = anthropic.Anthropic(api_key=ctx.settings.get("ANTHROPIC_API_KEY"), max_retries=3, timeout=120)
            tr = triage.Triager(client, ctx.settings.model_for(self.cfg), ctx.settings.effort_for(self.name),
                                ctx.budget, spec, batch_size)
            st.res = tr.run(st.cands)
        st.errors += st.res.errors
        if ctx.budget.tripped:
            ctx.notes.append(f"{self.name}: budget guard tripped ({ctx.budget.trip_reason}); "
                             f"{len(st.res.unscored)} item(s) shown keyword-ranked without LLM summaries")
        elif st.res.unscored:
            ctx.notes.append(f"{self.name}: {len(st.res.unscored)} item(s) not triaged (API unavailable)")
        ctx.counts.update({"retrieved": st.retrieved, "new": len(st.fresh), "screened": len(st.cands),
                           "scored": len(st.res.scored), "unscored": len(st.res.unscored)})
        return self.finalize(st)
