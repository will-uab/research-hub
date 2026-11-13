"""Shared behaviour for the non-literature streams: select, record, dry-run summary."""
from __future__ import annotations

import logging

from .base import RunState, Stream

log = logging.getLogger(__name__)


class ItemStream(Stream):
    """Streams whose output is a flat list of Items (grants, trials, awards)."""

    stored_shown: list = []

    @property
    def min_relevance(self) -> float:
        return self.cfg.get("min_relevance", 4)

    def describe(self, item) -> str:
        """One line for the dry-run sample listing."""
        return f"{item.title[:90]}"

    def select(self, st: RunState) -> tuple[list, list, list]:
        """(shown, below_threshold, fallback). Unscored (budget guard) items are shown keyword-ranked."""
        shown = [i for i in st.res.scored if (i.relevance or 0) >= self.min_relevance]
        below = [i for i in st.res.scored if (i.relevance or 0) < self.min_relevance]
        shown.sort(key=lambda i: (-(i.relevance or 0), -i.kw_score))
        fallback = sorted(st.res.unscored, key=lambda i: i.kw_score, reverse=True)[:self.cfg.get("fallback_max_items", 10)]
        return shown, below, fallback

    def extra_items(self, st: RunState, shown: list) -> list:
        """Hook: items to store/show in addition (e.g. the weekly grants summary)."""
        return []

    def finalize(self, st: RunState) -> int:
        ctx, db = self.ctx, self.ctx.db
        shown, below, fallback = self.select(st)
        extra = self.extra_items(st, shown)
        self.stored_shown = shown + fallback + extra
        for it in self.stored_shown:
            db.record(it, "shown")                    # in a dry-run this is the throwaway preview DB
        if ctx.opts.dry_run:
            self.print_dry_run(st, shown, fallback, extra)
            return 0
        for it in below:
            db.record(it, "screened")
        ctx.counts.update({"shown": len(shown) + len(fallback), "below_threshold": len(below)})
        log.info("[%s] stored %d shown (%d keyword-ranked fallback), %d screened out; run cost $%.5f",
                 self.name, len(shown) + len(fallback), len(fallback), len(below), ctx.budget.run_cost)
        return 0

    def print_dry_run(self, st: RunState, shown: list, fallback: list, extra: list) -> None:
        ctx, est = self.ctx, st.est
        print(f"\nDry run [{self.name}]: {len(shown) + len(fallback)} item(s) would be shown, "
              f"{len(extra)} extra, from {st.retrieved} new ({st.by_source}); {len(st.cands)} to triage")
        print(f"  estimated LLM cost for this run: ${est['cost_usd']:.4f} ({est['calls']} calls, "
              f"~{est['input_tokens']:,} in / ~{est['output_tokens']:,} out tokens)"
              + (" (prices from .env.example)" if ctx.used_example_prices else ""))
        for k, v in (ctx.counts or {}).items():
            if k not in ("retrieved", "new", "screened", "scored", "unscored"):
                print(f"  {k}: {v}")
        for e in st.errors:
            print(f"  ! {e}")
        for it in (shown + fallback)[:12]:
            print(f"   - {self.describe(it)}")
        for it in extra:
            print(f"   * {it.title}")
