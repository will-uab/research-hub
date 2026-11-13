"""Hard budget guard across all streams. Prices come from settings, never from code."""
from __future__ import annotations

import logging
from datetime import date, timedelta

from .db import DB

log = logging.getLogger(__name__)

CHARS_PER_TOKEN = 3.0  # conservative (real English text is ~4); over-estimates cost


class PricesMissing(RuntimeError):
    pass


class Prices:
    """USD per million tokens."""

    def __init__(self, inp: float, out: float, cache_write: float | None = None,
                 cache_read: float | None = None):
        self.inp, self.out = inp, out
        self.cache_write = cache_write if cache_write is not None else inp * 1.25
        self.cache_read = cache_read if cache_read is not None else inp * 0.1

    @classmethod
    def from_env(cls, env: dict[str, str], stream: str | None = None) -> "Prices":
        """PRICE_<STREAM>_* overrides PRICE_* (a stream on a different model needs its own prices)."""
        def f(suffix):
            keys = ([f"PRICE_{stream.upper()}_{suffix}"] if stream else []) + [f"PRICE_{suffix}"]
            for k in keys:
                v = env.get(k, "").strip()
                if v:
                    return float(v)
            return None

        inp, out = f("INPUT_PER_MTOK"), f("OUTPUT_PER_MTOK")
        if inp is None or out is None:
            raise PricesMissing(f"PRICE_INPUT_PER_MTOK and PRICE_OUTPUT_PER_MTOK must be set in .env"
                                + (f" (or PRICE_{stream.upper()}_* for the {stream} stream)" if stream else ""))
        return cls(inp, out, f("CACHE_WRITE_PER_MTOK"), f("CACHE_READ_PER_MTOK"))

    def cost(self, usage: dict) -> float:
        return (
            usage.get("input_tokens", 0) * self.inp
            + usage.get("output_tokens", 0) * self.out
            + usage.get("cache_write_tokens", 0) * self.cache_write
            + usage.get("cache_read_tokens", 0) * self.cache_read
        ) / 1_000_000

    def estimate(self, input_chars: int, max_output_tokens: int) -> float:
        """Worst-case cost of one call: all input uncached-at-full-price (or cache write), full output."""
        in_tok = input_chars / CHARS_PER_TOKEN
        return (in_tok * max(self.inp, self.cache_write) + max_output_tokens * self.out) / 1_000_000


def iso_week_bounds(day: str) -> tuple[str, str]:
    """Monday..Sunday (ISO week) containing `day`, as ISO date strings."""
    d = date.fromisoformat(day)
    mon = d - timedelta(days=d.weekday())
    return mon.isoformat(), (mon + timedelta(days=6)).isoformat()


class Budget:
    """One stream's view of the budget.

    Normal streams are limited by the combined daily cap (all streams) and an optional per-stream cap.
    A stream with `weekly_limit` (the journal club) is limited by that weekly allowance only.
    A stream with `separate` (the on-demand podcast) has its own daily budget (`stream_cap`) that is not part of the combined cap:
    its spend is not counted against the other streams, and theirs is not counted against it.
    """

    def __init__(self, db: DB, day: str, limit_usd: float, prices: Prices, stream: str = "literature",
                 stream_cap: float | None = None, weekly_limit: float | None = None, separate: bool = False,
                 exclude_streams: tuple[str, ...] = ()):
        self.db, self.day, self.limit, self.prices = db, day, limit_usd, prices
        self.stream, self.stream_cap, self.weekly_limit = stream, stream_cap, weekly_limit
        self.separate, self.exclude_streams = separate, tuple(exclude_streams)
        self.tripped = False
        self.trip_reason = ""
        self.run_cost = 0.0

    @property
    def spent_today(self) -> float:
        """Combined spend across every stream that shares the daily cap (separate-budget streams are not part of it)."""
        return self.db.spent(self.day, exclude=self.exclude_streams)

    @property
    def spent_stream_today(self) -> float:
        return self.db.spent(self.day, self.stream)

    @property
    def spent_week(self) -> float:
        a, b = iso_week_bounds(self.day)
        return self.db.spent_between(a, b, self.stream if self.weekly_limit is not None else None)

    def allow(self, worst_case_cost: float) -> bool:
        """True if a call costing up to `worst_case_cost` keeps every applicable cap intact."""
        if self.separate:
            cap = self.stream_cap if self.stream_cap is not None else self.limit
            ok = self.spent_stream_today + worst_case_cost <= cap
            reason = f"{self.stream} budget ${cap:.2f}"
        elif self.weekly_limit is not None:
            ok = self.spent_week + worst_case_cost <= self.weekly_limit
            reason = f"weekly allowance ${self.weekly_limit:.2f}"
        else:
            ok = self.spent_today + worst_case_cost <= self.limit
            reason = f"daily cap ${self.limit:.2f}"
            if ok and self.stream_cap is not None and self.spent_stream_today + worst_case_cost > self.stream_cap:
                ok = False
                reason = f"{self.stream} cap ${self.stream_cap:.2f}"
        if not ok:
            self.tripped = True
            self.trip_reason = reason
            log.warning("Budget guard (%s): worst case %.4f would exceed %s (today %.4f, stream %.4f, week %.4f)",
                        self.stream, worst_case_cost, reason, self.spent_today, self.spent_stream_today,
                        self.spent_week)
        return ok

    def record(self, usage: dict) -> float:
        cost = self.prices.cost(usage)
        self.db.add_spend(self.day, self.stream, usage, cost)
        self.run_cost += cost
        return cost
