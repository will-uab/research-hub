"""Per-run context handed to a stream."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .budget import Budget, Prices
from .config import Settings
from .db import DB
from .domains import DomainTagger


@dataclass
class RunOptions:
    dry_run: bool = False
    backfill: int | None = None
    max_candidates: int | None = None
    force: bool = False               # tuning: run even though too few new ratings have accumulated
    redo: bool = False                # podcast: replace today's episode (costs another model call)
    methods_topic: str = "auto"       # podcast: "auto", "none", a curriculum topic id, or "custom:<free text>"
    dry_db: "Path | None" = None      # dry-run only: keep the preview DB here instead of a throwaway temp file


@dataclass
class RunContext:
    settings: Settings
    cfg: dict                    # full v2 interests config
    stream: str
    stream_cfg: dict
    db: DB
    opts: RunOptions
    today: date
    now: datetime
    prices: Prices
    budget: Budget
    tagger: DomainTagger
    used_example_prices: bool = False
    counts: dict[str, Any] = field(default_factory=dict)     # recorded on the runs row
    notes: list[str] = field(default_factory=list)           # shown in the status banner
