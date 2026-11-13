"""medRxiv via the bioRxiv/medRxiv details API."""
from __future__ import annotations

import logging
from datetime import date

from ..core.http import RateLimiter, request
from ..core.models import Paper

log = logging.getLogger(__name__)
API = "https://api.biorxiv.org/details/medrxiv/{start}/{end}/{cursor}/json"
MAX_PAGES = 60  # 6,000 records, far above a normal 2-day window


def _author(a: str) -> str:
    """'Smith, J. A.' -> 'Smith JA'"""
    last, _, first = a.partition(",")
    initials = "".join(c for c in first if c.isalpha() and c.isupper())
    return f"{last.strip()} {initials}".strip()


def parse_record(rec: dict) -> Paper:
    pub = (rec.get("published") or "").strip()
    return Paper(
        source="medrxiv",
        title=" ".join((rec.get("title") or "").split()),
        abstract=" ".join((rec.get("abstract") or "").split()),
        authors=[_author(a) for a in (rec.get("authors") or "").split(";") if a.strip()],
        journal="medRxiv",
        date=rec.get("date", ""),
        doi=rec.get("doi"),
        published_doi=None if pub.upper() in ("", "NA") else pub,
        is_preprint=True,
    )


def fetch_medrxiv(start: date, end: date, categories: list[str]) -> tuple[list[Paper], str | None]:
    limiter = RateLimiter(0.5)
    wanted = {c.lower() for c in categories}
    latest: dict[str, dict] = {}
    total_seen = 0
    try:
        cursor = 0
        for _ in range(MAX_PAGES):
            url = API.format(start=start.isoformat(), end=end.isoformat(), cursor=cursor)
            data = request("GET", url, limiter).json()
            coll = data.get("collection") or []
            for rec in coll:
                total_seen += 1
                if (rec.get("category") or "").lower() not in wanted:
                    continue
                prev = latest.get(rec["doi"])
                if prev is None or int(rec.get("version", 1)) > int(prev.get("version", 1)):
                    latest[rec["doi"]] = rec
            msg = (data.get("messages") or [{}])[0]
            total = int(msg.get("total", 0) or 0)
            cursor += len(coll)
            if not coll or cursor >= total:
                break
    except Exception as e:  # noqa: BLE001
        log.exception("medRxiv fetch failed")
        return [parse_record(r) for r in latest.values()], f"medRxiv: {e}"
    log.info("medRxiv: %d records, %d in target categories", total_seen, len(latest))
    return [parse_record(r) for r in latest.values()], None
