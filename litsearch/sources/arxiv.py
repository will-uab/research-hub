"""arXiv via the public Atom API."""
from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from datetime import date

from ..core.http import RateLimiter, request
from ..core.models import Paper
from ..scoring import has_health_terms

log = logging.getLogger(__name__)
API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
PAGE = 200
MAX_PAGES = 25  # 5,000 records


def _clean(s: str | None) -> str:
    return " ".join((s or "").split())


def parse_feed(xml_bytes: bytes) -> tuple[list[Paper], int]:
    """Returns (papers, total_results)."""
    root = ET.fromstring(xml_bytes)
    total = int(root.findtext("{http://a9.com/-/spec/opensearch/1.1/}totalResults") or 0)
    out = []
    for e in root.findall("a:entry", NS):
        raw_id = e.findtext("a:id", "", NS)
        m = re.search(r"arxiv\.org/abs/(.+?)(v\d+)?$", raw_id)
        if not m:
            continue
        out.append(Paper(
            source="arxiv",
            title=_clean(e.findtext("a:title", "", NS)),
            abstract=_clean(e.findtext("a:summary", "", NS)),
            authors=[_clean(a.findtext("a:name", "", NS)) for a in e.findall("a:author", NS)],
            journal="arXiv",
            date=(e.findtext("a:published", "", NS) or "")[:10],
            arxiv_id=m.group(1),
            published_doi=e.findtext("x:doi", None, NS),
            is_preprint=True,
        ))
    return out, total


def fetch_arxiv(query: str, health_terms: list[str] | None) -> tuple[list[Paper], str | None]:
    """`query` from queries.arxiv_query. health_terms=None disables the local clinical gate."""
    limiter = RateLimiter(3.2)  # arXiv asks for >=3s between calls
    kept, seen = [], 0
    try:
        start = 0
        for _ in range(MAX_PAGES):
            r = request("GET", API, limiter, params={
                "search_query": query, "start": start, "max_results": PAGE,
                "sortBy": "submittedDate", "sortOrder": "descending"})
            papers, total = parse_feed(r.content)
            seen += len(papers)
            kept += [p for p in papers if health_terms is None or has_health_terms(p, health_terms)]
            start += PAGE
            if not papers or start >= total:
                break
    except Exception as e:  # noqa: BLE001
        log.exception("arXiv fetch failed")
        return kept, f"arXiv: {e}"
    log.info("arXiv: %d records, %d pass health-term gate", seen, len(kept))
    return kept, None
