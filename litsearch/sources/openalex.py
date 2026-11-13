"""OpenAlex (free): disambiguated authors and their recent works, incl. preprints that PubMed doesn't index."""
from __future__ import annotations

import logging
import re
from datetime import date

from ..core.http import RateLimiter, request

log = logging.getLogger(__name__)
BASE = "https://api.openalex.org"
SKIP_TYPES = {"erratum", "paratext", "peer-review", "retraction", "other"}
WORK_FIELDS = "id,doi,title,publication_date,type,primary_location,abstract_inverted_index,authorships,ids"
_LIM = RateLimiter(0.25)


def short_id(url: str | None) -> str:
    return (url or "").rstrip("/").rsplit("/", 1)[-1]


def decode_abstract(inv: dict | None) -> str:
    """OpenAlex ships abstracts as {word: [positions]}; rebuild the text."""
    if not inv:
        return ""
    slots: dict[int, str] = {}
    for word, positions in inv.items():
        for p in positions:
            slots[p] = word
    return " ".join(slots[i] for i in sorted(slots))


def search_authors(name: str, limiter: RateLimiter | None = None, per_page: int = 6) -> list[dict]:
    r = request("GET", f"{BASE}/authors", limiter or _LIM, params={"search": name, "per-page": per_page}, timeout=45).json()
    out = []
    for a in r.get("results", []):
        out.append({"id": short_id(a["id"]), "name": a.get("display_name", ""), "works": a.get("works_count", 0),
                    "cited_by": a.get("cited_by_count", 0), "orcid": (a.get("orcid") or "").replace("https://orcid.org/", ""),
                    "institutions": [i.get("display_name", "") for i in (a.get("last_known_institutions") or [])][:3]})
    return out


def parse_work(w: dict, watched: dict[str, str]) -> dict | None:
    """`watched` maps OpenAlex author id -> display name used on your list."""
    if (w.get("type") or "") in SKIP_TYPES or not (w.get("title") or "").strip():
        return None
    loc = w.get("primary_location") or {}
    src = loc.get("source") or {}
    auths = w.get("authorships") or []
    names = [(a.get("author") or {}).get("display_name", "") for a in auths]
    hit_ids = [short_id((a.get("author") or {}).get("id")) for a in auths]
    pmid = short_id((w.get("ids") or {}).get("pmid"))
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", w.get("doi") or "")
    return {
        "id": short_id(w["id"]), "title": re.sub(r"\s+", " ", w["title"]).strip(), "doi": doi or None,
        "pmid": pmid if pmid.isdigit() else None, "date": w.get("publication_date") or "", "type": w.get("type") or "",
        "journal": src.get("display_name") or "", "authors": [n for n in names if n],
        "abstract": decode_abstract(w.get("abstract_inverted_index")),
        "url": f"https://doi.org/{doi}" if doi else (loc.get("landing_page_url") or w["id"]),
        "watched": [watched[i] for i in dict.fromkeys(hit_ids) if i in watched],
        "is_preprint": (w.get("type") == "preprint") or (src.get("type") == "repository"),
    }


def fetch_works(watched: dict[str, str], since: date, limiter: RateLimiter | None = None,
                max_records: int = 300) -> list[dict]:
    """Works by any watched author published on/after `since`. 50 authors per request."""
    limiter = limiter or _LIM
    ids = list(watched)
    out: dict[str, dict] = {}
    for i in range(0, len(ids), 50):
        chunk = "|".join(ids[i:i + 50])
        page = 1
        while len(out) < max_records:
            r = request("GET", f"{BASE}/works", limiter, timeout=60, params={
                "filter": f"authorships.author.id:{chunk},from_publication_date:{since.isoformat()}",
                "sort": "publication_date:desc", "per-page": 100, "page": page, "select": WORK_FIELDS}).json()
            for w in r.get("results", []):
                p = parse_work(w, watched)
                if p and p["id"] not in out:
                    out[p["id"]] = p
            if page * 100 >= (r.get("meta") or {}).get("count", 0) or not r.get("results"):
                break
            page += 1
    return list(out.values())[:max_records]
