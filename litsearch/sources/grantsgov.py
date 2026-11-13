"""Grants.gov search2 + fetchOpportunity (no key needed). NOFOs for NIH moved here in FY2026 (NOT-OD-25-143)."""
from __future__ import annotations

import html
import logging
import re
from datetime import date, datetime

from ..core.http import RateLimiter, request

log = logging.getLogger(__name__)
SEARCH = "https://api.grants.gov/v1/api/search2"
FETCH = "https://api.grants.gov/v1/api/fetchOpportunity"
PAGE = 100
MAX_PAGES = 30


def strip_html(s: str | None, limit: int = 2500) -> str:
    t = re.sub(r"</(p|div|li|tr|h\d)>|<br\s*/?>", "\n", s or "", flags=re.I)
    t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    t = re.sub(r"[ \t\r\f\v]+", " ", t)
    t = re.sub(r"\s*\n\s*", "\n", t).strip()
    return t if len(t) <= limit else t[:limit].rsplit(" ", 1)[0] + " …"


def iso_date(s: str | None) -> str:
    """'10/29/2026', 'Jan 26, 2027 12:00:00 AM EST' or 'Jan 26, 2027' -> '2027-01-26' ('' if unparseable)."""
    if not s:
        return ""
    s = s.strip()
    for pat, fmt in ((r"^\d{1,2}/\d{1,2}/\d{4}", "%m/%d/%Y"),
                     (r"^[A-Z][a-z]{2} \d{1,2}, \d{4}", "%b %d, %Y"),
                     (r"^\d{4}-\d{2}-\d{2}", "%Y-%m-%d")):
        m = re.match(pat, s)
        if m:
            try:
                return datetime.strptime(m.group(0), fmt).date().isoformat()
            except ValueError:
                pass
    return ""


def parse_hit(h: dict) -> dict:
    return {"opp_id": str(h["id"]), "number": h.get("number") or "", "title": h.get("title") or "",
            "agency_code": h.get("agencyCode") or "", "agency": h.get("agency") or "",
            "open_date": iso_date(h.get("openDate")), "close_date": iso_date(h.get("closeDate")),
            "status": h.get("oppStatus") or "", "doc_type": h.get("docType") or ""}


def search_recent(agencies: list[str], since: date, limiter: RateLimiter | None = None) -> list[dict]:
    """Posted/forecasted opportunities opened on/after `since`, newest first."""
    limiter = limiter or RateLimiter(0.4)
    out: list[dict] = []
    for page in range(MAX_PAGES):
        body = {"keyword": "", "oppStatuses": "forecasted|posted", "rows": PAGE, "sortBy": "openDate|desc",
                "startRecordNum": page * PAGE, "agencies": "|".join(agencies)}
        r = request("POST", SEARCH, limiter, json=body).json()
        if r.get("errorcode") not in (0, None):
            raise RuntimeError(f"Grants.gov search error: {r.get('msg')}")
        hits = [parse_hit(h) for h in (r.get("data") or {}).get("oppHits", [])]
        out += [h for h in hits if not h["open_date"] or h["open_date"] >= since.isoformat()]
        if len(hits) < PAGE or (hits and hits[-1]["open_date"] and hits[-1]["open_date"] < since.isoformat()):
            break
    return out


def fetch_opportunity(opp_id: str, limiter: RateLimiter | None = None) -> dict:
    r = request("POST", FETCH, limiter or RateLimiter(0.4), json={"opportunityId": int(opp_id)}).json()
    if r.get("errorcode") not in (0, None):
        raise RuntimeError(f"Grants.gov fetchOpportunity error: {r.get('msg')}")
    return parse_detail(r["data"])


def parse_detail(d: dict) -> dict:
    """Normalise a fetchOpportunity payload (synopsis or forecast)."""
    syn, fc = d.get("synopsis") or {}, d.get("forecast") or {}
    doc_type = d.get("docType") or ("synopsis" if syn else "forecast")
    base = syn if (doc_type == "synopsis" and syn) else fc
    desc = syn.get("synopsisDesc") if doc_type == "synopsis" else fc.get("forecastDesc")
    agency = (d.get("agencyDetails") or {}).get("agencyName") or syn.get("agencyName") or ""
    instruments = [i.get("description") for i in (base.get("fundingInstruments") or []) if i.get("description")]
    return {
        "opp_id": str(d.get("id") or ""), "number": d.get("opportunityNumber") or "",
        "title": d.get("opportunityTitle") or "", "agency_code": d.get("owningAgencyCode") or "",
        "agency": agency, "doc_type": doc_type, "status": "forecasted" if doc_type == "forecast" else "posted",
        "open_date": iso_date(base.get("postingDate")),
        "close_date": iso_date(syn.get("responseDate")) if doc_type == "synopsis" else iso_date(fc.get("estApplicationResponseDate")),
        "archive_date": iso_date(base.get("archiveDate")),
        "est_award_date": iso_date(fc.get("estAwardDate")),
        "description": strip_html(desc),
        "award_ceiling": syn.get("awardCeiling") if syn.get("awardCeiling") not in (None, "none") else None,
        "award_floor": syn.get("awardFloor") if syn.get("awardFloor") not in (None, "none") else None,
        "funding_instruments": instruments,
        "revision": int(d.get("revision") or 0), "last_updated": base.get("lastUpdatedDate") or "",
        "modified_comments": d.get("modifiedComments") or "",
    }
