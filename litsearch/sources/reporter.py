"""NIH RePORTER API v2 (free, no key): newly awarded projects."""
from __future__ import annotations

import logging
import re

from ..core.http import RateLimiter, request

log = logging.getLogger(__name__)
URL = "https://api.reporter.nih.gov/v2/projects/search"
PAGE = 500            # API maximum
INCLUDE = ["ApplId", "ProjectNum", "ProjectTitle", "PrincipalInvestigators", "Organization", "AgencyIcAdmin",
           "AwardAmount", "DirectCostAmt", "IndirectCostAmt", "FiscalYear", "AwardNoticeDate", "AbstractText",
           "ActivityCode", "ProjectDetailUrl", "AwardType", "ProjectStartDate"]


def text_query(keywords: list[str]) -> str:
    """Boolean text for `advanced_text_search`: quoted phrases/terms joined with OR ('*' wildcards dropped)."""
    seen, parts = set(), []
    for k in keywords:
        k = k.replace("*", "").strip()
        if k and k.lower() not in seen:
            seen.add(k.lower())
            parts.append(f'"{k}"' if re.search(r"[\s-]", k) else k)
    return "(" + " OR ".join(parts) + ")"


def profile_ids(people: list[dict]) -> list[int]:
    """people_to_watch entries pinned to a RePORTER person: {name: ..., profile_id: 12345}."""
    return [int(p["profile_id"]) for p in people if p.get("profile_id")]


def pi_criteria(people: list[dict]) -> list[dict]:
    """Name-only people_to_watch [{name: 'Last, First'}] -> RePORTER pi_names criteria (pinned entries excluded)."""
    out = []
    for p in people:
        if p.get("profile_id"):
            continue
        last, _, first = (p.get("name") or "").partition(",")
        if last.strip():
            out.append({"last_name": last.strip(), **({"first_name": first.strip()} if first.strip() else {})})
    return out


def search(criteria: dict, limiter: RateLimiter | None = None, max_records: int = 1000) -> tuple[list[dict], int]:
    """(records, total). Text queries come back relevance-sorted, so a cap drops the weakest matches."""
    limiter = limiter or RateLimiter(1.0)            # NIH asks for <= 1 request/second
    out: list[dict] = []
    total, offset = 0, 0
    while offset < min(max_records, 14000):
        body = {"criteria": criteria, "include_fields": INCLUDE, "limit": PAGE, "offset": offset}
        r = request("POST", URL, limiter, json=body, timeout=90).json()
        if "results" not in r:
            raise RuntimeError(f"RePORTER error: {str(r)[:200]}")
        total = r["meta"]["total"]
        out += r["results"]
        offset += PAGE
        if offset >= total:
            break
    return out[:max_records], total


def _clean(s: str | None) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def parse_record(r: dict) -> dict:
    pis = r.get("principal_investigators") or []
    pis = sorted(pis, key=lambda p: not p.get("is_contact_pi"))        # contact PI first
    org = r.get("organization") or {}
    amount = r.get("award_amount")
    if amount is None and (r.get("direct_cost_amt") is not None or r.get("indirect_cost_amt") is not None):
        amount = (r.get("direct_cost_amt") or 0) + (r.get("indirect_cost_amt") or 0)
    ic = r.get("agency_ic_admin") or {}
    return {
        "appl_id": str(r["appl_id"]), "project_num": r.get("project_num") or "",
        "title": _clean(r.get("project_title")), "abstract": _clean(r.get("abstract_text")),
        "pis": [{"name": _clean(p.get("full_name")) or _clean(f"{p.get('first_name')} {p.get('last_name')}"),
                 "first": _clean(p.get("first_name")), "last": _clean(p.get("last_name")), "profile_id": p.get("profile_id"),
                 "contact": bool(p.get("is_contact_pi"))} for p in pis],
        "institution": _clean(org.get("org_name")), "city": _clean(org.get("org_city")), "state": _clean(org.get("org_state")),
        "institute": ic.get("abbreviation") or ic.get("code") or "", "mechanism": r.get("activity_code") or "",
        "award_type": str(r.get("award_type") or ""), "fiscal_year": r.get("fiscal_year"),
        "amount": amount, "award_notice_date": (r.get("award_notice_date") or "")[:10],
        "start_date": (r.get("project_start_date") or "")[:10], "url": r.get("project_detail_url") or "",
    }


def match_watch(pis: list[dict], institution: str, people: list[dict]) -> list[str]:
    """Which people_to_watch entries match this project's PIs (last name + first-name prefix; institution if given)."""
    hits = []
    for p in people:
        if p.get("profile_id"):                        # pinned: exact person, no name guessing
            if any(str(pi.get("profile_id")) == str(p["profile_id"]) for pi in pis):
                hits.append(p["name"])
            continue
        last, _, first = (p.get("name") or "").partition(",")
        last, first = last.strip().lower(), first.strip().lower()
        inst = (p.get("institution") or "").strip().lower()
        if not last or (inst and inst not in institution.lower()):
            continue
        for pi in pis:
            if pi["last"].lower() == last and (not first or pi["first"].lower().startswith(first[:1])
                                               and (len(first) < 2 or pi["first"].lower().startswith(first))):
                hits.append(p["name"])
                break
    return hits
