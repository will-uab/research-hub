"""ClinicalTrials.gov API v2 (free, no key)."""
from __future__ import annotations

import logging
import re

from ..core.http import RateLimiter, request

log = logging.getLogger(__name__)
BASE = "https://clinicaltrials.gov/api/v2/studies"
FIELDS = ("NCTId,BriefTitle,OfficialTitle,BriefSummary,OverallStatus,WhyStopped,StudyFirstPostDate,LastUpdatePostDate,"
          "ResultsFirstPostDate,HasResults,LeadSponsorName,LeadSponsorClass,Phase,EnrollmentCount,EnrollmentType,"
          "InterventionType,InterventionName,InterventionDescription,PrimaryOutcomeMeasure,Condition,StartDate,CompletionDate")
STATUS_FIELDS = "NCTId,OverallStatus,WhyStopped,HasResults,ResultsFirstPostDate,LastUpdatePostDate"
PAGE = 200


def term_query(keywords: list[str]) -> str:
    """Essie expression: quoted phrases/terms joined with OR ('*' wildcards dropped)."""
    seen, parts = set(), []
    for k in keywords:
        k = k.replace("*", "").strip()
        if k and k.lower() not in seen:
            seen.add(k.lower())
            parts.append(f'"{k}"' if re.search(r"[\s-]", k) else k)
    return "(" + " OR ".join(parts) + ")"


def range_filter(field: str, since: str) -> str:
    return f"AREA[{field}]RANGE[{since},MAX]"


def search(term: str, advanced: str, limiter: RateLimiter | None = None, max_records: int = 600) -> tuple[list[dict], int]:
    limiter = limiter or RateLimiter(0.5)
    out, total, token = [], 0, None
    while len(out) < max_records:
        params = {"query.term": term, "filter.advanced": advanced, "fields": FIELDS, "pageSize": PAGE, "countTotal": "true"}
        if token:
            params["pageToken"] = token
        r = request("GET", BASE, limiter, params=params, timeout=90).json()
        total = r.get("totalCount", total)
        out += r.get("studies", [])
        token = r.get("nextPageToken")
        if not token:
            break
    return out[:max_records], total


def fetch_status(ids: list[str], limiter: RateLimiter | None = None) -> dict[str, dict]:
    """Current status for already-tracked trials, 100 ids per request."""
    limiter = limiter or RateLimiter(0.5)
    out: dict[str, dict] = {}
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        r = request("GET", BASE, limiter, params={"filter.ids": "|".join(chunk), "fields": STATUS_FIELDS,
                                                 "pageSize": 100}, timeout=90).json()
        for s in r.get("studies", []):
            ps = s["protocolSection"]
            st = ps.get("statusModule", {})
            out[ps["identificationModule"]["nctId"]] = {
                "status": st.get("overallStatus") or "", "why_stopped": st.get("whyStopped") or "",
                "has_results": bool(s.get("hasResults")),
                "results_first_post": (st.get("resultsFirstPostDateStruct") or {}).get("date", ""),
                "last_update": (st.get("lastUpdatePostDateStruct") or {}).get("date", "")}
    return out


def _clean(s: str | None) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def parse_study(s: dict) -> dict:
    ps = s["protocolSection"]
    idm, st = ps.get("identificationModule", {}), ps.get("statusModule", {})
    des, arms = ps.get("designModule", {}), ps.get("armsInterventionsModule", {})
    spons = (ps.get("sponsorCollaboratorsModule") or {}).get("leadSponsor") or {}
    enr = des.get("enrollmentInfo") or {}
    date = lambda k: (st.get(k) or {}).get("date", "")
    return {
        "nct_id": idm["nctId"], "title": _clean(idm.get("briefTitle") or idm.get("officialTitle")),
        "official_title": _clean(idm.get("officialTitle")),
        "summary": _clean((ps.get("descriptionModule") or {}).get("briefSummary")),
        "status": st.get("overallStatus") or "", "why_stopped": _clean(st.get("whyStopped")),
        "first_post": date("studyFirstPostDateStruct"), "last_update": date("lastUpdatePostDateStruct"),
        "results_first_post": date("resultsFirstPostDateStruct"), "has_results": bool(s.get("hasResults")),
        "start_date": date("startDateStruct"), "completion_date": date("completionDateStruct"),
        "sponsor": spons.get("name") or "", "sponsor_class": spons.get("class") or "",
        "phases": [p for p in (des.get("phases") or []) if p != "NA"] or (["N/A"] if des else []),
        "enrollment": enr.get("count"), "enrollment_type": (enr.get("type") or "").lower(),
        "interventions": [{"type": i.get("type", ""), "name": _clean(i.get("name")),
                           "description": _clean(i.get("description"))[:300]} for i in (arms.get("interventions") or [])][:6],
        "primary_outcomes": [_clean(o.get("measure")) for o in ((ps.get("outcomesModule") or {}).get("primaryOutcomes") or [])][:2],
        "conditions": ((ps.get("conditionsModule") or {}).get("conditions") or [])[:6],
    }
