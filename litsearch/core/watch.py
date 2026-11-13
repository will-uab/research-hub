"""The watch list (people_to_watch in interests.yaml): search, add, remove, and auto-resolve OpenAlex ids."""
from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

from ..sources import openalex
from ..sources import reporter as rp
from ..core.http import RateLimiter, request
from . import config

log = logging.getLogger(__name__)
GENERIC = {"university", "hospital", "medicine", "school", "department", "division", "health", "college", "center",
           "centre", "institute", "medical", "public", "infectious", "diseases", "disease", "sciences", "research",
           "clinical", "care", "system", "systems", "usa", "united", "states", "foundation", "network"}


def normalize_name(q: str) -> tuple[str, str]:
    """'Elena F Rivera', 'Rivera, Elena' or 'Rivera EF' -> (last, first). Returns ('', '') if unusable."""
    q = re.sub(r"\s+", " ", (q or "").strip())
    if not q:
        return "", ""
    if "," in q:
        last, _, first = q.partition(",")
        return last.strip(), first.strip().split(" ")[0] if first.strip() else ""
    parts = q.split(" ")
    return (parts[-1], parts[0]) if len(parts) > 1 else (parts[0], "")


def tidy_name(name: str) -> str:
    """'SPRINGER, SANDRA' / 'SANDRA Ann SPRINGER' (RePORTER capitals) -> 'Springer, Sandra'; 'Last, First' passes through."""
    name = re.sub(r"\s+", " ", (name or "").strip())
    if "," not in name and " " in name:                      # 'First [Middle] Last' -> 'Last, First'
        parts = name.split(" ")
        name = f"{parts[-1]}, {parts[0]}"
    fix = lambda w: w[0] + w[1:].lower() if len(w) > 1 and w.isupper() else w
    return ", ".join(" ".join(fix(w) for w in part.strip().split(" ")) for part in name.split(",", 1))


def display_name(last: str, first: str) -> str:
    nice = lambda s: s.title() if s.isupper() else s
    return f"{nice(last)}, {nice(first)}" if first else nice(last)


def oa_ids(entry: dict) -> list[str]:
    """An entry's OpenAlex author ids; OpenAlex often splits one person into several records, so there can be many."""
    v = entry.get("openalex_id")
    return [str(x) for x in (v if isinstance(v, list) else [v] if v else [])]


def _fold(s: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", s or "") if not unicodedata.combining(c)).lower()


def name_ok(cand_name: str, last: str, first: str) -> bool:
    """The candidate's display name must contain the last name and agree on the first initial (OpenAlex search is fuzzy)."""
    parts = re.findall(r"[a-z]+", _fold(cand_name))
    if _fold(last) not in parts:
        return False
    rest = [p for p in parts if p != _fold(last)]
    return not first or not rest or any(p[0] == _fold(first)[0] for p in rest[:2])


def toks(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", (s or "").lower()) if w not in GENERIC}


# ------------------------------------------------------------------ list management
def load(settings) -> list[dict]:
    return list(config.load_interests(settings.interests_path).get("people_to_watch") or [])


def _same(existing: dict, new: dict) -> bool:
    """Same person if an id matches; or the names match and no id says they are different people."""
    if existing.get("profile_id") and new.get("profile_id"):
        return str(existing["profile_id"]) == str(new["profile_id"])
    if oa_ids(existing) and oa_ids(new) and (set(oa_ids(existing)) & set(oa_ids(new))):
        return True
    return existing.get("name", "").lower() == new.get("name", "").lower()


def add(settings, entry: dict) -> tuple[list[dict], bool]:
    """Add (or enrich an existing entry with new ids). Returns (list, newly_added)."""
    clean = {k: entry[k] for k in ("name", "profile_id", "openalex_id", "note") if entry.get(k)}
    if isinstance(clean.get("openalex_id"), list):
        clean["openalex_id"] = clean["openalex_id"][0] if len(clean["openalex_id"]) == 1 else clean["openalex_id"]
    if not clean.get("name"):
        raise ValueError("a name is required")
    clean["name"] = tidy_name(clean["name"])
    if clean.get("profile_id"):
        clean["profile_id"] = int(clean["profile_id"])
    people = load(settings)
    for p in people:
        if _same(p, clean):
            before = dict(p)
            for k in ("profile_id", "note"):
                if clean.get(k) and not p.get(k):
                    p[k] = clean[k]
            merged = list(dict.fromkeys(oa_ids(p) + oa_ids(clean)))
            if merged:
                p["openalex_id"] = merged[0] if len(merged) == 1 else merged
            if p != before:
                config.save_people(settings.interests_path, people)
            return people, False
    people.append(clean)
    config.save_people(settings.interests_path, people)
    return people, True


def remove(settings, name: str | None = None, profile_id=None, openalex_id=None) -> tuple[list[dict], int]:
    people = load(settings)
    keep = [p for p in people if not (
        (name and p.get("name", "").lower() == name.lower())
        or (profile_id and str(p.get("profile_id")) == str(profile_id))
        or (openalex_id and str(p.get("openalex_id")) == str(openalex_id)))]
    if len(keep) != len(people):
        config.save_people(settings.interests_path, keep)
    return keep, len(people) - len(keep)


# ------------------------------------------------------------------ finding people
def reporter_profiles(last: str, first: str, limiter: RateLimiter | None = None) -> list[dict]:
    """RePORTER PI profiles (since 2019) matching a name, with their institutions."""
    crit = {"pi_names": [{"last_name": last, **({"first_name": first} if first else {})}],
            "award_notice_date": {"from_date": "2019-01-01", "to_date": "2099-01-01"}}
    r = request("POST", rp.URL, limiter or RateLimiter(1.0), json={
        "criteria": crit, "include_fields": ["PrincipalInvestigators", "Organization"], "limit": 200}, timeout=60).json()
    profs: dict = defaultdict(lambda: {"orgs": Counter(), "awards": 0, "name": ""})
    for x in r.get("results", []):
        for pi in x["principal_investigators"]:
            if pi["last_name"].lower() == last.lower() and (not first or pi["first_name"].lower().startswith(first.lower()[:1])):
                d = profs[pi["profile_id"]]
                d["orgs"][x["organization"]["org_name"]] += 1
                d["awards"] += 1
                d["name"] = pi["full_name"]
    return sorted(({"profile_id": pid, "name": d["name"], "awards": d["awards"], "institutions": [o for o, _ in d["orgs"].most_common(2)]}
                   for pid, d in profs.items()), key=lambda c: -c["awards"])


def search_candidates(q: str) -> dict:
    last, first = normalize_name(q)
    if not last:
        raise ValueError("enter a name like 'Last, First' or 'First Last'")
    out = {"query": q, "name": display_name(last, first), "reporter": [], "openalex": [], "errors": []}
    try:
        out["reporter"] = reporter_profiles(last, first)
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"RePORTER: {e}")
    try:
        out["openalex"] = openalex.search_authors(f"{first} {last}".strip())
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"OpenAlex: {e}")
    return out


# ------------------------------------------------------------------ auto-resolve OpenAlex ids for existing entries
def pick_openalex(cands: list[dict], hint_orgs: str, last: str = "", first: str = "") -> tuple[list[dict], str]:
    """Candidates that are plausibly this person: the name agrees AND an institution overlaps the hint (RePORTER org).
    Several records at the same institution are one person split by OpenAlex, so all are returned.
    Without a hint, only a single dominant record is accepted. Anything else stays unresolved for a human to pick."""
    cands = [c for c in cands if c["works"] > 0 and (not last or name_ok(c["name"], last, first))]
    h = toks(hint_orgs)
    if h:
        match = [c for c in cands if h & toks(" ".join(c["institutions"]))]
        if match:
            return sorted(match, key=lambda c: -c["works"]), "institution match" + (f" ({len(match)} records)" if len(match) > 1 else "")
        return [], "no candidate at the person's institution: pick in the watch panel"
    if len(cands) == 1:
        return cands, "only candidate"
    big = sorted(cands, key=lambda c: -c["works"])
    if len(big) > 1 and big[0]["works"] >= 20 and big[0]["works"] >= 10 * big[1]["works"]:
        return [big[0]], "dominant record (no institution to confirm)"
    return [], "ambiguous: pick in the watch panel"


def hint_for(entry: dict, limiter: RateLimiter | None = None) -> str:
    if not entry.get("profile_id"):
        return entry.get("note", "")
    r = request("POST", rp.URL, limiter or RateLimiter(1.0), json={
        "criteria": {"pi_profile_ids": [int(entry["profile_id"])]}, "include_fields": ["Organization"], "limit": 25}, timeout=60).json()
    return " ".join(x["organization"]["org_name"] for x in r.get("results", []))


def resolve_missing(settings) -> list[dict]:
    """Attach `openalex_id` to entries lacking one. Returns a report [{name, status, detail}]."""
    people, report, changed = load(settings), [], False
    for p in people:
        if oa_ids(p):
            continue
        last, first = normalize_name(p["name"])
        try:
            cands = openalex.search_authors(f"{first} {last}".strip(), per_page=10)
            chosen, why = pick_openalex(cands, hint_for(p), last, first)
        except Exception as e:  # noqa: BLE001
            report.append({"name": p["name"], "status": "error", "detail": str(e)})
            continue
        if chosen:
            ids = [c["id"] for c in chosen]
            p["openalex_id"] = ids[0] if len(ids) == 1 else ids
            changed = True
            c0 = chosen[0]
            report.append({"name": p["name"], "status": "resolved", "detail":
                           f"{c0['name']} ({'; '.join(c0['institutions'][:1]) or 'no institution'}), {sum(c['works'] for c in chosen)} works, {len(ids)} record(s) [{why}]"})
        else:
            report.append({"name": p["name"], "status": "unresolved", "detail": why})
    if changed:
        config.save_people(settings.interests_path, people)
    return report
