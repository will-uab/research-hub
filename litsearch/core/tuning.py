"""Tune-up logic: turn your ratings into *proposed* edits to the learned: block (nothing applies until you accept).

Deterministic parts (here): candidate rules with support thresholds and time decay, follow suggestions, where the LLM's scores and
your taps disagree, backtests of a rule over the last 30 days, and applying / undoing an accepted proposal.
The LLM only chooses from, words and explains what this module found (see streams/tuning.py).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone

from . import config, learned, methods

HALF_LIFE_DAYS = 180
STOP = set("""a about above after again against all also among an and any are as at based be because been before being between both but by can
could did do does during each few for from further had has have having here how i if in into is it its just more most no nor not of off on
once only or other our out over own same so some such than that the their them then there these they this those through to too under until
up use used using very was we were what when where which while who why will with within without would you your study studies analysis
review systematic trial trials clinical patients patient health medical medicine model models large language care case cohort outcomes
results evaluation approach novel effect effects impact across versus new via toward towards paper papers number article articles report reports""".split())
NEUTRAL_NEGATIVES = {"already_knew", "too_basic"}        # "not new to me" / "too elementary" say nothing about topic or source
QUALITY_ONLY = {"weak_study"}                            # says something about the paper's source, not its topic


def _age_weight(ts: str, now: datetime) -> float:
    try:
        days = max(0.0, (now - datetime.fromisoformat(ts)).total_seconds() / 86400)
    except ValueError:
        days = 0.0
    return 0.5 ** (days / HALF_LIFE_DAYS)


def title_terms(title: str) -> set[str]:
    words = [w for w in re.findall(r"[a-z][a-z\-]{3,}", (title or "").lower()) if w not in STOP]
    uni = set(words)
    bi = {f"{a} {b}" for a, b in zip(words, words[1:])}
    return uni | bi


def _features(r: dict) -> dict[str, set[str]]:
    return {"term": title_terms(r.get("title", "")), "journal": {r["journal"]} if r.get("journal") else set(),
            "author": set(r.get("authors") or []), "institute": set(r.get("institutes") or []),
            "mechanism": {r["mechanism"]} if r.get("mechanism") else set(), "sponsor": {r["sponsor"]} if r.get("sponsor") else set()}


def _counts_for(kind: str, r: dict) -> bool:
    """Does this rating inform rules of this kind, given the reason the reader gave?"""
    if r["sign"] < 0:
        if r.get("reason") in NEUTRAL_NEGATIVES:
            return False
        if r.get("reason") in QUALITY_ONLY and kind == "term":
            return False
    return True


def suggest_weight(likes: int, dislikes: int, share: float) -> float:
    n = likes + dislikes
    mag = 1 if n < 5 else 2 if (n < 8 or max(share, 1 - share) < 0.9) else 3
    return float(mag if share >= 0.5 else -mag)


def candidate_rules(rated: list[dict], current: dict, min_support: int = 3, min_share: float = 0.7, now: datetime | None = None) -> list[dict]:
    """Features with enough rated items and a consistent direction. A single tap can never create a rule."""
    now = now or datetime.now(timezone.utc)
    tally: dict[tuple[str, str], dict] = defaultdict(lambda: {"wl": 0.0, "wd": 0.0, "l": 0, "d": 0, "ids": []})
    for r in rated:
        w = _age_weight(r["ts"], now)
        for kind, vals in _features(r).items():
            if not _counts_for(kind, r):
                continue
            for v in vals:
                t = tally[(kind, v)]
                t["wl" if r["sign"] > 0 else "wd"] += w
                t["l" if r["sign"] > 0 else "d"] += 1
                t["ids"].append(r["item_id"])
    have = {(x.get("kind"), str(x.get("value", "")).lower()): float(x.get("weight") or 0) for x in current["prefer"] + current["deprioritize"]}
    out = []
    for (kind, value), t in tally.items():
        n = t["l"] + t["d"]
        if n < min_support:
            continue
        share = (t["wl"] + 1) / (t["wl"] + t["wd"] + 2)                                # decayed, Laplace-smoothed like-rate
        if min_share > share > 1 - min_share:
            continue
        weight = suggest_weight(t["l"], t["d"], share)
        if have.get((kind, value.lower()), 0) * weight > 0:                              # already have this rule in that direction
            continue
        out.append({"kind": kind, "value": value, "likes": t["l"], "dislikes": t["d"], "share": round(share, 2), "weight": weight,
                    "evidence": sorted(set(t["ids"]))})
    # terms: drop a unigram when a bigram containing it has the same evidence (keep the more specific phrase)
    drop = set()
    terms = [c for c in out if c["kind"] == "term"]
    for a in terms:
        for b in terms:
            if a is not b and " " in b["value"] and a["value"] in b["value"].split() and a["evidence"] == b["evidence"]:
                drop.add(id(a))
    out = [c for c in out if id(c) not in drop]
    out.sort(key=lambda c: (-(c["likes"] + c["dislikes"]), -abs(c["share"] - 0.5), c["kind"], c["value"]))
    return out[:40]


def _name_key(name: str) -> tuple[str, str]:
    """('smith', 'j') for 'Smith J', 'Smith, John' and 'John Smith' alike."""
    name = (name or "").strip()
    if "," in name:
        last, _, first = name.partition(",")
    else:
        parts = name.split()
        if len(parts) >= 2 and len(parts[-1]) <= 3 and parts[-1].isupper():             # PubMed style "Smith JA"
            last, first = " ".join(parts[:-1]), parts[-1]
        else:
            last, first = (parts[-1], " ".join(parts[:-1])) if len(parts) > 1 else (name, "")
    return last.strip().lower(), (first.strip()[:1] or "").lower()


def follow_candidates(rated: list[dict], followed: list[str], min_likes: int = 2) -> list[dict]:
    """Authors on at least `min_likes` papers you liked (or one you tagged 'follow this person') and don't follow yet."""
    have = {_name_key(n) for n in followed}
    tally: dict[str, dict] = defaultdict(lambda: {"likes": 0, "ids": [], "tagged": False})
    for r in rated:
        if r["sign"] < 0:
            continue
        for a in (r.get("authors") or []):
            t = tally[a]
            t["likes"] += 1
            t["ids"].append(r["item_id"])
            t["tagged"] = t["tagged"] or r.get("reason") == "follow_person"
    out = [{"name": a, "likes": t["likes"], "evidence": sorted(set(t["ids"])), "tagged": t["tagged"]} for a, t in tally.items()
           if (t["likes"] >= min_likes or t["tagged"]) and _name_key(a) not in have and a.strip()]
    return sorted(out, key=lambda c: (-c["likes"], c["name"]))[:10]


def disagreements(rated: list[dict], n: int = 6) -> list[dict]:
    """Where the LLM's score at the time and your tap point opposite ways: the most informative ratings."""
    out = []
    for r in rated:
        rel = r.get("relevance")
        if rel is None:
            continue
        gap = (8 - rel) if r["sign"] > 0 else (rel - 3)
        if gap >= 4 or (r["sign"] > 0 and rel <= 4) or (r["sign"] < 0 and rel >= 8):
            out.append({"item_id": r["item_id"], "sign": r["sign"], "llm_score": rel, "reason": r.get("reason"), "title": r.get("title", "")[:90], "gap": gap})
    return sorted(out, key=lambda d: -d["gap"])[:n]


def backtest(db, rule: dict, ratings: dict[int, int], days: int = 30) -> dict:
    """What this rule would have touched in the last `days` days, and how those items compare with your taps."""
    since = datetime.now(timezone.utc).timestamp() - days * 86400
    matches = liked = disliked = 0
    for r in db.conn.execute("SELECT id, title, summary, stream_specific, first_seen FROM items WHERE status IN ('sent','shown') "
                             "AND stream NOT IN ('briefing','podcast','journal_club')"):
        try:
            if datetime.fromisoformat(r["first_seen"]).timestamp() < since:
                continue
        except ValueError:
            continue
        if learned.rule_matches(rule, learned.features(r["title"], r["summary"] or "", json.loads(r["stream_specific"]))):
            matches += 1
            liked += ratings.get(r["id"]) == 1
            disliked += ratings.get(r["id"]) == -1
    return {"days": days, "matches": matches, "liked": liked, "disliked": disliked, "unrated": matches - liked - disliked}


# ---------------------------------------------------------------------- apply / undo (only ever on an explicit accept)
def _learned_now(settings) -> dict:
    return learned.normalise(config.load_interests(settings.interests_path).get("learned"))


def apply_proposal(settings, prop: dict) -> dict | None:
    """Write an accepted proposal into interests.yaml. Returns the `before` state needed to undo it."""
    l = _learned_now(settings)
    kind, pl = prop["kind"], prop["payload"]
    if kind == "preferences":
        before = {"preferences": l["preferences"]}
        l["preferences"] = pl["new"].strip()
    elif kind == "rule":
        key = (pl["kind"], str(pl["value"]).lower())
        same = lambda r: (r.get("kind"), str(r.get("value", "")).lower()) == key
        before = {"prefer": [r for r in l["prefer"] if same(r)], "deprioritize": [r for r in l["deprioritize"] if same(r)]}
        l["prefer"] = [r for r in l["prefer"] if not same(r)]
        l["deprioritize"] = [r for r in l["deprioritize"] if not same(r)]
        rule = {"kind": pl["kind"], "value": pl["value"], "weight": float(pl["weight"]), "note": (prop.get("why") or "")[:140],
                "evidence": prop.get("evidence", [])[:8]}
        (l["prefer"] if rule["weight"] > 0 else l["deprioritize"]).append(rule)
    elif kind == "topic":                                                                   # a methods-curriculum topic: stored in your interests.yaml
        t = methods.normalise_topic(pl)
        if t is None:
            raise ValueError("invalid methods topic")
        mine = list(config.load_interests(settings.interests_path).get("methods_topics") or [])
        if all(x.get("id") != t["id"] for x in mine):
            mine.append(t)
            config.save_key(settings.interests_path, "methods_topics", mine)
        return {"added": t["id"]}
    else:                                                                                  # follow: acknowledged; you follow via the search box
        return None
    config.save_key(settings.interests_path, "learned", l)
    return before


def undo_proposal(settings, prop: dict) -> None:
    before = prop.get("before")
    if before is None:
        return
    if prop["kind"] == "topic":
        mine = [x for x in (config.load_interests(settings.interests_path).get("methods_topics") or []) if x.get("id") != prop["payload"].get("id")]
        config.save_key(settings.interests_path, "methods_topics", mine)
        return
    l = _learned_now(settings)
    if prop["kind"] == "preferences":
        l["preferences"] = before["preferences"]
    elif prop["kind"] == "rule":
        pl = prop["payload"]
        key = (pl["kind"], str(pl["value"]).lower())
        same = lambda r: (r.get("kind"), str(r.get("value", "")).lower()) == key
        l["prefer"] = [r for r in l["prefer"] if not same(r)] + before["prefer"]
        l["deprioritize"] = [r for r in l["deprioritize"] if not same(r)] + before["deprioritize"]
    config.save_key(settings.interests_path, "learned", l)
