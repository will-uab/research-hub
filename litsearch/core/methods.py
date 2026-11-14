"""The methods curriculum for the podcast's teaching segment: loading, choosing a topic, remembering what was covered.

Topics come from config/methods_topics.yaml (the shared starting list) plus `methods_topics` in your interests.yaml (additions you
approved in the Tuning panel). The choice for a day is: what you asked for; else the topic whose keywords best match the day's items
(so the lesson can refer to a paper of the day) and that has not been covered recently; else the least recently covered topic.
"""
from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path

import yaml

ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,47}$")
AREA_CAP, TITLE_CAP, SCOPE_CAP = 40, 110, 600


def normalise_topic(d: dict) -> dict | None:
    """A validated topic, or None. Everything is bounded: these come from files and from model suggestions."""
    if not isinstance(d, dict):
        return None
    tid, title = str(d.get("id", "")).strip().lower(), re.sub(r"\s+", " ", str(d.get("title", ""))).strip()
    kws = [re.sub(r"\s+", " ", str(k)).strip() for k in (d.get("keywords") or []) if str(k).strip()] if isinstance(d.get("keywords"), list) else []
    if not ID_RE.match(tid) or not title or len(kws) < 1:
        return None
    out = {"id": tid, "title": title[:TITLE_CAP], "area": re.sub(r"\s+", " ", str(d.get("area", "General"))).strip()[:AREA_CAP] or "General",
           "keywords": [k[:40] for k in kws[:14]], "scope": re.sub(r"\s+", " ", str(d.get("scope", ""))).strip()[:SCOPE_CAP]}
    if d.get("worked_example") in ("confusion",):
        out["worked_example"] = d["worked_example"]
    return out


def slugify(text: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", text.lower())).strip("-")[:48]


def load_curriculum(root: Path, cfg: dict | None = None, base_file: str = "config/methods_topics.yaml") -> list[dict]:
    """Shared starting list, then your approved additions (an id already present is not replaced)."""
    raw: list = []
    f = root / base_file
    if f.exists():
        raw += (yaml.safe_load(f.read_text()) or {}).get("topics") or []
    raw += (cfg or {}).get("methods_topics") or []
    out, seen = [], set()
    for d in raw:
        t = normalise_topic(d)
        if t and t["id"] not in seen:
            seen.add(t["id"])
            out.append(t)
    return out


def covered(db, today: date, days: int) -> dict[str, date]:
    """topic id -> date it was last covered, within `days` days."""
    since = (today - timedelta(days=days)).isoformat()
    out: dict[str, date] = {}
    for r in db.conn.execute("SELECT stream_specific FROM items WHERE stream='podcast' AND date>=? ORDER BY id", (since,)):
        m = (json.loads(r["stream_specific"]).get("methods") or {})
        if m.get("topic_id") and m.get("date"):
            out[m["topic_id"]] = date.fromisoformat(m["date"])
    return out


def _hits(keyword: str, text: str) -> int:
    short_caps = len(keyword) <= 4 and keyword.isupper()
    pat = rf"(?<![A-Za-z0-9]){re.escape(keyword)}(?![A-Za-z0-9])"
    return len(re.findall(pat, text, 0 if short_caps else re.I))


def score_item(topic: dict, row: dict) -> float:
    """How strongly one item points at a topic: keyword hits (title counts double), scaled by the item's relevance."""
    title, body = row.get("title") or "", f"{row.get('takeaway') or ''} {row.get('summary') or ''}"
    s = 0.0
    for k in topic["keywords"]:
        s += 2 * min(1, _hits(k, title)) + min(2, _hits(k, body))
    return s * ((row.get("relevance") if row.get("relevance") is not None else 3) / 10.0 + 0.3)


def linked_items(topic: dict, rows: list[dict], n: int = 2, min_score: float = 1.0) -> list[dict]:
    scored = sorted(((score_item(topic, r), r) for r in rows), key=lambda x: (-x[0], x[1]["id"]))
    return [r for s, r in scored[:n] if s >= min_score]


def choose(curriculum: list[dict], rows: list[dict], cov: dict[str, date], requested: str = "auto", repeat_days: int = 60,
           min_link_score: float = 2.0) -> tuple[dict | None, str]:
    """(topic or None, how it was chosen): requested | custom | linked | rotation | none."""
    req = (requested or "auto").strip()
    if req == "none":
        return None, "none"
    by_id = {t["id"]: t for t in curriculum}
    if req in by_id:
        return by_id[req], "requested"
    if req.startswith("custom:") and req[7:].strip():
        text = re.sub(r"\s+", " ", req[7:]).strip()[:120]
        return {"id": "custom-" + (slugify(text) or "topic"), "title": text, "area": "Requested", "keywords": [text],
                "scope": "A topic you asked for: cover it at textbook level, starting from the basics, and say clearly where practice varies."}, "custom"
    fresh = [t for t in curriculum if t["id"] not in cov]
    best, best_score = None, 0.0
    for t in fresh:
        s = sum(sorted((score_item(t, r) for r in rows), reverse=True)[:2])
        if s > best_score:
            best, best_score = t, s
    if best and best_score >= min_link_score:
        return best, "linked"
    pool = sorted(curriculum, key=lambda t: (cov.get(t["id"], date.min), curriculum.index(t)))
    return (pool[0], "rotation") if pool else (None, "none")
