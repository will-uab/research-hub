"""Generate the hub page's JSON data files (and copy the static web app) after each stream run.

site/
  index.html app.js style.css          the static app (copied from web/, always refreshed)
  data/items.json                      last 14 days + open grants with a future deadline (default view, small)
  data/items-90d.json                  the full rolling 90-day window (loaded on demand)
  data/abstracts/YYYY-MM.json          id -> abstract/summary, lazy-loaded when a card is expanded
  data/status.json                     per-stream run status, spend vs caps, domain/stream metadata

Older items stay in SQLite. Files are written atomically under a lock, so overlapping timers can't corrupt them.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import shutil
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from . import learned as learned_mod
from .budget import iso_week_bounds
from .config import Settings
from .db import DB

log = logging.getLogger(__name__)
WINDOW_DAYS = 90
RECENT_DAYS = 14
EVENT_DAYS = 45
WEB_FILES = ("index.html", "app.js", "style.css")
VISIBLE = ("sent", "shown", "notice")

STREAMS = [  # display order + labels
    {"id": "briefing", "label": "Briefing", "cadence_hours": 36},
    {"id": "podcast", "label": "Podcast", "cadence_hours": 24 * 4},
    {"id": "tuning", "label": "Tuning", "cadence_hours": 36},
    {"id": "journal_club", "label": "Journal Club", "cadence_hours": 24 * 8},
    {"id": "literature", "label": "Literature", "cadence_hours": 36},
    {"id": "people", "label": "People", "cadence_hours": 36},
    {"id": "grants", "label": "Grants", "cadence_hours": 36},
    {"id": "reporter", "label": "Funded Projects", "cadence_hours": 24 * 8},
    {"id": "trials", "label": "Trials", "cadence_hours": 36},
]


def _atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)


def web_build(web: Path) -> str:
    """Short content hash of the three web files: changes whenever any of them does."""
    import hashlib
    h = hashlib.sha1()
    for name in WEB_FILES:
        if (web / name).exists():
            h.update((web / name).read_bytes())
    return h.hexdigest()[:10]


def prune_ss(stream: str, ss: dict) -> dict:
    """Keep only what the cards display (keeps the phone payload small)."""
    if stream == "literature":
        a = ss.get("authors") or []
        return {k: v for k, v in {"authors": ", ".join(a[:3]) + (" et al." if len(a) > 3 else ""),
                                  "journal": ss.get("source_label") or ss.get("journal"), "is_preprint": ss.get("is_preprint"),
                                  "is_major": ss.get("is_major"), "links": ss.get("links"),
                                  "watched_authors": ss.get("watched_authors")}.items() if v}
    if stream == "trials":
        d = dict(ss)
        d["interventions"] = [{"type": i["type"], "name": i["name"]} for i in ss.get("interventions", [])]
        return d
    if stream == "journal_club":
        a = ss.get("authors") or []
        return {**ss, "authors": ", ".join(a[:3]) + (" et al." if len(a) > 3 else "")}
    return ss


def collect(db: DB, today: date, now: datetime, learned: dict | None = None) -> tuple[list[dict], dict[str, dict]]:
    """(items within the 90-day window, {shard: {id: abstract}})."""
    cutoff = (now - timedelta(days=WINDOW_DAYS)).isoformat()
    ev_cutoff = (now - timedelta(days=EVENT_DAYS)).isoformat()
    events: dict[int, list] = {}
    for e in db.events(since=ev_cutoff):
        events.setdefault(e["item_id"], []).append({"ts": e["ts"], "kind": e["kind"], "detail": json.loads(e["detail"])})
    saved = set(db.marks()["saved"])                  # things you saved stay on the page however old they get
    items, shards = [], {}
    for r in db.conn.execute(f"SELECT * FROM items WHERE status IN ({','.join('?' * len(VISIBLE))}) ORDER BY id", VISIBLE):
        ss = json.loads(r["stream_specific"])
        if r["stream"] in ("briefing", "podcast"):     # served separately as data/briefing.json / data/podcast.json
            continue
        if ss.get("migrated_from_v1"):        # pre-hub rows never stored authors/dates/takeaways; they stay in the digest archive
            continue
        upcoming = r["stream"] == "grants" and ss.get("kind") == "nofo" and (ss.get("close_date") or "") >= today.isoformat()
        if r["first_seen"] < cutoff and not upcoming and r["id"] not in saved:
            continue
        shard = (r["first_seen"] or "")[:7]
        it = {"id": r["id"], "stream": r["stream"], "source": r["source"], "source_id": r["source_id"], "title": r["title"], "url": r["url"],
              "date": r["date"] or "", "first_seen": r["first_seen"], "last_updated": r["last_updated"],
              "domains": json.loads(r["domains"]), "relevance": r["relevance"], "takeaway": r["takeaway"] or "",
              "fit_note": r["fit_note"] or "", "track": r["track"], "status": r["status"],
              "ss": prune_ss(r["stream"], ss), "events": events.get(r["id"], []), "shard": shard}
        if learned and (learned["prefer"] or learned["deprioritize"]):
            adj, why = learned_mod.adjust(learned_mod.features(r["title"], r["summary"] or "", ss), learned)
            if adj:
                it["adj"], it["adj_why"] = adj, why[:3]
        items.append({k: v for k, v in it.items() if v not in ("", None, [], {})} | {"id": r["id"]})
        if r["summary"]:
            shards.setdefault(shard, {})[str(r["id"])] = r["summary"]
    return items, shards


def build_briefing(db: DB) -> dict:
    """Latest daily and weekly briefing (written by the briefing stream), for the page's pop-up."""
    out = {"daily": None, "weekly": None}
    for r in db.conn.execute("SELECT * FROM items WHERE stream='briefing' ORDER BY id DESC"):
        ss = json.loads(r["stream_specific"])
        per = ss.get("period")
        if per in out and out[per] is None:
            out[per] = {"key": r["source_id"], "period": per, "title": r["title"], "date": ss.get("date"), "headline": ss.get("headline", ""),
                        "paragraphs": ss.get("paragraphs", []), "sections": ss.get("sections", []), "refs": ss.get("refs", []), "covers": ss.get("covers", {}),
                        "counts": ss.get("counts", {}), "model": ss.get("model", ""), "generated": r["first_seen"]}
    return out


def build_podcast(db: DB, site_dir: Path, limit: int = 7) -> dict:
    """The latest episodes whose audio file still exists, newest first, with the script (for the transcript)."""
    eps = []
    for r in db.conn.execute("SELECT * FROM items WHERE stream='podcast' ORDER BY id DESC LIMIT 30"):
        ss = json.loads(r["stream_specific"])
        if not ss.get("audio") or not (site_dir / ss["audio"]).exists():
            continue
        eps.append({"key": r["source_id"], "date": ss.get("date"), "title": r["title"], "headline": ss.get("headline", ""),
                    "audio": ss["audio"], "seconds": ss.get("seconds"), "bytes": ss.get("bytes"), "turns": ss.get("turns", []),
                    "hosts": ss.get("hosts", {}), "dive": ss.get("dive"), "model": ss.get("model", ""), "generated": r["first_seen"]})
        if len(eps) >= limit:
            break
    return {"episodes": eps}


def build_status(db: DB, settings: Settings, cfg: dict, today: date, now: datetime) -> dict:
    daily, weekly = settings.daily_budget, settings.weekly_extra
    sep = tuple(k for k, v in (cfg.get("streams") or {}).items() if v.get("separate_budget"))      # streams with their own budget
    wk0, wk1 = iso_week_bounds(today.isoformat())
    streams = []
    for meta in STREAMS:
        sid = meta["id"]
        scfg = (cfg.get("streams") or {}).get(sid, {})
        last = db.conn.execute("SELECT * FROM runs WHERE stream=? AND finished IS NOT NULL ORDER BY id DESC LIMIT 1", (sid,)).fetchone()
        ok = db.conn.execute("SELECT finished FROM runs WHERE stream=? AND ok=1 ORDER BY id DESC LIMIT 1", (sid,)).fetchone()
        stale = False
        if scfg.get("enabled", True) and not scfg.get("on_demand"):
            age = (now - datetime.fromisoformat(ok["finished"])).total_seconds() / 3600 if ok else None
            stale = age is None or age > meta["cadence_hours"]
        streams.append({
            "id": sid, "label": meta["label"], "enabled": scfg.get("enabled", True), "schedule": scfg.get("schedule", ""), "on_demand": bool(scfg.get("on_demand")),
            "last_run": last["finished"] if last else None, "last_ok": ok["finished"] if ok else None,
            "last_run_ok": bool(last["ok"]) if last else None, "error": (last["error"] or "") if last and not last["ok"] else "",
            "counts": json.loads(last["counts"]) if last else {}, "notes": json.loads(last["notes"]) if last else [],
            "stale": stale, "never_run": last is None,
            "spend_today": round(db.spent(today.isoformat(), sid), 5), "cap": scfg.get("daily_cap_usd")})
    spend_log = [{"day": r["day"], "stream": r["stream"], "calls": r["calls"], "cost": round(r["cost_usd"], 5)}
                 for r in db.conn.execute("SELECT * FROM spend WHERE day>=? ORDER BY day DESC, stream",
                                          ((today - timedelta(days=14)).isoformat(),)) if r["stream"] not in sep]
    return {
        "generated": now.isoformat(timespec="seconds"), "today": today.isoformat(),
        "domains": [{"id": d, "label": c.get("label", d), "weight": c.get("weight")} for d, c in cfg["domains"].items()],
        "streams": streams, "deadline_flag_days": cfg["streams"].get("grants", {}).get("deadline_flag_days", 60),
        "spend": {"today": round(db.spent(today.isoformat(), exclude=sep), 5), "daily_cap": daily,
                  "separate": {k: {"today": round(db.spent(today.isoformat(), k), 5), "cap": cfg["streams"][k].get("daily_cap_usd")} for k in sep},
                  "week": round(db.spent_between(wk0, wk1, exclude=sep), 5), "week_journal_club": round(db.spent_between(wk0, wk1, "journal_club"), 5),
                  "weekly_extra": weekly, "log": spend_log}}


def regenerate(settings: Settings, cfg: dict, db: DB, out_dir: Path, now: datetime | None = None) -> dict:
    """Write the data files (and refresh the static app) into `out_dir`. Returns summary counts."""
    now = now or datetime.now(timezone.utc)
    today = now.astimezone(settings.tz).date()
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        web = settings.root / "web"
        build = web_build(web)
        for name in WEB_FILES:
            if (web / name).exists():
                if name == "index.html":      # stamp the asset URLs, so a changed script/style is never served from a stale cache
                    (out_dir / name).write_text((web / name).read_text().replace("__BUILD__", build))
                else:
                    shutil.copy(web / name, out_dir / name)
        manifest = out_dir / "manifest.json"                 # the daily-digest archive keeps its own index page
        if manifest.exists():
            from .. import publish
            publish.write_index(out_dir, publish._load(out_dir))
        items, shards = collect(db, today, now, learned_mod.normalise(cfg.get("learned")))
        recent_cut = (now - timedelta(days=RECENT_DAYS)).isoformat()
        recent = [i for i in items if i["first_seen"] >= recent_cut or (i["stream"] == "grants"
                  and (i["ss"].get("close_date") or "") >= today.isoformat())]
        meta = {"generated": now.isoformat(timespec="seconds"), "window_days": WINDOW_DAYS}
        data = out_dir / "data"
        _atomic_json(data / "items.json", {**meta, "recent_only": True, "items": recent})
        _atomic_json(data / "items-90d.json", {**meta, "recent_only": False, "items": items})
        for shard, m in shards.items():
            _atomic_json(data / "abstracts" / f"{shard}.json", m)
        _atomic_json(data / "status.json", build_status(db, settings, cfg, today, now))
        _atomic_json(data / "briefing.json", build_briefing(db))
        _atomic_json(data / "podcast.json", build_podcast(db, out_dir))
    log.info("Page data written to %s: %d items (%d recent), %d abstract shard(s)", out_dir, len(items), len(recent), len(shards))
    return {"items": len(items), "recent": len(recent), "shards": len(shards)}
