"""Web delivery: dated digest pages + an index in site/, served over Tailscale."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from html import escape
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from .digest import TEMPLATES, Digest

log = logging.getLogger(__name__)
MANIFEST = "manifest.json"
KEEP_DAYS = 90


def _load(site: Path) -> list[dict]:
    try:
        return json.loads((site / MANIFEST).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _free_name(site: Path, stem: str) -> str:
    """digest-DATE.html, or -2, -3 ... so a second run the same day never overwrites the first."""
    name, n = f"{stem}.html", 2
    while (site / name).exists():
        name, n = f"{stem}-{n}.html", n + 1
    return name


def write_index(site: Path, entries: list[dict]) -> None:
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=True)
    html = env.get_template("index.html.j2").render(entries=entries)
    _atomic(site / "digests.html", html)


def _atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _prune(site: Path, entries: list[dict], now: datetime, keep_days: int) -> list[dict]:
    cutoff = (now - timedelta(days=keep_days)).isoformat()
    keep = []
    for e in entries:
        if e["iso"] < cutoff:
            (site / e["file"]).unlink(missing_ok=True)
        else:
            keep.append(e)
    return keep


def _add(site: Path, entry: dict, now: datetime, keep_days: int) -> None:
    entries = _prune(site, _load(site), now, keep_days) + [entry]
    entries.sort(key=lambda e: e["iso"], reverse=True)
    _atomic(site / MANIFEST, json.dumps(entries, indent=1))
    write_index(site, entries)


def publish(site: Path, d: Digest, date_str: str, now: datetime, keep_days: int = KEEP_DAYS) -> Path:
    """Write the digest page, then update manifest + index. Returns the page path."""
    site.mkdir(parents=True, exist_ok=True)
    name = _free_name(site, date_str)
    _atomic(site / name, d.html)
    _add(site, {"file": name, "subject": d.subject, "kind": "digest", "iso": now.isoformat(timespec="seconds"),
                "published": now.strftime("%a %b %-d, %-I:%M %p"), "n_mine": len(d.mine) + len(d.notices),
                "n_major": len(d.major)}, now, keep_days)
    log.info("Published %s", site / name)
    return site / name


def publish_failure(site: Path, date_str: str, err: str, now: datetime) -> Path:
    site.mkdir(parents=True, exist_ok=True)
    name = _free_name(site, f"failed-{date_str}")
    body = (f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>Lit digest FAILED {date_str}</title></head>"
            f'<body style="font-family:-apple-system,Helvetica,Arial,sans-serif;padding:16px;max-width:720px;margin:auto;">'
            f'<a href="digests.html">&larr; Daily digests</a><h1 style="color:#a02323;font-size:20px;">Run failed · {date_str}</h1>'
            f'<pre style="white-space:pre-wrap;">{escape(err)}</pre><p>See logs/litsearch.log on the Pi.</p></body></html>')
    _atomic(site / name, body)
    _add(site, {"file": name, "subject": f"Lit digest FAILED {date_str}", "kind": "failure",
                "iso": now.isoformat(timespec="seconds"), "published": now.strftime("%a %b %-d, %-I:%M %p"),
                "n_mine": 0, "n_major": 0}, now, KEEP_DAYS)
    return site / name
