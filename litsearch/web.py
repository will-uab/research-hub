"""The hub's web server: static files from site/ plus a small JSON API (ratings, saved/read, watch list).

Bound to 127.0.0.1 and exposed to the tailnet only by `tailscale serve` (unchanged). Because a browser on your phone can
be tricked into sending requests to any host it can reach, state-changing calls are protected against cross-site requests:

  * the Host header must be an allowed name (blocks DNS rebinding);
  * POST needs `Content-Type: application/json` AND a custom `X-LH` header, which forces a CORS preflight that this
    server never approves, so other websites cannot send them;
  * if an Origin header is present it must match the Host;
  * request bodies are limited to 8 KB.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import mimetypes
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .core import config, methods, pagedata, tuning, watch
from .core.db import DB

log = logging.getLogger("litsearch.web")
MAX_BODY = 8192
API_VERSION = 3      # 2 = reasons + tuning endpoints; 3 = on-demand podcast; the page hides features an older server lacks
PODCAST_STALE_MIN = 45   # a run marker older than this is treated as dead (the pipeline's own timeout is 30 min)
REASONS = {-1: ("not_topic", "already_knew", "weak_study", "wrong_population", "too_basic"),
           1: ("on_topic", "great_method", "useful_project", "follow_person")}
RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)$")
DEFAULT_HOSTS = ("localhost", "127.0.0.1", "[::1]")


def parse_range(header: str, size: int) -> tuple[int, int] | None:
    """'bytes=a-b' / 'bytes=a-' / 'bytes=-n' -> inclusive (start, end) within a file of `size` bytes; None if unsatisfiable."""
    m = RANGE_RE.match((header or "").strip())
    if not m or (not m.group(1) and not m.group(2)) or size <= 0:
        return None
    if not m.group(1):                                              # suffix: the last n bytes
        n = int(m.group(2))
        return (max(0, size - n), size - 1) if n > 0 else None
    start = int(m.group(1))
    end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
    return (start, end) if start <= end and start < size else None


def host_allowed(host_header: str, extra: tuple[str, ...] = ()) -> bool:
    host = (host_header or "").rsplit(":", 1)[0] if not (host_header or "").startswith("[") else (host_header or "").split("]")[0] + "]"
    host = host.lower()
    return bool(host) and (host in DEFAULT_HOSTS or host in extra or host.endswith(".ts.net"))


class Api:
    """Pure request logic (no sockets), so it is directly testable."""

    def __init__(self, settings, search=watch.search_candidates, regenerate=None, spawn=None):
        self.settings = settings
        self.search = search
        self.regenerate = regenerate
        self.spawn = spawn or self._spawn_podcast
        self._podcast_lock = threading.Lock()

    # ------------------------------------------------------------------ on-demand podcast
    @property
    def _run_marker(self):
        return self.settings.root / "data" / "podcast-run.json"

    def podcast_running(self) -> dict | None:
        """The marker of a podcast run that is still alive, else None."""
        try:
            m = json.loads(self._run_marker.read_text())
            if (datetime.now(timezone.utc) - datetime.fromisoformat(m["started"])).total_seconds() > PODCAST_STALE_MIN * 60:
                return None
            os.kill(int(m["pid"]), 0)
            return m
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _spawn_podcast(self, redo: bool, topic: str = "auto") -> int:
        logs = self.settings.root / "logs"
        logs.mkdir(exist_ok=True)
        out = open(logs / "podcast-ondemand.out", "ab")
        proc = subprocess.Popen([sys.executable, "-m", "litsearch", "--stream", "podcast", f"--methods-topic={topic}"] + (["--redo"] if redo else []),
                                cwd=str(self.settings.root), stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)

        def reap():
            proc.wait()
            out.close()
            try:
                if json.loads(self._run_marker.read_text()).get("pid") == proc.pid:
                    self._run_marker.unlink()
            except (OSError, ValueError):
                pass
        threading.Thread(target=reap, daemon=True).start()
        return proc.pid

    def _podcast_state(self) -> dict:
        full = config.load_interests(self.settings.interests_path)
        cfg = full["streams"].get("podcast", {})
        today = datetime.now(self.settings.tz).date()
        day = today.isoformat()
        db = self._db()
        try:
            cov = methods.covered(db, today, 365)
            topics = [{"id": t["id"], "title": t["title"], "area": t["area"], "days_ago": (today - cov[t["id"]]).days if t["id"] in cov else None}
                      for t in methods.load_curriculum(self.settings.root, full, cfg.get("methods_file", "config/methods_topics.yaml"))]
            row = db.conn.execute("SELECT stream_specific FROM items WHERE stream='podcast' AND source_id=?", (f"daily:{day}",)).fetchone()
            have = bool(row and json.loads(row["stream_specific"]).get("audio") and (self.settings.site_dir / json.loads(row["stream_specific"])["audio"]).exists())
            last = db.conn.execute("SELECT ok, finished, notes, error FROM runs WHERE stream='podcast' AND finished IS NOT NULL ORDER BY id DESC LIMIT 1").fetchone()
            run = self.podcast_running()
            return {"enabled": cfg.get("enabled", True), "running": bool(run), "started": run["started"] if run else None,
                    "episode_today": have, "spend_today": round(db.spent(day, "podcast"), 4), "cap": cfg.get("daily_cap_usd"),
                    "est_cost": cfg.get("est_cost_usd", 0.3), "methods": cfg.get("methods", True), "topics": topics,
                    "last": ({"ok": bool(last["ok"]), "finished": last["finished"], "notes": json.loads(last["notes"] or "[]"),
                              "error": last["error"] or ""} if last else None)}
        finally:
            db.close()

    def _podcast_generate(self, redo: bool, topic: str = "auto") -> tuple[int, dict]:
        topic = re.sub(r"[\x00-\x1f]", " ", topic).strip()
        with self._podcast_lock:
            st = self._podcast_state()
            if topic.startswith("custom:"):
                topic = "custom:" + re.sub(r"\s+", " ", topic[7:]).strip()[:120]
                if len(topic) <= 7:
                    return 400, {"error": "enter a topic, or choose Automatic"}
            elif topic not in ("auto", "none") and topic not in {t["id"] for t in st["topics"]}:
                return 400, {"error": "unknown topic"}
            if not st["enabled"]:
                return 409, {"error": "the podcast is disabled in interests.yaml"}
            if st["running"]:
                return 409, {"error": "an episode is already being generated", "running": True}
            if st["episode_today"] and not redo:
                return 409, {"error": "today's episode already exists; use Regenerate to replace it", "exists": True}
            started = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self._run_marker.parent.mkdir(exist_ok=True)
            self._run_marker.write_text(json.dumps({"pid": os.getpid(), "started": started}))        # claim the slot before spawning
            try:
                pid = self.spawn(redo, topic)
            except Exception as e:  # noqa: BLE001
                self._run_marker.unlink(missing_ok=True)
                log.exception("could not start the podcast run")
                return 500, {"error": f"could not start: {type(e).__name__}: {e}"}
            self._run_marker.write_text(json.dumps({"pid": pid, "started": started}))
            return 202, {"ok": True, "started": started, "redo": redo, "topic": topic}

    def _regenerate(self) -> None:
        """Rebuild the page data so an accepted preference re-ranks the page straight away."""
        cfg = config.load_interests(self.settings.interests_path)
        db = self._db()
        try:
            pagedata.regenerate(self.settings, cfg, db, self.settings.site_dir)
        finally:
            db.close()

    def _tuning_state(self) -> dict:
        db = self._db()
        try:
            rated = db.rated()
            last = db.latest_report()
            cfgd = config.load_interests(self.settings.interests_path)["streams"].get("tuning", {})
            titles = {r["id"]: r["title"] for r in db.conn.execute("SELECT id, title FROM items")}
            ratings = db.ratings()

            def view(p):
                p["evidence_items"] = [{"id": i, "title": titles.get(i, "(item no longer available)"), "sign": ratings.get(i, 0)} for i in p["evidence"][:6]]
                return p
            new = [r for r in rated if r["fid"] > (last["last_feedback_id"] if last else 0)]
            return {"pending": [view(p) for p in db.proposals(("pending",), 60)[::-1]],
                    "history": [view(p) for p in db.proposals(("accepted", "rejected", "undone"), 15)],
                    "ratings": len(rated), "new_ratings": len(new),
                    "need": cfgd.get("first_after", 10) if not last else cfgd.get("every", 20),
                    "notes": (last["notes"] if last else ""), "report_ts": (last["ts"] if last else None)}
        finally:
            db.close()

    def _db(self) -> DB:
        return DB(self.settings.db_path)

    def get(self, path: str, query: dict) -> tuple[int, dict]:
        if path == "/api/ping":
            return 200, {"ok": True, "v": API_VERSION}
        if path == "/api/state":
            db = self._db()
            try:
                return 200, {"ratings": {str(k): v for k, v in db.ratings().items()}, **db.marks(), "acked": db.acks()}
            finally:
                db.close()
        if path == "/api/tuning":
            return 200, self._tuning_state()
        if path == "/api/podcast":
            return 200, self._podcast_state()
        if path == "/api/watch":
            return 200, {"people": watch.load(self.settings)}
        if path == "/api/watch/search":
            q = (query.get("q") or [""])[0]
            try:
                return 200, self.search(q)
            except ValueError as e:
                return 400, {"error": str(e)}
        return 404, {"error": "not found"}

    def post(self, path: str, body: dict) -> tuple[int, dict]:
        if path in ("/api/rate", "/api/mark", "/api/reason"):
            try:
                item_id = int(body["id"])
            except (KeyError, TypeError, ValueError):
                return 400, {"error": "id required"}
            db = self._db()
            try:
                if not db.item_exists(item_id):
                    return 404, {"error": "no such item"}
                if path == "/api/rate":
                    sign = body.get("sign")
                    if sign not in (-1, 0, 1):
                        return 400, {"error": "sign must be -1, 0 or 1"}
                    db.set_rating(item_id, sign)
                    return 200, {"ok": True, "id": item_id, "sign": sign}
                if path == "/api/reason":
                    reason = body.get("reason")
                    if reason is not None and reason not in REASONS[-1] + REASONS[1]:
                        return 400, {"error": "unknown reason"}
                    if not db.set_reason(item_id, reason):
                        return 409, {"error": "rate the item first"}
                    return 200, {"ok": True, "id": item_id, "reason": reason}
                saved, read = body.get("saved"), body.get("read")
                if not all(isinstance(v, bool) or v is None for v in (saved, read)) or (saved is None and read is None):
                    return 400, {"error": "saved and/or read (booleans) required"}
                db.set_mark(item_id, saved=saved, read=read)
                return 200, {"ok": True, "id": item_id, **{k: v for k, v in (("saved", saved), ("read", read)) if v is not None}}
            finally:
                db.close()
        if path == "/api/podcast/generate":
            topic = body.get("topic", "auto")
            if not isinstance(topic, str) or len(topic) > 200:
                return 400, {"error": "topic must be a short string"}
            return self._podcast_generate(body.get("redo") is True, topic)
        if path == "/api/tuning/decide":
            try:
                pid, action = int(body["id"]), body.get("action")
            except (KeyError, TypeError, ValueError):
                return 400, {"error": "id and action required"}
            if action not in ("accept", "reject", "undo"):
                return 400, {"error": "action must be accept, reject or undo"}
            db = self._db()
            try:
                p = db.get_proposal(pid)
                if not p:
                    return 404, {"error": "no such proposal"}
                if action in ("accept", "reject") and p["status"] != "pending":
                    return 409, {"error": f"already {p['status']}"}
                if action == "undo" and p["status"] != "accepted":
                    return 409, {"error": "only accepted proposals can be undone"}
                if action == "accept":
                    db.decide_proposal(pid, "accepted", tuning.apply_proposal(self.settings, p) or {})
                elif action == "reject":
                    db.decide_proposal(pid, "rejected")
                else:
                    tuning.undo_proposal(self.settings, p)
                    db.decide_proposal(pid, "undone")
            finally:
                db.close()
            if action != "reject":
                try:
                    (self.regenerate or self._regenerate)()
                except Exception:  # noqa: BLE001 - the YAML is already updated; the page catches up on the next run
                    log.exception("page data regeneration after tuning decision failed")
            return 200, {"ok": True, "id": pid, "action": action}
        if path == "/api/ack":
            key = str(body.get("key", ""))
            if not re.fullmatch(r"(daily|weekly):\d{4}-(\d{2}-\d{2}|W\d{2})", key):
                return 400, {"error": "bad briefing key"}
            db = self._db()
            try:
                db.ack(key)
            finally:
                db.close()
            return 200, {"ok": True, "key": key}
        if path == "/api/watch/add":
            try:
                people, added = watch.add(self.settings, body)
            except (ValueError, TypeError) as e:
                return 400, {"error": str(e)}
            return 200, {"ok": True, "added": added, "people": people}
        if path == "/api/watch/remove":
            people, n = watch.remove(self.settings, body.get("name"), body.get("profile_id"), body.get("openalex_id"))
            return 200, {"ok": True, "removed": n, "people": people}
        return 404, {"error": "not found"}


def make_handler(site_dir: Path, api: Api, extra_hosts: tuple[str, ...] = ()):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(site_dir), **kw)

        def log_message(self, fmt, *args):                      # quiet; journald has the app logs
            log.debug("%s %s", self.address_string(), fmt % args)

        def end_headers(self):
            if "Cache-Control" not in self._headers_buffer_text():                  # static files: revalidate, never serve blindly stale
                self.send_header("Cache-Control", "no-cache")
            super().end_headers()

        def _headers_buffer_text(self) -> str:
            return b"".join(getattr(self, "_headers_buffer", [])).decode("latin-1")

        def _json(self, code: int, obj: dict) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _host_ok(self) -> bool:
            if not host_allowed(self.headers.get("Host", ""), extra_hosts):
                self._json(403, {"error": "host not allowed"})
                return False
            return True

        def _send_audio(self, head_only: bool) -> bool:
            """Audio files are served with byte-range support (iOS needs it to play and seek). True when handled."""
            u = urlparse(self.path)
            if not u.path.endswith(".mp3") or any(part.startswith(".") for part in u.path.split("/") if part):
                return False
            fs = self.translate_path(u.path)
            if not os.path.isfile(fs):
                return False
            size = os.path.getsize(fs)
            rng = self.headers.get("Range")
            span = parse_range(rng, size) if rng else None
            if rng and span is None:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return True
            start, end = span if span else (0, size - 1)
            self.send_response(206 if span else 200)
            self.send_header("Content-Type", mimetypes.guess_type(fs)[0] or "audio/mpeg")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if span:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if not head_only:
                with open(fs, "rb") as f:
                    f.seek(start)
                    left = end - start + 1
                    while left > 0:
                        chunk = f.read(min(65536, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
            return True

        def do_GET(self):
            if not self._host_ok():
                return
            if self._send_audio(False):
                return
            u = urlparse(self.path)
            if u.path.startswith("/api/"):
                try:
                    code, obj = api.get(u.path, parse_qs(u.query))
                except Exception as e:  # noqa: BLE001
                    log.exception("API error on %s", u.path)
                    code, obj = 500, {"error": f"{type(e).__name__}: {e}"}
                return self._json(code, obj)
            if any(part.startswith(".") for part in u.path.split("/") if part):     # no dotfiles (.lock etc.)
                return self.send_error(404)
            super().do_GET()

        def do_HEAD(self):
            if self._host_ok() and not self._send_audio(True):
                super().do_HEAD()

        def do_POST(self):
            if not self._host_ok():
                return
            u = urlparse(self.path)
            host = self.headers.get("Host", "")
            origin = self.headers.get("Origin")
            if not u.path.startswith("/api/"):
                return self._json(404, {"error": "not found"})
            if not (self.headers.get("Content-Type", "").split(";")[0].strip().lower() == "application/json"
                    and self.headers.get("X-LH") == "1"):
                return self._json(403, {"error": "JSON content type and X-LH header required"})
            if origin:
                o_host = re.sub(r"^https?://", "", origin).lower()
                # same origin as Host, or one of your tailnet names (the proxy may rewrite Host); never another website
                if not (o_host == host.lower() or host_allowed(o_host, extra_hosts)):
                    return self._json(403, {"error": "cross-origin request refused"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._json(400, {"error": "bad Content-Length"})
            if n > MAX_BODY:
                return self._json(413, {"error": "body too large"})
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("object expected")
            except ValueError as e:
                return self._json(400, {"error": f"bad JSON: {e}"})
            try:
                code, obj = api.post(u.path, body)
            except Exception as e:  # noqa: BLE001
                log.exception("API error on %s", u.path)
                code, obj = 500, {"error": f"{type(e).__name__}: {e}"}
            self._json(code, obj)

    return Handler


def serve(settings, bind: str = "127.0.0.1", port: int = 8081, extra_hosts: tuple[str, ...] = ()) -> ThreadingHTTPServer:
    settings.site_dir.mkdir(exist_ok=True)
    srv = ThreadingHTTPServer((bind, port), make_handler(settings.site_dir, Api(settings), extra_hosts))
    srv.daemon_threads = True
    return srv


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="litsearch.web")
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8081)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = config.load_settings(require_env=False)
    extra = tuple(h.strip().lower() for h in settings.get("LIT_ALLOWED_HOSTS", "").split(",") if h.strip())
    srv = serve(settings, args.bind, args.port, extra)
    log.info("Serving %s on http://%s:%d (API enabled)", settings.site_dir, args.bind, args.port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
