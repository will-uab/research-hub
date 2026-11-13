"""SQLite state: unified items (dedup per stream), events, feedback, per-stream spend, run log."""
from __future__ import annotations

import json
import logging
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .models import norm_doi, title_key

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stream TEXT NOT NULL, source TEXT NOT NULL, source_id TEXT NOT NULL,
    title TEXT NOT NULL, url TEXT, date TEXT, summary TEXT,
    domains TEXT NOT NULL DEFAULT '[]',          -- JSON list of domain ids
    relevance REAL, takeaway TEXT, fit_note TEXT,
    stream_specific TEXT NOT NULL DEFAULT '{}',  -- JSON: deadlines, NCT status, award amounts, authors...
    status TEXT NOT NULL,                        -- sent | screened | notice | shown
    track TEXT, kw_score REAL DEFAULT 0,
    first_seen TEXT NOT NULL, last_updated TEXT NOT NULL,
    UNIQUE (stream, source, source_id)
);
CREATE INDEX IF NOT EXISTS ix_items_stream_date ON items(stream, first_seen);

CREATE TABLE IF NOT EXISTS item_keys (          -- alternate identifiers for dedup (pmid/doi/arxiv/title...)
    item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    stream TEXT NOT NULL, kind TEXT NOT NULL, value TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_keys ON item_keys(stream, kind, value);

CREATE TABLE IF NOT EXISTS item_events (        -- results_posted, status_change, nofo_modified, reissued ...
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    ts TEXT NOT NULL, kind TEXT NOT NULL, detail TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS ix_events_item ON item_events(item_id, ts);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id INTEGER NOT NULL REFERENCES items(id),
    sign INTEGER NOT NULL,                       -- +1 like, -1 dislike, 0 cleared
    ts TEXT NOT NULL,
    reason TEXT,                                 -- optional one-tap reason (see REASONS)
    snapshot TEXT                                -- JSON: the item as it looked when rated (incl. the LLM's score then)
);

CREATE TABLE IF NOT EXISTS spend (
    day TEXT NOT NULL, stream TEXT NOT NULL,
    calls INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens INTEGER NOT NULL DEFAULT 0, cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (day, stream)
);

CREATE TABLE IF NOT EXISTS runs (               -- feeds the page's status banner
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stream TEXT NOT NULL, started TEXT NOT NULL, finished TEXT,
    ok INTEGER, error TEXT, counts TEXT NOT NULL DEFAULT '{}', notes TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS ix_runs ON runs(stream, started);

CREATE TABLE IF NOT EXISTS marks (               -- your saved / read flags (ratings live in `feedback`)
    item_id INTEGER PRIMARY KEY REFERENCES items(id) ON DELETE CASCADE,
    saved INTEGER NOT NULL DEFAULT 0, read INTEGER NOT NULL DEFAULT 0, ts TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS acks (                -- briefings you have acknowledged ("Got it"), e.g. daily:2026-11-05
    key TEXT PRIMARY KEY, ts TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tuning_reports (       -- one per tune-up
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, n_ratings INTEGER, last_feedback_id INTEGER,
    notes TEXT, model TEXT
);
CREATE TABLE IF NOT EXISTS proposals (            -- what a tune-up suggests; nothing applies until you accept
    id INTEGER PRIMARY KEY AUTOINCREMENT, report_id INTEGER REFERENCES tuning_reports(id),
    kind TEXT NOT NULL,                           -- preferences | rule | follow
    payload TEXT NOT NULL, why TEXT, evidence TEXT NOT NULL DEFAULT '[]', backtest TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending',       -- pending | accepted | rejected | undone
    decided TEXT, before TEXT
);

CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _j(v) -> str:
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))


class DB:
    def __init__(self, path: Path | str):
        is_file = str(path) != ":memory:"
        if is_file:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(str(path), timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA busy_timeout=30000")
        if is_file:
            self.conn.execute("PRAGMA journal_mode=WAL")   # timers may overlap briefly
        legacy = self._has_table("papers") and not self._has_table("items")
        if legacy and is_file:
            bak = Path(str(path) + ".v1.bak")
            if not bak.exists():
                self.conn.commit()
                shutil.copy(path, bak)
                log.info("Backed up v1 database to %s", bak)
        self.conn.executescript(SCHEMA)
        if legacy:
            self._migrate_v1()
        self._ensure_columns()

    def close(self) -> None:
        self.conn.close()

    def _ensure_columns(self) -> None:
        """Add columns introduced after a database was first created."""
        have = {r[1] for r in self.conn.execute("PRAGMA table_info(feedback)")}
        for col in ("reason", "snapshot"):
            if col not in have:
                self.conn.execute(f"ALTER TABLE feedback ADD COLUMN {col} TEXT")
        self.conn.commit()

    @staticmethod
    def copy_file(src: Path | str, dest: Path | str) -> None:
        """Consistent copy (includes WAL contents) via SQLite's backup API; for throwaway dry-run DBs."""
        s = sqlite3.connect(str(src))
        d = sqlite3.connect(str(dest))
        with d:
            s.backup(d)
        s.close()
        d.close()

    def _has_table(self, name: str) -> bool:
        return self.conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None

    # ------------------------------------------------------------ v1 -> unified migration
    def _migrate_v1(self) -> None:
        """papers/feedback/spend (literature only) -> items/feedback/spend. Ids are preserved."""
        c = self.conn
        n = 0
        with c:
            for r in c.execute("SELECT * FROM papers ORDER BY id").fetchall():
                if r["pmid"]:
                    sid = r["pmid"]
                elif r["arxiv_id"]:
                    sid = f"arxiv:{r['arxiv_id']}"
                elif r["doi"]:
                    sid = f"doi:{r['doi']}"
                else:
                    sid = f"title:{(r['title_key'] or '')[:40]}"
                url = (f"https://doi.org/{r['doi']}" if r["doi"] else
                       f"https://pubmed.ncbi.nlm.nih.gov/{r['pmid']}/" if r["pmid"] else
                       f"https://arxiv.org/abs/{r['arxiv_id']}" if r["arxiv_id"] else "")
                ss = {"journal": r["journal"], "is_preprint": bool(r["is_preprint"]), "pmid": r["pmid"],
                      "doi": r["doi"], "arxiv_id": r["arxiv_id"], "published_doi": r["published_doi"],
                      "migrated_from_v1": True}
                try:
                    c.execute(
                        """INSERT INTO items (id, stream, source, source_id, title, url, date, summary, domains,
                               relevance, takeaway, fit_note, stream_specific, status, track, kw_score, first_seen, last_updated)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (r["id"], "literature", r["source"], sid, r["title"], url, "", "", "[]", r["relevance"], "", "",
                         _j(ss), r["status"], r["track"], 0, r["first_seen"], r["first_seen"]))
                except sqlite3.IntegrityError:
                    log.warning("v1 migration: skipped duplicate row %s (%s)", r["id"], sid)
                    continue
                for kind, val in (("pmid", r["pmid"]), ("doi", r["doi"]), ("arxiv", r["arxiv_id"]),
                                  ("pdoi", r["published_doi"]),
                                  ("title", r["title_key"] if len(r["title_key"] or "") >= 20 else None)):
                    if val:
                        c.execute("INSERT INTO item_keys (item_id, stream, kind, value) VALUES (?,?,?,?)",
                                  (r["id"], "literature", kind, val))
                n += 1
        # feedback/spend had incompatible shapes: copy from the renamed originals
        self._migrate_v1_side_tables()
        with c:
            c.execute("ALTER TABLE papers RENAME TO papers_v1")
        log.info("Migrated %d literature rows from v1 schema (old table kept as papers_v1)", n)

    def _migrate_v1_side_tables(self) -> None:
        c = self.conn
        # The v1 feedback/spend tables share names with the new ones; CREATE IF NOT EXISTS left the old
        # shapes in place, so detect and rebuild.
        fb_cols = [x[1] for x in c.execute("PRAGMA table_info(feedback)")]
        if "paper_id" in fb_cols:
            rows = c.execute("SELECT paper_id, sign, ts FROM feedback ORDER BY id").fetchall()
            with c:
                c.execute("ALTER TABLE feedback RENAME TO feedback_v1")
                c.execute("""CREATE TABLE feedback (id INTEGER PRIMARY KEY AUTOINCREMENT,
                             item_id INTEGER NOT NULL REFERENCES items(id), sign INTEGER NOT NULL, ts TEXT NOT NULL,
                             reason TEXT, snapshot TEXT)""")
                for r in rows:
                    c.execute("INSERT INTO feedback (item_id, sign, ts) VALUES (?,?,?)", tuple(r))
        sp_cols = [x[1] for x in c.execute("PRAGMA table_info(spend)")]
        if "stream" not in sp_cols:
            rows = c.execute("SELECT * FROM spend").fetchall()
            with c:
                c.execute("ALTER TABLE spend RENAME TO spend_v1")
                c.execute("""CREATE TABLE spend (day TEXT NOT NULL, stream TEXT NOT NULL,
                    calls INTEGER NOT NULL DEFAULT 0, input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0, cache_write_tokens INTEGER NOT NULL DEFAULT 0,
                    cache_read_tokens INTEGER NOT NULL DEFAULT 0, cost_usd REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY (day, stream))""")
                for r in rows:
                    c.execute("INSERT INTO spend VALUES (?,?,?,?,?,?,?,?)",
                              (r["day"], "literature", r["calls"], r["input_tokens"], r["output_tokens"],
                               r["cache_write_tokens"], r["cache_read_tokens"], r["cost_usd"]))

    # ------------------------------------------------------------------ dedup
    def find(self, item, stream: str | None = None) -> sqlite3.Row | None:
        """Existing item of the same stream (or of `stream`) matching any lookup key of `item`."""
        keys = item.lookup_keys()
        if not keys:
            return None
        clause = " OR ".join("(k.kind=? AND k.value=?)" for _ in keys)
        args = [stream or item.stream] + [x for kv in keys for x in kv]
        return self.conn.execute(
            f"""SELECT i.* FROM items i JOIN item_keys k ON k.item_id=i.id
                WHERE k.stream=? AND ({clause}) ORDER BY i.id LIMIT 1""", args).fetchone()

    def find_by_key(self, stream: str, kind: str, value: str) -> sqlite3.Row | None:
        return self.conn.execute(
            """SELECT i.* FROM items i JOIN item_keys k ON k.item_id=i.id
               WHERE k.stream=? AND k.kind=? AND k.value=? ORDER BY i.id LIMIT 1""", (stream, kind, value)).fetchone()

    def events(self, since: str | None = None, kinds: tuple[str, ...] | None = None) -> list[sqlite3.Row]:
        q, a = "SELECT * FROM item_events WHERE 1=1", []
        if since:
            q += " AND ts>=?"; a.append(since)
        if kinds:
            q += f" AND kind IN ({','.join('?' * len(kinds))})"; a += list(kinds)
        return self.conn.execute(q + " ORDER BY id", a).fetchall()

    def record(self, item, status: str) -> int:
        now = _now()
        cur = self.conn.execute(
            """INSERT INTO items (stream, source, source_id, title, url, date, summary, domains, relevance,
                   takeaway, fit_note, stream_specific, status, track, kw_score, first_seen, last_updated)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (item.stream, item.source, item.source_id, item.title, item.url, item.date, item.summary,
             _j(item.domains), item.relevance, item.takeaway, item.fit_note, _j(item.stream_specific),
             status, item.track, item.kw_score, now, now))
        iid = cur.lastrowid
        for kind, val in item.store_keys():
            self.conn.execute("INSERT INTO item_keys (item_id, stream, kind, value) VALUES (?,?,?,?)",
                              (iid, item.stream, kind, val))
        self.conn.commit()
        item.id = iid
        return iid

    def update_item(self, item_id: int, **fields) -> None:
        """Update columns; dict/list values are stored as JSON. Always bumps last_updated."""
        sets, args = ["last_updated=?"], [_now()]
        for k, v in fields.items():
            sets.append(f"{k}=?")
            args.append(_j(v) if isinstance(v, (dict, list)) else v)
        self.conn.execute(f"UPDATE items SET {', '.join(sets)} WHERE id=?", args + [item_id])
        self.conn.commit()

    def add_event(self, item_id: int, kind: str, detail: dict | None = None) -> None:
        self.conn.execute("INSERT INTO item_events (item_id, ts, kind, detail) VALUES (?,?,?,?)",
                          (item_id, _now(), kind, _j(detail or {})))
        self.conn.commit()

    def items(self, stream: str | None = None, statuses: tuple[str, ...] | None = None,
              since: str | None = None) -> list[sqlite3.Row]:
        q, a = "SELECT * FROM items WHERE 1=1", []
        if stream:
            q += " AND stream=?"; a.append(stream)
        if statuses:
            q += f" AND status IN ({','.join('?' * len(statuses))})"; a += list(statuses)
        if since:
            q += " AND last_updated>=?"; a.append(since)
        return self.conn.execute(q + " ORDER BY id", a).fetchall()

    def was_sent_pmid(self, pmid: str) -> bool:
        return self.conn.execute(
            """SELECT 1 FROM items i JOIN item_keys k ON k.item_id=i.id
               WHERE i.stream='literature' AND k.kind='pmid' AND k.value=? AND i.status='sent'""", (pmid,)
        ).fetchone() is not None

    # --------------------------------------------------------------- feedback
    def resolve_code(self, code: str) -> sqlite3.Row | None:
        """Feedback code (without sign): PMID, arxiv:ID, or doi:DOI / bare DOI. Literature items."""
        c = code.strip()
        if c.isdigit():
            kind, val = "pmid", c
        elif c.lower().startswith("arxiv:"):
            kind, val = "arxiv", c[6:]
        else:
            kind, val = "doi", norm_doi(c[4:] if c.lower().startswith("doi:") else c)
        return self.conn.execute(
            """SELECT i.* FROM items i JOIN item_keys k ON k.item_id=i.id
               WHERE i.stream='literature' AND k.kind=? AND k.value=? ORDER BY i.id LIMIT 1""", (kind, val)
        ).fetchone()

    def add_feedback(self, item_id: int, sign: int, reason: str | None = None, snapshot: dict | None = None) -> None:
        self.conn.execute("INSERT INTO feedback (item_id, sign, ts, reason, snapshot) VALUES (?,?,?,?,?)",
                          (item_id, sign, _now(), reason, _j(snapshot) if snapshot else None))
        self.conn.commit()

    def snapshot(self, item_id: int) -> dict:
        """The item as it looks right now (what a rating is a judgement of), incl. the LLM's score and the features rules match on."""
        r = self.conn.execute("SELECT * FROM items WHERE id=?", (item_id,)).fetchone()
        ss = json.loads(r["stream_specific"])
        authors = ss.get("authors") or ss.get("pis") or []
        if isinstance(authors, str):
            authors = [a.strip() for a in authors.replace(" et al.", "").split(",") if a.strip()]
        return {"stream": r["stream"], "title": r["title"], "relevance": r["relevance"], "takeaway": r["takeaway"] or "",
                "kw_score": r["kw_score"], "domains": json.loads(r["domains"]), "date": r["date"] or "",
                "journal": ss.get("source_label") or ss.get("journal") or "", "authors": authors[:8],
                "institutes": ss.get("institutes") or ([ss["institute"]] if ss.get("institute") else []),
                "mechanism": ss.get("mechanism") or "", "sponsor": ss.get("sponsor") or ss.get("institution") or "",
                "followed": ss.get("watched") or ss.get("watched_authors") or [],
                "is_preprint": bool(ss.get("is_preprint")), "ai_intervention": bool(ss.get("ai_intervention"))}

    def set_reason(self, item_id: int, reason: str | None) -> bool:
        """Attach a reason to the item's latest (non-cleared) rating."""
        row = self.conn.execute("SELECT id, sign FROM feedback WHERE item_id=? ORDER BY id DESC LIMIT 1", (item_id,)).fetchone()
        if not row or row["sign"] == 0:
            return False
        self.conn.execute("UPDATE feedback SET reason=? WHERE id=?", (reason, row["id"]))
        self.conn.commit()
        return True

    def recent_feedback(self, n: int, stream: str | None = None) -> list[tuple[int, str]]:
        """Short-term memory for prompts: latest verdict per item (cleared ratings ignored), newest first, BALANCED: up to n/2
        likes and n/2 dislikes (the other side fills any shortfall), so a run of one kind can't crowd out the other."""
        q = """SELECT f.id, f.sign, i.title FROM feedback f JOIN items i ON i.id=f.item_id
               WHERE f.id IN (SELECT MAX(id) FROM feedback GROUP BY item_id) AND f.sign != 0"""
        a: list = []
        if stream:
            q += " AND i.stream=?"
            a.append(stream)
        rows = self.conn.execute(q + " ORDER BY f.id DESC", a).fetchall()
        likes, dislikes = [r for r in rows if r["sign"] > 0], [r for r in rows if r["sign"] < 0]
        half = n // 2
        take_l = min(len(likes), max(half, n - min(len(dislikes), half)))
        take_d = min(len(dislikes), n - take_l)
        chosen = sorted(likes[:take_l] + dislikes[:take_d], key=lambda r: -r["id"])
        return [(r["sign"], r["title"]) for r in chosen]

    def item_exists(self, item_id: int) -> bool:
        return self.conn.execute("SELECT 1 FROM items WHERE id=?", (item_id,)).fetchone() is not None

    def set_rating(self, item_id: int, sign: int, reason: str | None = None) -> None:
        """+1 like, -1 dislike, 0 clear. Appends; the latest row per item wins. A snapshot of the item is kept with it."""
        self.add_feedback(item_id, sign, reason, self.snapshot(item_id) if sign else None)

    def ratings(self) -> dict[int, int]:
        rows = self.conn.execute(
            """SELECT item_id, sign FROM feedback WHERE id IN (SELECT MAX(id) FROM feedback GROUP BY item_id)
               AND sign != 0""").fetchall()
        return {r["item_id"]: r["sign"] for r in rows}

    def set_mark(self, item_id: int, saved: bool | None = None, read: bool | None = None) -> None:
        row = self.conn.execute("SELECT saved, read FROM marks WHERE item_id=?", (item_id,)).fetchone()
        sv = int(saved) if saved is not None else (row["saved"] if row else 0)
        rd = int(read) if read is not None else (row["read"] if row else 0)
        self.conn.execute(
            """INSERT INTO marks (item_id, saved, read, ts) VALUES (?,?,?,?)
               ON CONFLICT(item_id) DO UPDATE SET saved=excluded.saved, read=excluded.read, ts=excluded.ts""",
            (item_id, sv, rd, _now()))
        self.conn.commit()

    def marks(self) -> dict[str, list[int]]:
        rows = self.conn.execute("SELECT item_id, saved, read FROM marks").fetchall()
        return {"saved": [r["item_id"] for r in rows if r["saved"]], "read": [r["item_id"] for r in rows if r["read"]]}

    # ------------------------------------------------------------------ spend
    def spent(self, day: str, stream: str | None = None, exclude: tuple[str, ...] = ()) -> float:
        """Spend on `day`: all streams, one stream, or all except `exclude` (streams with their own separate budget)."""
        q, a = "SELECT COALESCE(SUM(cost_usd),0) FROM spend WHERE day=?", [day]
        if stream:
            q += " AND stream=?"; a.append(stream)
        if exclude:
            q += f" AND stream NOT IN ({','.join('?' * len(exclude))})"; a += list(exclude)
        return float(self.conn.execute(q, a).fetchone()[0])

    def spent_between(self, start_day: str, end_day: str, stream: str | None = None, exclude: tuple[str, ...] = ()) -> float:
        q, a = "SELECT COALESCE(SUM(cost_usd),0) FROM spend WHERE day>=? AND day<=?", [start_day, end_day]
        if stream:
            q += " AND stream=?"; a.append(stream)
        if exclude:
            q += f" AND stream NOT IN ({','.join('?' * len(exclude))})"; a += list(exclude)
        return float(self.conn.execute(q, a).fetchone()[0])

    def add_spend(self, day: str, stream: str, usage: dict, cost: float) -> None:
        self.conn.execute(
            """INSERT INTO spend (day, stream, calls, input_tokens, output_tokens, cache_write_tokens,
                                  cache_read_tokens, cost_usd)
               VALUES (?,?,1,?,?,?,?,?)
               ON CONFLICT(day, stream) DO UPDATE SET
                 calls=calls+1, input_tokens=input_tokens+excluded.input_tokens,
                 output_tokens=output_tokens+excluded.output_tokens,
                 cache_write_tokens=cache_write_tokens+excluded.cache_write_tokens,
                 cache_read_tokens=cache_read_tokens+excluded.cache_read_tokens,
                 cost_usd=cost_usd+excluded.cost_usd""",
            (day, stream, usage.get("input_tokens", 0), usage.get("output_tokens", 0),
             usage.get("cache_write_tokens", 0), usage.get("cache_read_tokens", 0), cost))
        self.conn.commit()

    def ack(self, key: str) -> None:
        self.conn.execute("INSERT INTO acks (key, ts) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET ts=excluded.ts", (key, _now()))
        self.conn.commit()

    def acks(self) -> list[str]:
        return [r["key"] for r in self.conn.execute("SELECT key FROM acks")]

    # ------------------------------------------------------------ tuning
    def add_report(self, n_ratings: int, last_feedback_id: int, notes: str, model: str) -> int:
        cur = self.conn.execute("INSERT INTO tuning_reports (ts, n_ratings, last_feedback_id, notes, model) VALUES (?,?,?,?,?)",
                                (_now(), n_ratings, last_feedback_id, notes, model))
        self.conn.commit()
        return cur.lastrowid

    def add_proposal(self, report_id: int, kind: str, payload: dict, why: str, evidence: list[int], backtest: dict) -> int:
        cur = self.conn.execute("INSERT INTO proposals (report_id, kind, payload, why, evidence, backtest) VALUES (?,?,?,?,?,?)",
                                (report_id, kind, _j(payload), why, _j(evidence), _j(backtest)))
        self.conn.commit()
        return cur.lastrowid

    def get_proposal(self, pid: int) -> dict | None:
        r = self.conn.execute("SELECT * FROM proposals WHERE id=?", (pid,)).fetchone()
        return self._proposal(r) if r else None

    @staticmethod
    def _proposal(r) -> dict:
        d = dict(r)
        for k in ("payload", "evidence", "backtest", "before"):
            d[k] = json.loads(d[k]) if d.get(k) else (None if k == "before" else {})
        return d

    def proposals(self, statuses: tuple[str, ...], limit: int = 100) -> list[dict]:
        q = f"SELECT * FROM proposals WHERE status IN ({','.join('?' * len(statuses))}) ORDER BY id DESC LIMIT ?"
        return [self._proposal(r) for r in self.conn.execute(q, list(statuses) + [limit])]

    def decide_proposal(self, pid: int, status: str, before: dict | None = None) -> None:
        self.conn.execute("UPDATE proposals SET status=?, decided=?, before=COALESCE(?, before) WHERE id=?",
                          (status, _now(), _j(before) if before is not None else None, pid))
        self.conn.commit()

    def latest_report(self) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM tuning_reports ORDER BY id DESC LIMIT 1").fetchone()

    def rated(self) -> list[dict]:
        """Latest non-cleared rating per item with its snapshot (what the item looked like when rated) and reason."""
        out = []
        for r in self.conn.execute("""SELECT f.id fid, f.item_id, f.sign, f.ts, f.reason, f.snapshot FROM feedback f
                                      WHERE f.id IN (SELECT MAX(id) FROM feedback GROUP BY item_id) AND f.sign != 0 ORDER BY f.id"""):
            snap = json.loads(r["snapshot"]) if r["snapshot"] else (self.snapshot(r["item_id"]) if self.item_exists(r["item_id"]) else None)
            if snap:
                out.append({"fid": r["fid"], "item_id": r["item_id"], "sign": r["sign"], "ts": r["ts"], "reason": r["reason"], **snap})
        return out

    # ---------------------------------------------------------- runs / state
    def start_run(self, stream: str) -> int:
        cur = self.conn.execute("INSERT INTO runs (stream, started) VALUES (?,?)", (stream, _now()))
        self.conn.commit()
        return cur.lastrowid

    def finish_run(self, run_id: int, ok: bool, error: str = "", counts: dict | None = None,
                   notes: list[str] | None = None) -> None:
        self.conn.execute("UPDATE runs SET finished=?, ok=?, error=?, counts=?, notes=? WHERE id=?",
                          (_now(), int(ok), error, _j(counts or {}), _j(notes or []), run_id))
        self.conn.commit()

    def get_state(self, key: str, default: str | None = None) -> str | None:
        r = self.conn.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return r["value"] if r else default

    def set_state(self, key: str, value: str) -> None:
        self.conn.execute("INSERT INTO state (key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                          (key, value))
        self.conn.commit()
