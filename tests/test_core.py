"""Schema migration, domain tagging, config migration, and the combined/per-stream/weekly budget guard."""
import json
import sqlite3

import pytest
import yaml

from litsearch.core import config
from litsearch.core.budget import Budget, Prices, iso_week_bounds
from litsearch.core.db import DB
from litsearch.core.domains import DomainTagger
from litsearch.core.models import Item, Paper, title_key

PRICES = Prices(1.0, 5.0)     # $1 / $5 per MTok: 1,000 in-tokens = $0.001


# ------------------------------------------------------------------ v1 database -> unified schema
V1_SCHEMA = """
CREATE TABLE papers (id INTEGER PRIMARY KEY AUTOINCREMENT, pmid TEXT, doi TEXT, published_doi TEXT, arxiv_id TEXT,
  title_key TEXT, title TEXT, journal TEXT, source TEXT, is_preprint INTEGER DEFAULT 0, status TEXT NOT NULL,
  relevance REAL, track TEXT, first_seen TEXT NOT NULL, sent_date TEXT);
CREATE TABLE feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, paper_id INTEGER NOT NULL, sign INTEGER NOT NULL, ts TEXT NOT NULL);
CREATE TABLE spend (day TEXT PRIMARY KEY, calls INTEGER NOT NULL DEFAULT 0, input_tokens INTEGER NOT NULL DEFAULT 0,
  output_tokens INTEGER NOT NULL DEFAULT 0, cache_write_tokens INTEGER NOT NULL DEFAULT 0,
  cache_read_tokens INTEGER NOT NULL DEFAULT 0, cost_usd REAL NOT NULL DEFAULT 0);
"""
LONG = "Large language models for extracting HIV outcomes from clinical notes"


def make_v1(path):
    c = sqlite3.connect(path)
    c.executescript(V1_SCHEMA)
    c.execute("INSERT INTO papers (id,pmid,doi,title_key,title,journal,source,status,relevance,track,first_seen) "
              "VALUES (7,'111','10.1/abc',?,?,'JAMIA','pubmed','sent',8,'mine','2026-11-03T10:00:00')", (title_key(LONG), LONG))
    c.execute("INSERT INTO papers (id,arxiv_id,title_key,title,journal,source,status,first_seen,published_doi) "
              "VALUES (9,'2411.5','anarxivpaperaboutagentsforscienceworkflows','An arXiv paper about agents for science workflows','arXiv','arxiv','screened','2026-11-03T10:00:00','10.9/pub')")
    c.execute("INSERT INTO feedback (paper_id,sign,ts) VALUES (7,1,'2026-11-03T11:00:00')")
    c.execute("INSERT INTO spend VALUES ('2026-11-03',4,35163,8142,995,2985,0.00774)")
    c.commit()
    c.close()


def test_v1_database_migrates_losslessly_and_keeps_dedup_working(tmp_path):
    path = tmp_path / "seen.db"
    make_v1(path)
    db = DB(path)
    assert (tmp_path / "seen.db.v1.bak").exists()
    rows = {r["id"]: r for r in db.conn.execute("SELECT * FROM items")}
    assert set(rows) == {7, 9} and rows[7]["stream"] == "literature" and rows[7]["source_id"] == "111"
    assert rows[7]["status"] == "sent" and rows[7]["relevance"] == 8 and rows[9]["source_id"] == "arxiv:2411.5"
    # dedup identical to v1: pmid, doi (either direction), arxiv, published_doi link, normalised title
    assert db.find(Paper("pubmed", "zzz", pmid="111"))
    assert db.find(Paper("medrxiv", "zzzzzzzzzzzzzzzzzzzzzzzzzz", doi="10.1/ABC"))
    assert db.find(Paper("arxiv", "q", arxiv_id="2411.5"))
    assert db.find(Paper("pubmed", "Brand new unrelated title of some length", pmid="5", doi="10.9/pub"))
    assert db.find(Paper("pubmed", LONG.upper() + "."))
    assert db.find(Paper("pubmed", "Something else entirely unrelated here", pmid="999")) is None
    # feedback + spend carried over
    assert db.recent_feedback(10) == [(1, LONG)]
    assert db.spent("2026-11-03") == pytest.approx(0.00774) and db.spent("2026-11-03", "literature") == pytest.approx(0.00774)
    assert db.was_sent_pmid("111") and not db.was_sent_pmid("9")
    db.close()
    DB(path).close()                                   # reopening a migrated DB is a no-op
    assert sqlite3.connect(path).execute("SELECT COUNT(*) FROM items").fetchone()[0] == 2


# ------------------------------------------------------------------ unified items, per-stream dedup, events
def test_dedup_is_per_stream_and_events_are_stored(db):
    a = Item("grants", "grants.gov", "RFA-DA-28-012", "A NOFO title", stream_specific={"close": "2027-01-26"})
    b = Item("trials", "ctgov", "RFA-DA-28-012", "Same id, other stream")
    iid = db.record(a, "shown")
    assert db.find(a)["id"] == iid and db.find(b) is None
    db.record(b, "shown")
    db.add_event(iid, "nofo_modified", {"revision": 7})
    db.update_item(iid, stream_specific={"close": "2027-02-02"}, takeaway="t")
    row = db.conn.execute("SELECT * FROM items WHERE id=?", (iid,)).fetchone()
    assert json.loads(row["stream_specific"]) == {"close": "2027-02-02"} and row["takeaway"] == "t"
    assert db.conn.execute("SELECT kind FROM item_events WHERE item_id=?", (iid,)).fetchone()[0] == "nofo_modified"
    assert row["last_updated"] >= row["first_seen"]


def test_run_log_and_state(db):
    rid = db.start_run("grants")
    db.finish_run(rid, False, "boom", {"retrieved": 3}, ["note"])
    r = db.conn.execute("SELECT * FROM runs WHERE id=?", (rid,)).fetchone()
    assert r["ok"] == 0 and r["error"] == "boom" and json.loads(r["notes"]) == ["note"]
    assert db.get_state("rotation") is None
    db.set_state("rotation", "1"); db.set_state("rotation", "2")
    assert db.get_state("rotation") == "2"


def test_dry_run_copy_includes_wal_contents(tmp_path):
    src = DB(tmp_path / "a.db")
    src.record(Item("grants", "g", "X1", "t"), "shown")          # sits in the WAL, not yet checkpointed
    DB.copy_file(tmp_path / "a.db", tmp_path / "b.db")
    assert DB(tmp_path / "b.db").conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 1


# ------------------------------------------------------------------ domain tagging
def test_domain_tagging_rules(cfg2):
    tg = DomainTagger(cfg2["domains"])
    assert tg.ids == ["llm_ai", "infectious_diseases", "hiv_addiction"]
    assert tg.tag("Large language models for chart review") == ["llm_ai"]                       # title hit
    assert tg.tag("A cohort study", "We used machine learning and deep learning models.") == ["llm_ai"]   # 2 text hits
    assert tg.tag("A cohort study", "We used machine learning once.") == []                       # 1 abstract hit: no
    assert tg.tag("x", mesh=["Sepsis"]) == ["infectious_diseases"]                                # MeSH hit
    assert tg.tag("LLM triage of PrEP candidates") == ["llm_ai", "hiv_addiction"]                 # several, config order
    assert tg.tag("A tool that aids diagnosis") == []                                             # AIDS is case-sensitive
    assert tg.valid(["hiv_addiction", "bogus", "llm_ai"]) == ["llm_ai", "hiv_addiction"]


# ------------------------------------------------------------------ config v2
def test_v1_to_v2_is_lossless_for_literature(cfg2, tmp_path):
    v1 = yaml.safe_load((config.ROOT / "tests" / "fixtures" / "interests.v1.yaml").read_text())
    view = config.to_literature_cfg(cfg2)
    assert {g["id"]: g for g in v1["groups"]} == {g["id"]: {k: v for k, v in g.items() if k != "domains"} for g in view["groups"]}
    for k in ("profile", "major", "medrxiv", "arxiv", "exclude", "limits", "digest"):
        assert v1[k] == view[k], k
    groups = {g["id"]: g["domains"] for g in view["groups"]}
    assert groups["hiv_care_continuum"] == ["hiv_addiction"]
    assert groups["ai_prediction_id_addiction"] == ["llm_ai", "hiv_addiction", "infectious_diseases"]


def test_migrate_file_backs_up_and_is_idempotent(tmp_path):
    src = (config.ROOT / "tests" / "fixtures" / "interests.v1.yaml").read_text()
    f = tmp_path / "interests.yaml"
    f.write_text(src)
    diff = config.migrate_file(f)
    assert "+version: 2" in diff and (tmp_path / "interests.v1.yaml.bak").read_text() == src
    assert yaml.safe_load(f.read_text())["version"] == 2
    assert config.migrate_file(f) == ""                                  # already v2
    assert config.load_interests(f)["streams"]["literature"]["max_candidates"] == 60


def test_v1_file_still_loads_in_memory(tmp_path):
    f = tmp_path / "interests.yaml"
    f.write_text((config.ROOT / "tests" / "fixtures" / "interests.v1.yaml").read_text())
    assert config.load_interests(f)["version"] == 2 and f.read_text().startswith("# Research-interest")


def test_settings_model_effort_and_budget_defaults(tmp_path):
    s = config.Settings(env={"LLM_MODEL": "m1", "LLM_MODEL_JOURNAL_CLUB": "m2", "LLM_EFFORT": "low",
                             "LLM_EFFORT_JOURNAL_CLUB": "medium"}, root=tmp_path)
    assert s.model_for({"model_env": "LLM_MODEL_JOURNAL_CLUB"}) == "m2" and s.model_for({}) == "m1"
    assert s.effort_for("journal_club") == "medium" and s.effort_for("grants") == "low"
    assert s.daily_budget == 0.15 and s.weekly_extra == 0.10
    assert config.Settings(env={"LLM_MODEL_JOURNAL_CLUB": ""}, root=tmp_path).model_for(
        {"model_env": "LLM_MODEL_JOURNAL_CLUB"}) == "claude-haiku-5-5"


# ------------------------------------------------------------------ combined budget guard
def test_per_stream_prices_override_defaults():
    env = {"PRICE_INPUT_PER_MTOK": "0.1", "PRICE_OUTPUT_PER_MTOK": "0.5",
           "PRICE_JOURNAL_CLUB_INPUT_PER_MTOK": "2", "PRICE_JOURNAL_CLUB_OUTPUT_PER_MTOK": "10"}
    assert Prices.from_env(env, "grants").inp == 0.1 and Prices.from_env(env, "journal_club").out == 10.0


def test_daily_cap_is_combined_across_streams(db):
    lit = Budget(db, "2026-11-03", 0.15, PRICES, "literature")
    trials = Budget(db, "2026-11-03", 0.15, PRICES, "trials")
    lit.record({"input_tokens": 20_000, "output_tokens": 20_000})       # $0.02 + $0.10 = $0.12
    assert trials.spent_today == pytest.approx(0.12) and trials.spent_stream_today == 0
    assert trials.allow(0.03) and not trials.allow(0.0301) and trials.tripped and "daily cap" in trials.trip_reason
    assert db.spent("2026-11-03", "literature") == pytest.approx(0.12) and db.spent("2026-11-03", "trials") == 0


def test_per_stream_cap_blocks_before_the_combined_cap(db):
    g = Budget(db, "2026-11-03", 0.15, PRICES, "grants", stream_cap=0.03)
    g.record({"input_tokens": 20_000})                                   # $0.02
    assert g.allow(0.01) and not g.allow(0.0101) and "grants cap" in g.trip_reason
    assert Budget(db, "2026-11-03", 0.15, PRICES, "trials", stream_cap=0.03).allow(0.03)    # other streams unaffected


def test_weekly_allowance_ignores_daily_cap_and_resets_each_iso_week(db):
    assert iso_week_bounds("2026-11-04") == ("2026-11-02", "2026-11-08")           # Wed -> Mon..Sun
    lit = Budget(db, "2026-11-02", 0.15, PRICES, "literature")
    lit.record({"output_tokens": 30_000})                                           # $0.15: daily cap is gone
    jc = Budget(db, "2026-11-02", 0.15, PRICES, "journal_club", weekly_limit=0.10)
    assert jc.allow(0.05)                                                           # daily cap would refuse this
    jc.record({"output_tokens": 14_000})                                            # $0.07
    assert jc.spent_week == pytest.approx(0.07) and jc.allow(0.03) and not jc.allow(0.0301)
    assert "weekly" in jc.trip_reason
    nxt = Budget(db, "2026-11-09", 0.15, PRICES, "journal_club", weekly_limit=0.10)  # next Monday: fresh allowance
    assert nxt.spent_week == 0 and nxt.allow(0.10)


# ------------------------------------------------------------------ page data
def _site_items(tmp_path, db, cfg2, now=None):
    from datetime import datetime, timezone
    from litsearch.core import pagedata
    s = config.Settings(env={}, root=tmp_path)
    out = tmp_path / "site"
    pagedata.regenerate(s, cfg2, db, out, now or datetime(2026, 11, 3, 15, 0, tzinfo=timezone.utc))
    return out, json.loads((out / "data" / "items.json").read_text()), json.loads((out / "data" / "items-90d.json").read_text()), \
        json.loads((out / "data" / "status.json").read_text())


def test_pagedata_files_window_and_abstract_shards(tmp_path, db, cfg2):
    from datetime import datetime, timedelta, timezone
    now = datetime(2026, 11, 3, 15, 0, tzinfo=timezone.utc)
    a = Paper("pubmed", "Recent paper about large language models in care", "ABSTRACT TEXT " * 5, pmid="1", journal="JAMIA",
              authors=["A B", "C D", "E F", "G H"], domains=["llm_ai"], relevance=8, takeaway="tk")
    old = Paper("pubmed", "Old paper about large language models in care", "old abstract", pmid="2", relevance=9)
    scr = Paper("pubmed", "Screened paper about large language models in care", "x", pmid="3", relevance=1)
    mid = Paper("pubmed", "Mid-age paper about large language models in care", "mid abstract", pmid="4", relevance=7)
    for p, st in ((a, "sent"), (old, "sent"), (scr, "screened"), (mid, "sent")):
        db.record(p, st)
    db.conn.execute("UPDATE items SET first_seen=? WHERE source_id='2'", ((now - timedelta(days=120)).isoformat(),))
    db.conn.execute("UPDATE items SET first_seen=? WHERE source_id='4'", ((now - timedelta(days=30)).isoformat(),))
    db.conn.commit()
    out, recent, full, status = _site_items(tmp_path, db, cfg2, now)
    assert [i["source_id"] for i in recent["items"]] == ["1"]                       # 14-day default view
    assert sorted(i["source_id"] for i in full["items"]) == ["1", "4"]              # 90-day window; screened + 120-day-old excluded
    it = recent["items"][0]
    assert it["ss"]["authors"] == "A B, C D, E F et al." and it["ss"]["journal"] == "JAMIA" and "summary" not in it
    assert "abstract" not in json.dumps(it).lower()                                  # abstracts live in the shards
    ab = json.loads((out / "data" / "abstracts" / "2026-11.json").read_text())
    assert ab[str(it["id"])].startswith("ABSTRACT TEXT") and it["shard"] == "2026-11"
    assert (out / "data" / "abstracts" / "2026-10.json").exists()                    # the 30-day-old item's month
    assert not (out / "data" / "abstracts" / "2026-07.json").exists()                # nothing from outside the window


def test_pagedata_excludes_v1_migrated_rows_and_keeps_open_grants_past_the_window(tmp_path, cfg2):
    from datetime import datetime, timedelta, timezone
    path = tmp_path / "seen.db"
    make_v1(path)
    db = DB(path)
    now = datetime(2026, 11, 3, 15, 0, tzinfo=timezone.utc)
    g = Item("grants", "grants.gov", "RFA-X-1", "Old but still open NOFO", stream_specific={"kind": "nofo", "close_date": "2027-02-01"})
    g2 = Item("grants", "grants.gov", "RFA-X-2", "Old closed NOFO", stream_specific={"kind": "nofo", "close_date": "2026-09-01"})
    for x in (g, g2):
        db.record(x, "shown")
    db.conn.execute("UPDATE items SET first_seen=? WHERE stream='grants'", ((now - timedelta(days=200)).isoformat(),))
    db.conn.commit()
    _, recent, full, _ = _site_items(tmp_path, db, cfg2, now)
    titles = [i["title"] for i in full["items"]]
    assert titles == ["Old but still open NOFO"]                                     # v1 rows hidden; closed old grant dropped
    assert [i["title"] for i in recent["items"]] == titles                           # open grants are always in the default view


def test_pagedata_events_status_and_spend(tmp_path, db, cfg2):
    from datetime import datetime, timezone
    now = datetime(2026, 11, 3, 15, 0, tzinfo=timezone.utc)
    iid = db.record(Item("trials", "ctgov", "NCT1", "A trial", stream_specific={"kind": "trial", "status": "RECRUITING"}), "shown")
    db.add_event(iid, "status_change", {"from": "NOT_YET_RECRUITING", "to": "RECRUITING"})
    ok = db.start_run("literature"); db.finish_run(ok, True, "", {"retrieved": 5}, [])
    bad = db.start_run("trials"); db.finish_run(bad, False, "RuntimeError: down", {}, ["trials: note"])
    db.add_spend("2026-11-03", "literature", {"input_tokens": 1}, 0.0123)
    db.add_spend("2026-11-03", "journal_club", {"input_tokens": 1}, 0.03)
    _, recent, _, st = _site_items(tmp_path, db, cfg2, now)
    assert recent["items"][0]["events"][0]["kind"] == "status_change" and recent["items"][0]["events"][0]["detail"]["to"] == "RECRUITING"
    by = {s["id"]: s for s in st["streams"]}
    assert by["literature"]["last_run_ok"] is True and by["literature"]["stale"] is False and by["literature"]["counts"] == {"retrieved": 5}
    assert by["trials"]["last_run_ok"] is False and by["trials"]["error"].startswith("RuntimeError") and by["trials"]["notes"] == ["trials: note"]
    assert by["grants"]["never_run"] is True and by["grants"]["stale"] is True
    sp = st["spend"]
    assert sp["today"] == pytest.approx(0.0423) and sp["daily_cap"] == 0.15 and sp["week_journal_club"] == pytest.approx(0.03)
    assert sp["weekly_extra"] == 0.10 and [d["id"] for d in st["domains"]] == ["llm_ai", "infectious_diseases", "hiv_addiction"]
    assert by["literature"]["spend_today"] == pytest.approx(0.0123) and st["deadline_flag_days"] == 60


def test_pagedata_is_written_atomically_and_refreshes_the_app(tmp_path, db, cfg2):
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "index.html").write_text("<html>v1</html>")
    out, *_ = _site_items(tmp_path, db, cfg2)
    assert (out / "index.html").read_text() == "<html>v1</html>"
    (tmp_path / "web" / "index.html").write_text("<html>v2</html>")
    _site_items(tmp_path, db, cfg2)
    assert (out / "index.html").read_text() == "<html>v2</html>" and not list(out.rglob("*.tmp"))


def test_stream_run_regenerates_page_data_even_when_it_fails(tmp_path, monkeypatch, cfg2):
    from litsearch.core import pipeline
    from litsearch.core.context import RunOptions
    from litsearch.streams import literature
    from tests.helpers import settings_for
    s = settings_for(tmp_path)
    def boom(*a, **k): raise RuntimeError("all sources failed")
    monkeypatch.setattr(literature, "retrieve", boom)
    with pytest.raises(RuntimeError):
        pipeline.run(s, cfg2, RunOptions())
    st = json.loads((s.site_dir / "data" / "status.json").read_text())
    assert {x["id"]: x for x in st["streams"]}["literature"]["last_run_ok"] is False        # the banner can show the failure
    pipeline.run(s, cfg2, RunOptions(dry_run=True), stream="journal_club")                   # dry-run -> preview dir, not site/
    assert (tmp_path / "out" / "preview" / "data" / "status.json").exists()


# ------------------------------------------------------------------ systemd generation
def test_schedule_to_oncalendar():
    from litsearch.core.systemd import on_calendar
    assert on_calendar("05:30") == "*-*-* 05:30:00 America/Chicago"
    assert on_calendar("sun 06:15") == "Sun *-*-* 06:15:00 America/Chicago"
    assert on_calendar("Monday 5:15") == "Mon *-*-* 05:15:00 America/Chicago"
    for bad in ("", "25:00", "noon", "xyz 05:00", "05:61"):
        with pytest.raises(ValueError):
            on_calendar(bad)


def test_units_one_per_enabled_stream_with_literature_keeping_its_names(cfg2, tmp_path):
    from litsearch.core.systemd import render_units
    cfg = json.loads(json.dumps(cfg2))
    cfg["streams"]["grants"]["enabled"] = False
    files = render_units(cfg, tmp_path, "pi")
    assert sorted(files) == ["litsearch-briefing.service", "litsearch-briefing.timer", "litsearch-journal-club.service",
                             "litsearch-journal-club.timer", "litsearch-people.service",
                             "litsearch-people.timer", "litsearch-reporter.service", "litsearch-reporter.timer",
                             "litsearch-trials.service", "litsearch-trials.timer", "litsearch-tuning.service", "litsearch-tuning.timer",
                             "litsearch-web.service", "litsearch.service", "litsearch.timer"]
    web = files["litsearch-web.service"]
    assert "-m litsearch.web --bind 127.0.0.1 --port 8081" in web and f"ReadWritePaths={tmp_path}" in web     # loopback only, writable project dir
    assert "OnCalendar=*-*-* 04:30:00 America/Chicago" in files["litsearch-people.timer"]
    assert "OnCalendar=*-*-* 04:45:00 America/Chicago" in files["litsearch-briefing.timer"]
    assert not any("podcast" in f for f in files)                      # on-demand: started from the page, never by a timer
    assert "--stream journal_club" in files["litsearch-journal-club.service"] and "User=pi" in files["litsearch.service"]
    assert "OnCalendar=Mon *-*-* 04:35:00 America/Chicago" in files["litsearch-journal-club.timer"]
    assert "OnCalendar=Sun *-*-* 04:15:00 America/Chicago" in files["litsearch-reporter.timer"]
    assert "OnCalendar=*-*-* 04:00:00 America/Chicago" in files["litsearch.timer"]
    assert all("Persistent=true" in t for n, t in files.items() if n.endswith(".timer"))
    assert f"ExecStart={tmp_path}/.venv/bin/python -m litsearch --stream trials" in files["litsearch-trials.service"]


# ------------------------------------------------------------------ ratings, saved/read marks, per-stream feedback
def test_ratings_clear_and_per_stream_feedback(db):
    a = db.record(Paper("pubmed", "A literature paper about large language models", "x", pmid="1"), "sent")
    g = db.record(Item("grants", "grants.gov", "RFA-1", "A grant title"), "shown")
    db.set_rating(a, 1); db.set_rating(g, -1)
    assert db.ratings() == {a: 1, g: -1}
    assert db.recent_feedback(10) == [(-1, "A grant title"), (1, "A literature paper about large language models")]
    assert db.recent_feedback(10, "grants") == [(-1, "A grant title")]                  # a stream's prompt only sees its own ratings
    assert db.recent_feedback(10, "literature") == [(1, "A literature paper about large language models")]
    db.set_rating(a, 0)                                                                 # cleared
    assert db.ratings() == {g: -1} and db.recent_feedback(10, "literature") == []
    db.set_rating(a, -1)
    assert db.ratings()[a] == -1                                                        # latest wins
    assert db.item_exists(a) and not db.item_exists(9999)


def test_marks_save_and_read_independently_and_persist(db):
    a = db.record(Item("grants", "g", "X1", "t1"), "shown")
    b = db.record(Item("grants", "g", "X2", "t2"), "shown")
    db.set_mark(a, saved=True)
    db.set_mark(a, read=True)                                                           # leaves saved alone
    db.set_mark(b, read=True)
    m = db.marks()
    assert m["saved"] == [a] and sorted(m["read"]) == [a, b]
    db.set_mark(a, saved=False)
    assert db.marks()["saved"] == []


def test_find_can_target_another_stream(db):
    db.record(Paper("pubmed", "Large language models in clinical care study", "x", pmid="77", doi="10.1/zz"), "sent")
    probe = Item("people", "openalex", "W1", "x")
    assert db.find(probe) is None
    p2 = Paper("pubmed", "Large language models in clinical care study", "", pmid="77")
    assert db.find(p2, stream="literature") is not None


def test_saved_items_stay_in_page_data_past_the_window(tmp_path, db, cfg2):
    from datetime import datetime, timedelta, timezone
    now = datetime(2026, 11, 3, 15, 0, tzinfo=timezone.utc)
    keep = db.record(Paper("pubmed", "An old paper that I saved about large language models", "abs", pmid="1"), "sent")
    drop = db.record(Paper("pubmed", "An old paper that I did not save about LLMs in care", "abs", pmid="2"), "sent")
    db.conn.execute("UPDATE items SET first_seen=?", ((now - timedelta(days=300)).isoformat(),)); db.conn.commit()
    db.set_mark(keep, saved=True)
    _, recent, full, _ = _site_items(tmp_path, db, cfg2, now)
    assert [i["id"] for i in full["items"]] == [keep] and recent["items"] == []                # in the 90-day file, not in the recent file


def test_snapshot_captures_what_was_rated_including_the_llm_score(db):
    p = Paper("pubmed", "A paper about large language models in clinical care", "abs", pmid="5", journal="JAMIA",
              authors=["Smith J", "Lee K"], relevance=9, takeaway="tk", domains=["llm_ai"], kw_score=12)
    pid = db.record(p, "sent")
    db.set_rating(pid, -1, reason="not_topic")
    row = db.conn.execute("SELECT * FROM feedback").fetchone()
    snap = json.loads(row["snapshot"])
    assert row["reason"] == "not_topic" and snap["relevance"] == 9 and snap["journal"] == "JAMIA" and snap["authors"] == ["Smith J", "Lee K"]
    assert snap["domains"] == ["llm_ai"] and snap["stream"] == "literature" and snap["kw_score"] == 12
    db.set_rating(pid, 0)
    assert db.conn.execute("SELECT snapshot FROM feedback ORDER BY id DESC LIMIT 1").fetchone()[0] is None      # nothing to snapshot on a clear


def test_existing_feedback_table_gains_reason_and_snapshot_columns(tmp_path):
    import sqlite3
    path = tmp_path / "old.db"
    c = sqlite3.connect(path)
    c.executescript(V1_SCHEMA.split("CREATE TABLE feedback")[0] + "CREATE TABLE items (id INTEGER PRIMARY KEY, stream TEXT, source TEXT, source_id TEXT, title TEXT, url TEXT, date TEXT, summary TEXT, domains TEXT, relevance REAL, takeaway TEXT, fit_note TEXT, stream_specific TEXT, status TEXT, track TEXT, kw_score REAL, first_seen TEXT, last_updated TEXT);"
                    "CREATE TABLE feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, item_id INTEGER NOT NULL, sign INTEGER NOT NULL, ts TEXT NOT NULL);"
                    "INSERT INTO feedback (item_id, sign, ts) VALUES (1, 1, '2026-11-01T00:00:00+00:00');")
    c.commit(); c.close()
    d = DB(path)
    cols = {r[1] for r in d.conn.execute("PRAGMA table_info(feedback)")}
    assert {"reason", "snapshot"} <= cols and d.conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 1        # old rows survive


def test_recent_feedback_is_balanced_between_likes_and_dislikes(db):
    ids = [db.record(Item("grants", "g", f"X{i}", f"title {i}"), "shown") for i in range(14)]
    for i in ids[:10]:
        db.set_rating(i, 1)                     # ten likes in a row...
    for i in ids[10:13]:
        db.set_rating(i, -1)                    # ...then three dislikes
    fb = db.recent_feedback(10)
    assert sum(1 for s, _ in fb if s > 0) == 7 and sum(1 for s, _ in fb if s < 0) == 3 and len(fb) == 10      # dislikes fill their share, likes the rest
    assert len(db.recent_feedback(4)) == 4 and sum(1 for s, _ in db.recent_feedback(4) if s < 0) == 2          # 2 + 2 when both sides have enough
    only_likes = DB(":memory:"); j = only_likes.record(Item("grants", "g", "Y", "t"), "shown"); only_likes.set_rating(j, 1)
    assert only_likes.recent_feedback(10) == [(1, "t")]
