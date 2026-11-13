import json
import re
from datetime import datetime

import pytest

from litsearch.core import pagedata, pipeline
from litsearch.core.context import RunOptions
from litsearch.core.db import DB
from litsearch.core.models import Item, Paper
from litsearch.streams import briefing
from litsearch.core.triage import ParseError
from tests.helpers import settings_for

BR_ENV = {"LLM_MODEL": "claude-haiku-5-5", "LLM_MODEL_BRIEFING": "claude-sonnet-5-5", "LLM_EFFORT_BRIEFING": "low",
          "PRICE_BRIEFING_INPUT_PER_MTOK": "2", "PRICE_BRIEFING_OUTPUT_PER_MTOK": "10"}


# ------------------------------------------------------------------ parsing
def test_parse_keeps_known_citations_strips_unknown_and_orders_refs():
    txt = json.dumps({"headline": "H", "paragraphs": [
        "A cohort study reports PrEP uptake rose after the intervention [i5] and a trial agrees [i3] [i999].",
        "Another substantial paragraph about funding and a second cite of the first item [i5], plus one more [i7]."]})
    out = briefing.parse_briefing(txt, {3, 5, 7})
    assert out["refs"] == [5, 3, 7] and "[i999]" not in " ".join(out["paragraphs"]) and "[i5]" in out["paragraphs"][0]
    assert out["headline"] == "H"


@pytest.mark.parametrize("bad", ["", "nope", '{"paragraphs": "x"}', '{"paragraphs": ["short"]}', '{"paragraphs": []}',
                                 json.dumps({"sections": [{"heading": "h", "paragraphs": ["x" * 60]}] * 9}),
                                 json.dumps({"sections": [{"heading": "h", "paragraphs": []}]}),
                                 json.dumps({"sections": "nope"})])
def test_parse_rejects_malformed(bad):
    with pytest.raises(ParseError):
        briefing.parse_briefing(bad, {1})


def test_parse_tolerates_fences_and_prose_around_json():
    body = json.dumps({"headline": "h", "paragraphs": ["p" * 50, "q" * 50]})
    assert len(briefing.parse_briefing("```json\n" + body + "\n```", set())["paragraphs"]) == 2
    assert len(briefing.parse_briefing("Here you go: " + body + " thanks", set())["paragraphs"]) == 2


# ------------------------------------------------------------------ the stream
class BriefClient:
    def __init__(self, bad=0):
        self.calls, self.messages, self.bad, self.users, self.systems = 0, self, bad, [], []

    def create(self, **kw):
        from tests.test_budget import _R
        self.calls += 1
        user = kw["messages"][0]["content"]
        self.users.append(user); self.systems.append(kw["system"][0]["text"])
        if self.calls <= self.bad:
            return _R("not json", i=3000, o=100)
        ids = re.findall(r"^\[i(\d+)\]", user, flags=re.M)
        body = {"headline": "A busy day for LLM evaluation",
                "paragraphs": [f"Several papers report on clinical NLP evaluation [i{ids[0]}] and a trial is testing a chatbot [i{ids[1]}], though one claim [i99999] was unfounded.",
                               f"Funding news is thin, but a followed author published again [i{ids[2]}], which is worth a look first."]}
        return _R(json.dumps(body), i=4000, o=700)


def seed(db):
    mk = lambda title, **kw: Paper("pubmed", title, "abstract " * 30, pmid=str(abs(hash(title)) % 10**7), **kw)
    ids = {}
    ids["lit_hi"] = db.record(mk("Large language models extract outcomes from clinical notes", relevance=9, takeaway="LLM extraction works", domains=["llm_ai"], journal="JAMIA"), "sent")
    ids["lit_lo"] = db.record(mk("A marginal paper about hospital parking in large language models era", relevance=2, takeaway="meh"), "sent")
    ids["lit_screened"] = db.record(mk("Screened out paper about LLM chatbots in medicine", relevance=9), "screened")
    ids["trial"] = db.record(Item("trials", "ctgov", "NCT1", "Chatbot coaching for OUD retention", relevance=8, takeaway="tests an AI coach", domains=["llm_ai", "hiv_addiction"],
                                  stream_specific={"kind": "trial", "status": "RECRUITING", "ai_intervention": True, "sponsor": "UAB"}), "shown")
    ids["person"] = db.record(Item("people", "openalex", "W1", "Endocarditis outcomes in people who inject drugs", relevance=3, takeaway="a cohort", domains=["hiv_addiction"],
                                   stream_specific={"kind": "paper", "watched": ["Rivera, Elena"], "journal": "CID", "is_preprint": False}), "shown")
    ids["grant"] = db.record(Item("grants", "grants.gov", "RFA-DA-1", "OUD technology R01", relevance=7, takeaway="funds tools", domains=["hiv_addiction"],
                                  stream_specific={"kind": "nofo", "number": "RFA-DA-1", "institutes": ["NIDA"], "mechanism": "R01", "close_date": "2027-01-26"}), "shown")
    ids["alt"] = db.record(Item("journal_club", "jc", "W:alt1", "An alternate pick title that is long", relevance=7, stream_specific={"kind": "journal_club_pick", "role": "alternate"}), "shown")
    ids["v1"] = db.record(Item("literature", "pubmed", "9", "Old migrated row about LLM", relevance=9, stream_specific={"migrated_from_v1": True}), "sent")
    db.add_event(ids["grant"], "nofo_modified", {"comments": "Extended the due date"})
    db.add_event(ids["trial"], "status_change", {"from": "NOT_YET_RECRUITING", "to": "RECRUITING"})
    return ids


@pytest.fixture
def world(tmp_path, monkeypatch, cfg2):
    s = settings_for(tmp_path, **BR_ENV)
    db = DB(s.db_path); ids = seed(db); db.set_rating(ids["lit_hi"], 1); db.set_rating(ids["lit_lo"], -1); db.close()
    client = BriefClient()
    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", lambda **k: client)
    cfg = json.loads(json.dumps(cfg2))
    cfg["streams"]["briefing"]["weekly_day"] = "never"                 # weekly only when a test asks for it
    return s, cfg, client, ids


def run(world, **kw):
    s, cfg, client, ids = world
    return pipeline.run(s, cfg, RunOptions(**kw), stream="briefing")


def items(world):
    db = DB(world[0].db_path)
    out = [dict(r) for r in db.items("briefing")]
    db.close()
    return out


def test_daily_briefing_is_written_with_valid_citations_and_the_right_prompt(world):
    assert run(world) == 0
    (b,) = items(world)
    ss = json.loads(b["stream_specific"])
    assert b["source_id"].startswith("daily:") and ss["period"] == "daily" and ss["model"] == "claude-sonnet-5-5"
    ids = world[3]
    assert set(ss["refs"]) <= {ids["lit_hi"], ids["trial"], ids["person"], ids["grant"]} and len(ss["refs"]) == 3
    assert "[i99999]" not in " ".join(ss["paragraphs"]) and ss["headline"].startswith("A busy day")
    user = world[2].users[0]
    assert f"[i{ids['lit_hi']}]" in user and f"[i{ids['trial']}]" in user and "AI intervention" in user
    assert f"[i{ids['lit_lo']}]" not in user                                    # below the relevance floor
    assert f"[i{ids['lit_screened']}]" not in user and f"[i{ids['alt']}]" not in user and f"[i{ids['v1']}]" not in user
    assert f"[i{ids['person']}]" in user and "Rivera, Elena" in user              # people you follow always make it in
    assert "VOLUME this period vs previous period" in user and "EVENT" in user and "Extended the due date" in user
    assert "RFA-DA-1" in user and "NIDA" in user


def test_prompt_includes_your_ratings_and_the_citation_rules(world):
    run(world)
    system = world[2].systems[0]
    assert "Liked:\n- Large language models extract outcomes" in system and "Disliked:\n- A marginal paper" in system
    assert "Use ONLY the supplied items" in system and "short titled sections" in system and "**double asterisks**" in system


def test_one_briefing_per_day_and_cost_logged_under_briefing(world):
    run(world); run(world)
    assert len(items(world)) == 1 and world[2].calls == 1
    db = DB(world[0].db_path)
    r = db.conn.execute("SELECT * FROM spend WHERE stream='briefing'").fetchone()
    assert r["calls"] == 1 and r["cost_usd"] == pytest.approx((4000 * 2 + 700 * 10) / 1e6)       # Sonnet prices from the briefing's own env


def test_monday_style_run_adds_a_weekly_piece_with_a_week_key_and_wider_window(world):
    from datetime import datetime as dt
    day = briefing.WEEKDAYS[dt.now(world[0].tz).weekday()]
    world[1]["streams"]["briefing"]["weekly_day"] = day
    run(world)
    keys = sorted(b["source_id"] for b in items(world))
    assert keys[0].startswith("daily:") and re.fullmatch(r"weekly:\d{4}-W\d{2}", keys[1]) and world[2].calls == 2
    assert "168 hours since the previous weekly briefing" in world[2].users[1]
    assert "weekly research briefing" in world[2].systems[1] and "'Trends'" in world[2].systems[1] and "'What to watch'" in world[2].systems[1]
    assert "daily research briefing" in world[2].systems[0] and "'Worth a look first'" in world[2].systems[0] and "'At a glance'" in world[2].systems[0]


def test_quiet_day_makes_no_call_and_says_so(world):
    world[1]["streams"]["briefing"]["min_items"] = 99
    run(world)
    assert items(world) == [] and world[2].calls == 0
    db = DB(world[0].db_path)
    notes = json.loads(db.conn.execute("SELECT notes FROM runs").fetchone()["notes"])
    assert any("quiet daily" in n for n in notes)


def test_budget_guard_blocks_the_call_without_a_briefing(world):
    world[0].env["DAILY_BUDGET_USD"] = "0.0000001"
    run(world)
    assert items(world) == [] and world[2].calls == 0
    db = DB(world[0].db_path)
    assert any("budget guard" in n for n in json.loads(db.conn.execute("SELECT notes FROM runs").fetchone()["notes"]))


def test_bad_json_retried_once_then_skipped_good_on_retry(world, monkeypatch):
    import anthropic
    c = BriefClient(bad=1); monkeypatch.setattr(anthropic, "Anthropic", lambda **k: c)
    run(world)
    assert c.calls == 2 and len(items(world)) == 1
    s = world[0]
    db = DB(s.db_path); db.conn.execute("DELETE FROM items WHERE stream='briefing'"); db.conn.execute("DELETE FROM item_keys WHERE stream='briefing'"); db.conn.commit(); db.close()
    c2 = BriefClient(bad=5); monkeypatch.setattr(anthropic, "Anthropic", lambda **k: c2)
    run(world)
    assert c2.calls == 2 and items(world) == []


def test_per_stream_cap_stops_one_stream_crowding_out_the_rest(world):
    db = DB(world[0].db_path)
    for i in range(30):
        db.record(Paper("pubmed", f"LLM evaluation paper number {i} in clinical medicine", "x" * 50, pmid=str(500 + i), relevance=9, domains=["llm_ai"]), "sent")
    db.close()
    world[1]["streams"]["briefing"]["max_items"] = 30
    run(world)
    given = re.findall(r"^\[i\d+\] (\w[\w ]*?) \|", world[2].users[0], flags=re.M)
    from collections import Counter
    assert Counter(given)["literature"] <= 10 and "trial" in given and "grant" in given        # cap = max_items // 3


def test_dry_run_stub_needs_no_api_and_leaves_the_real_db_alone(world, capsys):
    run(world, dry_run=True)
    out = capsys.readouterr().out
    assert "Dry run [briefing]" in out and "dry-run stub" in out and world[2].calls == 0 and items(world) == []


# ------------------------------------------------------------------ page data + API
def test_briefing_json_has_latest_daily_and_weekly_and_items_json_excludes_them(world):
    from datetime import datetime as dt
    world[1]["streams"]["briefing"]["weekly_day"] = briefing.WEEKDAYS[dt.now(world[0].tz).weekday()]
    run(world)
    site = world[0].site_dir / "data"
    b = json.loads((site / "briefing.json").read_text())
    assert b["daily"]["key"].startswith("daily:") and b["weekly"]["key"].startswith("weekly:")
    assert len(b["daily"]["paragraphs"]) == 2 and b["daily"]["refs"] and b["daily"]["headline"] and b["daily"]["model"] == "claude-sonnet-5-5"
    its = json.loads((site / "items-90d.json").read_text())["items"]
    assert not [i for i in its if i["stream"] == "briefing"]                    # never shown as a card in the lists
    st = json.loads((site / "status.json").read_text())
    assert any(x["id"] == "briefing" and x["last_run_ok"] for x in st["streams"])


def test_briefing_json_is_empty_until_one_exists(tmp_path, db, cfg2):
    assert pagedata.build_briefing(db) == {"daily": None, "weekly": None}


def test_window_starts_at_the_previous_briefing_so_consecutive_days_never_repeat(world):
    from datetime import datetime, timedelta, timezone
    db = DB(world[0].db_path)
    now = datetime.now(timezone.utc)
    early = db.record(Paper("pubmed", "Early paper about large language models in clinical care", "x" * 60, pmid="9001", relevance=9, domains=["llm_ai"]), "sent")
    late = db.record(Paper("pubmed", "Late paper about LLM agents for chart review in hospitals", "x" * 60, pmid="9002", relevance=9, domains=["llm_ai"]), "sent")
    db.conn.execute("UPDATE items SET first_seen=? WHERE stream='literature'", ((now - timedelta(hours=60)).isoformat(),))     # everything old...
    db.conn.execute("UPDATE items SET first_seen=? WHERE id=?", ((now - timedelta(hours=25)).isoformat(), early))             # ...except two
    db.conn.execute("UPDATE items SET first_seen=? WHERE id=?", ((now - timedelta(hours=23)).isoformat(), late))
    db.conn.execute("UPDATE items SET first_seen=? WHERE stream IN ('trials','people','grants','journal_club')", ((now - timedelta(hours=60)).isoformat(),))
    prev = Item("briefing", "claude", "daily:2000-01-01", "Daily briefing", stream_specific={"kind": "briefing", "period": "daily"})
    pid = db.record(prev, "shown")
    db.conn.execute("UPDATE items SET first_seen=? WHERE id=?", ((now - timedelta(hours=24)).isoformat(), pid)); db.conn.commit(); db.close()
    world[1]["streams"]["briefing"]["min_items"] = 1
    run(world)
    user = world[2].users[0]
    assert f"[i{late}]" in user and f"[i{early}]" not in user          # the 25h-old item belongs to yesterday's briefing (a flat 26h window would repeat it)
    assert "hours since the previous daily briefing" in user


def test_first_run_uses_the_nominal_lookback_and_lines_carry_publication_dates(world):
    db = DB(world[0].db_path)
    db.conn.execute("UPDATE items SET date='2026-10-29' WHERE stream='literature' AND relevance=9")
    db.conn.commit(); db.close()
    run(world)
    user = world[2].users[0]
    assert "26 hours since the previous daily briefing" in user and "| 2026-10-29 |" in user
    assert "never call an older item" in world[2].systems[0]


# ------------------------------------------------------------------ structured (sectioned) briefings
def test_parse_sections_headings_bullets_bold_and_citations():
    txt = json.dumps({"headline": "H", "sections": [
        {"heading": "At a glance", "paragraphs": ["A quiet day with **three** trials registered and one preprint on evaluation [i5]."], "bullets": []},
        {"heading": "LLM methods", "paragraphs": ["A paper reports that **retrieval** helps extraction [i3], though a second claim [i999] is unfounded.", "A short second paragraph with enough words to count."], "bullets": []},
        {"heading": "Worth a look first", "paragraphs": [], "bullets": ["The extraction paper [i3]", "The evaluation preprint [i5] and a trial [i7]"]}]})
    out = briefing.parse_briefing(txt, {3, 5, 7})
    assert [s["heading"] for s in out["sections"]] == ["At a glance", "LLM methods", "Worth a look first"]
    assert out["sections"][2]["bullets"] == ["The extraction paper [i3]", "The evaluation preprint [i5] and a trial [i7]"]
    assert "**retrieval**" in out["sections"][1]["paragraphs"][0] and "[i999]" not in json.dumps(out["sections"])
    assert out["refs"] == [5, 3, 7]                                            # citation order across paragraphs AND bullets
    assert len(out["paragraphs"]) == 3                                         # flat copy kept for older consumers


def test_flat_paragraph_shape_still_parses_as_one_untitled_section():
    out = briefing.parse_briefing(json.dumps({"headline": "h", "paragraphs": ["p" * 60, "q" * 60]}), set())
    assert len(out["sections"]) == 1 and out["sections"][0]["heading"] == "" and len(out["paragraphs"]) == 2


def test_section_limits_are_enforced():
    big = {"heading": "h" * 200, "paragraphs": ["x" * 50] * 5, "bullets": ["b" * 40] * 9}
    out = briefing.parse_briefing(json.dumps({"sections": [big]}), set())
    assert len(out["sections"][0]["heading"]) <= 70 and len(out["sections"][0]["paragraphs"]) == 3 and len(out["sections"][0]["bullets"]) == 6


class SectionClient(BriefClient):
    def create(self, **kw):
        from tests.test_budget import _R
        self.calls += 1
        user = kw["messages"][0]["content"]; self.users.append(user); self.systems.append(kw["system"][0]["text"])
        ids = re.findall(r"^\[i(\d+)\]", user, flags=re.M)
        body = {"headline": "A day of LLM evaluation", "sections": [
            {"heading": "At a glance", "paragraphs": [f"Thirty-six items today, led by clinical NLP evaluation [i{ids[0]}] and a **chatbot** trial [i{ids[1]}]."], "bullets": []},
            {"heading": "People you follow", "paragraphs": [f"A followed author published on endocarditis outcomes in people who inject drugs [i{ids[2]}]."], "bullets": []},
            {"heading": "Worth a look first", "paragraphs": [], "bullets": [f"The evaluation paper [i{ids[0]}]", f"The grant announcement [i{ids[3]}]"]}]}
        return _R(json.dumps(body), i=4000, o=800)


def test_sectioned_briefing_is_stored_and_served_with_sections(world, monkeypatch):
    import anthropic
    c = SectionClient(); monkeypatch.setattr(anthropic, "Anthropic", lambda **k: c)
    run(world)
    (b,) = items(world)
    ss = json.loads(b["stream_specific"])
    assert [s["heading"] for s in ss["sections"]] == ["At a glance", "People you follow", "Worth a look first"]
    assert ss["sections"][2]["bullets"] and len(ss["paragraphs"]) == 2 and len(ss["refs"]) == 4
    assert "## At a glance" in b["summary"] and "- The evaluation paper" in b["summary"]
    served = json.loads((world[0].site_dir / "data" / "briefing.json").read_text())["daily"]
    assert [s["heading"] for s in served["sections"]][0] == "At a glance" and served["sections"][2]["bullets"]


def test_dry_run_stub_shows_the_new_structure(world, capsys):
    run(world, dry_run=True)
    out = capsys.readouterr().out
    assert "## At a glance" in out and "## Worth a look first" in out and "* [dry-run stub] One-line pointer" in out
