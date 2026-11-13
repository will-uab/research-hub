import json

import pytest

from litsearch.core import pipeline
from litsearch.core.context import RunOptions
from litsearch.core.db import DB
from litsearch.core.models import Paper
from litsearch.core.triage import ParseError
from litsearch.streams import journal_club as jc
from tests.helpers import settings_for

ORDER = ["llm_ai", "infectious_diseases", "hiv_addiction"]
row = lambda i, rel, doms, kw=1: {"id": i, "relevance": rel, "kw_score": kw, "domains": doms}


# ------------------------------------------------------------------ rotation + selection (pure)
def test_rotation_domain_cycles_and_skips_empty_domains():
    assert jc.rotation_domain(ORDER, None, set(ORDER)) == "llm_ai"
    assert jc.rotation_domain(ORDER, "llm_ai", set(ORDER)) == "infectious_diseases"
    assert jc.rotation_domain(ORDER, "hiv_addiction", set(ORDER)) == "llm_ai"             # wraps
    assert jc.rotation_domain(ORDER, "llm_ai", {"llm_ai", "hiv_addiction"}) == "hiv_addiction"   # ID has nothing: skip it
    assert jc.rotation_domain(ORDER, "llm_ai", set()) is None
    assert jc.rotation_domain(ORDER, "bogus", {"hiv_addiction"}) == "hiv_addiction"


def test_primary_follows_the_rotation_domain_not_the_global_top():
    pool = [row(1, 10, ["llm_ai"]), row(2, 8, ["infectious_diseases"]), row(3, 7, ["hiv_addiction"]), row(4, 9, ["llm_ai"])]
    p, alts, d = jc.choose(pool, ORDER, "llm_ai", True)
    assert d == "infectious_diseases" and p["id"] == 2
    assert [a["id"] for a in alts] == [1, 4]                       # next best overall, by relevance
    p, alts, d = jc.choose(pool, ORDER, None, True)
    assert d == "llm_ai" and p["id"] == 1 and [a["id"] for a in alts] == [4, 2]


def test_rotation_off_picks_the_global_top_and_ties_break_by_keyword_score():
    pool = [row(1, 8, ["hiv_addiction"], kw=5), row(2, 8, ["hiv_addiction"], kw=9), row(3, 6, [])]
    p, alts, d = jc.choose(pool, ORDER, "llm_ai", False)
    assert d is None and p["id"] == 2 and [a["id"] for a in alts] == [1, 3]


def test_choose_handles_tiny_pools():
    assert jc.choose([], ORDER, None, True) == (None, [], None)
    p, alts, _ = jc.choose([row(1, 9, ["llm_ai"])], ORDER, None, True)
    assert p["id"] == 1 and alts == []


# ------------------------------------------------------------------ guide parsing
GOOD = {"id": "p1", "why_discuss": "w", "questions": ["a?", "b?", "c?", "d?"], "methods_point": "m", "background_reading": ["TRIPOD-LLM"]}


def test_parse_guide_ok_fenced_and_wrapped():
    assert jc.parse_guide(json.dumps([GOOD]), ["p1"])["p1"]["questions"] == ["a?", "b?", "c?", "d?"]
    assert "p1" in jc.parse_guide("```json\n" + json.dumps([GOOD]) + "\n```", ["p1"])
    assert "p1" in jc.parse_guide(json.dumps({"picks": [GOOD]}), ["p1"])


def test_parse_guide_drops_incomplete_and_caps_lists():
    many = dict(GOOD, questions=["q"] * 6, background_reading=["a", "b", "c", "d"])
    out = jc.parse_guide(json.dumps([many, dict(GOOD, id="p2", methods_point=""), dict(GOOD, id="p3", questions=["only one"]),
                                     dict(GOOD, id="p9")]), ["p1", "p2", "p3"])
    assert list(out) == ["p1"]                                      # p2 no methods point, p3 one question, p9 unknown id
    assert len(out["p1"]["questions"]) == 4 and len(out["p1"]["background_reading"]) == 3
    assert jc.parse_guide(json.dumps([dict(GOOD, background_reading="x")]), ["p1"])["p1"]["background_reading"] == []


@pytest.mark.parametrize("bad", ["", "nope", '{"a": 1}', "[1,"])
def test_parse_guide_malformed_raises(bad):
    with pytest.raises(ParseError):
        jc.parse_guide(bad, ["p1"])


# ------------------------------------------------------------------ the stream (seeded DB, fake model)
ABSTRACT = "We evaluated large language models for extracting outcomes from clinical notes in a multicenter cohort. " * 4


class GuideClient:
    """Fake Anthropic client for the journal club call. `bad` = number of initial malformed replies."""

    def __init__(self, bad=0):
        self.bad, self.calls, self.messages, self.last_kwargs = bad, 0, self, {}

    def create(self, **kw):
        import re
        from tests.test_budget import _R
        self.calls += 1
        self.last_kwargs = kw
        if self.calls <= self.bad:
            return _R("not json", i=3000, o=100)
        ids = re.findall(r"^\[(p\d+)\]", kw["messages"][0]["content"], flags=re.M)
        return _R(json.dumps([dict(GOOD, id=i) for i in ids]), i=3000, o=1500)


def seed(db, n=5, domains=(("llm_ai",), ("infectious_diseases",), ("hiv_addiction",), ("llm_ai",), ("hiv_addiction",)), rel=(9, 8, 7, 6, 8)):
    ids = []
    for i in range(n):
        p = Paper("pubmed", f"Paper number {i} about large language models in clinical care", ABSTRACT, pmid=str(100 + i),
                  journal="JAMIA", authors=["A B"], domains=list(domains[i]), relevance=rel[i], takeaway=f"takeaway {i}", kw_score=i)
        ids.append(db.record(p, "sent"))
    return ids


JC_ENV = {"LLM_MODEL": "claude-haiku-5-5", "LLM_MODEL_JOURNAL_CLUB": "claude-sonnet-5-5", "LLM_EFFORT_JOURNAL_CLUB": "medium",
          "PRICE_JOURNAL_CLUB_INPUT_PER_MTOK": "2", "PRICE_JOURNAL_CLUB_OUTPUT_PER_MTOK": "10", "WEEKLY_EXTRA_USD": "0.10"}


@pytest.fixture
def world(tmp_path, monkeypatch, cfg2):
    s = settings_for(tmp_path, **JC_ENV)
    client = GuideClient()
    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", lambda **k: client)
    db = DB(s.db_path)
    seed(db)
    db.close()
    return s, json.loads(json.dumps(cfg2)), client


def run(world, **kw):
    s, cfg, client = world
    return pipeline.run(s, cfg, RunOptions(**kw), stream="journal_club")


def picks(world):
    db = DB(world[0].db_path)
    out = [dict(r) for r in db.items("journal_club")]
    db.close()
    return out


def test_week_picks_one_primary_two_alternates_with_guides(world):
    assert run(world) == 0
    ps = picks(world)
    assert [json.loads(p["stream_specific"])["role"] for p in ps] == ["primary", "alternate", "alternate"]
    prim = json.loads(ps[0]["stream_specific"])
    assert prim["generated"] and prim["questions"][0] == "a?" and prim["abstract_only"] is True
    assert prim["pick_domain"] == "llm_ai" and ps[0]["title"].startswith("Paper number 0")           # first rotation domain, best llm_ai paper
    assert ps[0]["source_id"].endswith(":primary") and ps[1]["source_id"].endswith(":alt1")
    assert world[2].calls == 1                                                                         # ONE call for all three picks
    kw = world[2].last_kwargs
    assert kw["model"] == "claude-sonnet-5-5" and kw["output_config"] == {"effort": "medium"}
    assert "Abstract:" in kw["messages"][0]["content"] and "ONLY their abstracts" in kw["system"][0]["text"]


def test_spend_is_logged_under_journal_club_with_its_own_prices(world):
    run(world)
    db = DB(world[0].db_path)
    r = db.conn.execute("SELECT * FROM spend WHERE stream='journal_club'").fetchone()
    assert r["calls"] == 1 and r["cost_usd"] == pytest.approx((3000 * 2 + 1500 * 10) / 1e6)       # $0.021 at Sonnet prices
    db.close()


def test_idempotent_within_a_week_and_rotates_next_week(world, monkeypatch):
    run(world); run(world)
    assert len(picks(world)) == 3 and world[2].calls == 1
    # next week's run: pretend the week key changed; the rotation moves on and the same papers are never re-picked
    s, cfg, client = world
    db = DB(s.db_path)
    db.conn.execute("UPDATE items SET source_id=REPLACE(source_id, substr(source_id,1,8), '2020-W01') WHERE stream='journal_club'")
    db.conn.execute("UPDATE item_keys SET value=REPLACE(value, substr(value,4,8), '2020-W01') WHERE kind='id' AND stream='journal_club'")
    db.conn.commit(); db.close()
    run(world)
    ps = picks(world)
    new = [p for p in ps if not p["source_id"].startswith("2020-W01")]
    old_papers = {json.loads(p["stream_specific"])["paper_item_id"] for p in ps if p["source_id"].startswith("2020-W01")}
    assert len(new) == 2 and not ({json.loads(p["stream_specific"])["paper_item_id"] for p in new} & old_papers)   # only 2 unpicked papers left
    # week 1 took llm_ai as primary and the ID paper as an alternate, so ID has nothing left and rotation skips to hiv_addiction
    assert json.loads(new[0]["stream_specific"])["pick_domain"] == "hiv_addiction" and new[0]["title"].startswith("Paper number 2")


def test_papers_below_threshold_without_abstract_or_older_than_a_week_are_ineligible(world):
    s, cfg, client = world
    db = DB(s.db_path)
    db.conn.execute("UPDATE items SET relevance=2 WHERE stream='literature'")
    db.conn.execute("UPDATE items SET relevance=9 WHERE id=1")
    db.conn.execute("UPDATE items SET summary='' WHERE id=1")                       # migrated-from-v1 style: no abstract
    db.conn.commit(); db.close()
    run(world)
    assert picks(world) == [] and client.calls == 0
    db = DB(s.db_path)
    db.conn.execute("UPDATE items SET relevance=9, summary=?, first_seen='2020-01-01T00:00:00+00:00' WHERE stream='literature'", (ABSTRACT,))
    db.conn.commit(); db.close()
    run(world)
    assert picks(world) == []                                                       # too old


def test_malformed_json_retries_once_then_picks_without_guide(world, monkeypatch):
    import anthropic
    client = GuideClient(bad=1)
    monkeypatch.setattr(anthropic, "Anthropic", lambda **k: client)
    run(world)
    assert client.calls == 2 and all(json.loads(p["stream_specific"])["generated"] for p in picks(world))
    # reset and fail twice
    s = world[0]
    db = DB(s.db_path); db.conn.execute("DELETE FROM items WHERE stream='journal_club'"); db.conn.execute("DELETE FROM item_keys WHERE stream='journal_club'")
    db.conn.commit(); db.close()
    client2 = GuideClient(bad=5)
    monkeypatch.setattr(anthropic, "Anthropic", lambda **k: client2)
    run(world)
    ps = picks(world)
    assert client2.calls == 2 and len(ps) == 3 and not any(json.loads(p["stream_specific"])["generated"] for p in ps)
    db = DB(s.db_path)
    notes = json.loads(db.conn.execute("SELECT notes FROM runs ORDER BY id DESC LIMIT 1").fetchone()["notes"])
    assert any("not valid JSON" in n for n in notes)


def test_weekly_allowance_blocks_the_call_but_picks_are_still_stored(world):
    world[0].env["WEEKLY_EXTRA_USD"] = "0.0000001"
    run(world)
    ps = picks(world)
    assert world[2].calls == 0 and len(ps) == 3 and not json.loads(ps[0]["stream_specific"])["generated"]
    db = DB(world[0].db_path)
    notes = json.loads(db.conn.execute("SELECT notes FROM runs").fetchone()["notes"])
    assert any("weekly allowance" in n for n in notes)


def test_daily_cap_does_not_apply_to_journal_club(world):
    world[0].env["DAILY_BUDGET_USD"] = "0.0000001"            # would block every normal stream
    run(world)
    assert world[2].calls == 1 and all(json.loads(p["stream_specific"])["generated"] for p in picks(world))


def test_different_model_without_its_prices_is_refused(world):
    for k in ("PRICE_JOURNAL_CLUB_INPUT_PER_MTOK", "PRICE_JOURNAL_CLUB_OUTPUT_PER_MTOK"):
        world[0].env.pop(k)
    with pytest.raises(RuntimeError, match="PRICE_JOURNAL_CLUB_INPUT_PER_MTOK"):
        run(world)
    assert world[2].calls == 0
    world[0].env.pop("LLM_MODEL_JOURNAL_CLUB")                 # same model as LLM_MODEL: shared prices are fine
    assert run(world) == 0


def test_dry_run_uses_stubs_no_api_and_records_nothing(world, capsys):
    run(world, dry_run=True)
    out = capsys.readouterr().out
    assert "Dry run [journal_club]" in out and "PRIMARY" in out and out.count("ALTERNATE") == 2
    assert world[2].calls == 0 and picks(world) == []


def test_dry_run_after_real_picks_exist_does_not_crash_or_duplicate(world, capsys):
    run(world)                                         # real picks for this week
    before = len(picks(world))
    assert run(world, dry_run=True) == 0
    assert "already exist" in capsys.readouterr().out and len(picks(world)) == before
