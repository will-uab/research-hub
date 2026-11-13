import json

import pytest

from litsearch.core import pipeline
from litsearch.core.context import RunOptions
from litsearch.core.db import DB
from litsearch.sources import reporter as rp
from tests.helpers import FIXTURES, ScriptedClient, patch_client, settings_for

RAW = json.loads((FIXTURES / "reporter" / "search.json").read_text())["results"]


# ------------------------------------------------------------------ parsers (saved real response)
def test_parse_record_fields():
    r = rp.parse_record(RAW[0])
    assert r["project_num"] == "1R01DA065771-01" and r["institute"] == "NIDA" and r["mechanism"] == "R01"
    assert r["amount"] == 680353 and r["award_type"] == "1" and r["fiscal_year"] and r["appl_id"].isdigit()
    assert r["institution"] == "BROWN UNIVERSITY" and r["pis"][0]["name"] == "Yashaswini Singh"      # double space collapsed
    assert r["url"].startswith("https://reporter.nih.gov/project-details/")
    assert len(r["award_notice_date"]) == 10


def test_amount_falls_back_to_direct_plus_indirect_or_none():
    base = dict(RAW[0])
    assert rp.parse_record({**base, "award_amount": None, "direct_cost_amt": 100, "indirect_cost_amt": 50})["amount"] == 150
    assert rp.parse_record({**base, "award_amount": None, "direct_cost_amt": None, "indirect_cost_amt": None})["amount"] is None


def test_contact_pi_listed_first():
    r = dict(RAW[0], principal_investigators=[
        {"first_name": "A", "last_name": "Second", "full_name": "A Second", "is_contact_pi": False},
        {"first_name": "B", "last_name": "First", "full_name": "B First", "is_contact_pi": True}])
    assert [p["name"] for p in rp.parse_record(r)["pis"]] == ["B First", "A Second"]


def test_text_query_and_pi_criteria():
    q = rp.text_query(["large language model*", "LLM", "opioid use disorder", "PrEP", "LLM", "HIV-1"])
    assert q == '("large language model" OR LLM OR "opioid use disorder" OR PrEP OR "HIV-1")'
    assert rp.pi_criteria([{"name": "Doe, Jane"}, {"name": "Smith"}, {"name": ""}]) == [
        {"last_name": "Doe", "first_name": "Jane"}, {"last_name": "Smith"}]


def test_match_watch_rules():
    pis = [{"first": "Jane", "last": "Doe", "name": "Jane Doe", "contact": True},
           {"first": "J", "last": "Smith", "name": "J Smith", "contact": False}]
    m = lambda people, inst="Some University": rp.match_watch(pis, inst, people)
    assert m([{"name": "Doe, Jane"}]) == ["Doe, Jane"]
    assert m([{"name": "Doe, J"}]) == ["Doe, J"]                      # initial only
    assert m([{"name": "Doe, Janet"}]) == []                          # different first name
    assert m([{"name": "Doe"}]) == ["Doe"]                            # last name only
    assert m([{"name": "Doe, Jane", "institution": "some univ"}]) == ["Doe, Jane"]
    assert m([{"name": "Doe, Jane", "institution": "Harvard"}]) == []
    assert m([{"name": "Smith, J"}]) == ["Smith, J"] and m([{"name": "Nobody, X"}]) == []


# ------------------------------------------------------------------ the stream (RePORTER mocked)
def rec(i, title, abstract, pis=(("Pat", "Lee"),), amount=500_000, org="STATE UNIVERSITY", ic="NIDA", mech="R01"):
    return {"appl_id": 9000 + i, "project_num": f"1{mech}DA0{i}-01", "project_title": title, "abstract_text": abstract,
            "principal_investigators": [{"first_name": f, "last_name": l, "full_name": f"{f} {l}", "is_contact_pi": k == 0}
                                        for k, (f, l) in enumerate(pis)],
            "organization": {"org_name": org, "org_city": "CITY", "org_state": "ST"}, "agency_ic_admin": {"abbreviation": ic},
            "award_amount": amount, "direct_cost_amt": None, "indirect_cost_amt": None, "fiscal_year": 2027,
            "award_notice_date": "2026-10-28T00:00:00", "activity_code": mech, "award_type": "1",
            "project_start_date": "2026-11-01T00:00:00", "project_detail_url": f"https://reporter.nih.gov/project-details/{9000 + i}"}


@pytest.fixture
def world(tmp_path, monkeypatch, cfg2):
    st = {"by_label": {}, "queries": []}

    def fake_search(criteria, limiter=None, max_records=1000):
        label = "watch" if "pi_names" in criteria else "watch_ids" if "pi_profile_ids" in criteria else next(
            d for d, dom in cfg2["domains"].items() if rp.text_query(dom["keywords"]) == criteria["advanced_text_search"]["search_text"])
        st["queries"].append((label, criteria))
        v = st["by_label"].get(label, [])
        if isinstance(v, Exception):
            raise v
        return list(v), len(v)

    monkeypatch.setattr(rp, "search", fake_search)
    client = ScriptedClient(relevance=8, fit="relevant")
    patch_client(monkeypatch, client)
    cfg = json.loads(json.dumps(cfg2))
    cfg["people_to_watch"] = []            # the real config now lists people; tests opt in explicitly
    return settings_for(tmp_path), cfg, st, client


def run(world, **kw):
    s, cfg, st, client = world
    return pipeline.run(s, cfg, RunOptions(**kw), stream="reporter")


def rows(world, status=None):
    db = DB(world[0].db_path)
    out = [dict(r) for r in db.items("reporter", statuses=(status,) if status else None)]
    db.close()
    return out


def test_awards_stored_with_pi_money_institute_and_domain(world):
    st = world[2]
    st["by_label"]["hiv_addiction"] = [rec(1, "Buprenorphine retention study", "Opioid use disorder buprenorphine retention.")]
    st["by_label"]["llm_ai"] = [rec(2, "Large language models for chart review", "LLM extraction from notes.", pis=(("Ann", "Wu"), ("Bo", "Xi")))]
    assert run(world) == 0
    r = {x["source_id"]: x for x in rows(world, "shown")}
    ss = json.loads(r["9002"]["stream_specific"])
    assert ss["contact_pi"] == "Ann Wu" and ss["pis"] == ["Ann Wu", "Bo Xi"] and ss["amount"] == 500000
    assert ss["institution"] == "STATE UNIVERSITY" and ss["institute"] == "NIDA" and ss["kind"] == "award"
    assert r["9002"]["takeaway"] == "funds X" and r["9002"]["fit_note"] == "relevant" and r["9002"]["stream"] == "reporter"
    assert "$500,000" in world[3].last_user and "PI: Ann Wu" in world[3].last_user


def test_same_award_from_two_domain_queries_is_one_item_and_not_reshown(world):
    st = world[2]
    dup = rec(1, "LLM tools for HIV PrEP care", "Large language model support for PrEP.")
    st["by_label"]["llm_ai"] = [dup]; st["by_label"]["hiv_addiction"] = [dup]
    run(world); calls = world[3].calls
    assert len(rows(world, "shown")) == 1
    run(world)
    assert world[3].calls == calls and len(rows(world, "shown")) == 1


def test_untagged_awards_never_reach_the_llm(world):
    world[2]["by_label"]["infectious_diseases"] = [rec(3, "Marine sponge chemistry", "Natural products.")]
    run(world)
    assert world[3].calls == 0 and rows(world) == []


def test_watched_pi_always_shown_even_when_llm_says_irrelevant_and_even_off_topic(world):
    s, cfg, st, client = world
    cfg["people_to_watch"] = [{"name": "Lee, Pat"}]
    client.relevance = 1
    st["by_label"]["watch"] = [rec(4, "Zebrafish neural circuits", "Unrelated basic neuroscience.")]
    st["by_label"]["llm_ai"] = [rec(5, "Large language models for notes", "LLM chart review.", pis=(("Zed", "Other"),))]
    run(world)
    shown = rows(world, "shown")
    assert [x["source_id"] for x in shown] == ["9004"] and json.loads(shown[0]["stream_specific"])["watched"] == ["Lee, Pat"]
    assert [x["source_id"] for x in rows(world, "screened")] == ["9005"]


def test_watched_items_survive_the_candidate_cap(world):
    s, cfg, st, client = world
    cfg["people_to_watch"] = [{"name": "Lee, Pat"}]
    cfg["streams"]["reporter"]["max_candidates"] = 2
    st["by_label"]["llm_ai"] = [rec(10 + i, f"Large language model project {i}", "LLM machine learning deep learning.",
                                    pis=(("Zed", "Other"),)) for i in range(6)]
    st["by_label"]["watch"] = [rec(99, "Some watched thing about HIV", "HIV PrEP study.")]
    run(world)
    assert "9099" in {x["source_id"] for x in rows(world)} and len(rows(world)) <= 2


def test_watch_query_only_when_people_configured_and_award_types_applied(world):
    s, cfg, st, client = world
    run(world)
    assert {q[0] for q in st["queries"]} == {"llm_ai", "infectious_diseases", "hiv_addiction"}
    assert all(q[1]["award_types"] == ["1", "2"] for q in st["queries"])
    cfg["people_to_watch"] = [{"name": "Lee, Pat"}]
    st["queries"].clear(); run(world)
    labels = [q[0] for q in st["queries"]]
    assert "watch" in labels
    watch = next(q[1] for q in st["queries"] if q[0] == "watch")
    assert watch["pi_names"] == [{"last_name": "Lee", "first_name": "Pat"}]
    assert watch["award_types"] == ["1", "2", "3"]                       # watched people also get supplements
    assert all(q[1]["award_types"] == ["1", "2"] for q in st["queries"] if q[0] != "watch")
    cfg["streams"]["reporter"]["watch_award_types"] = ["1", "2", "3", "5"]
    st["queries"].clear(); run(world)
    assert next(q[1] for q in st["queries"] if q[0] == "watch")["award_types"] == ["1", "2", "3", "5"]


def test_window_is_14_days_by_default_and_backfill_overrides(world):
    run(world)
    c = world[2]["queries"][0][1]["award_notice_date"]
    from datetime import date
    assert (date.fromisoformat(c["to_date"]) - date.fromisoformat(c["from_date"])).days == 14
    world[2]["queries"].clear(); run(world, backfill=60)
    c = world[2]["queries"][0][1]["award_notice_date"]
    assert (date.fromisoformat(c["to_date"]) - date.fromisoformat(c["from_date"])).days == 60


def test_partial_query_failure_is_reported_total_failure_fails(world):
    st = world[2]
    st["by_label"]["hiv_addiction"] = [rec(1, "Buprenorphine retention", "Opioid use disorder buprenorphine.")]
    st["by_label"]["llm_ai"] = RuntimeError("503")
    assert run(world) == 0 and len(rows(world, "shown")) == 1
    db = DB(world[0].db_path)
    assert db.conn.execute("SELECT ok FROM runs").fetchone()["ok"] == 1
    db.close()
    for d in ("llm_ai", "infectious_diseases", "hiv_addiction"):
        st["by_label"][d] = RuntimeError("down")
    with pytest.raises(Exception, match="all RePORTER queries failed"):
        run(world)


def test_budget_trip_shows_watched_first_without_llm(world):
    s, cfg, st, client = world
    s.env["DAILY_BUDGET_USD"] = "0.0000001"
    cfg["people_to_watch"] = [{"name": "Lee, Pat"}]
    st["by_label"]["watch"] = [rec(4, "Watched project on HIV", "HIV PrEP outcomes study.")]
    st["by_label"]["llm_ai"] = [rec(5, "Large language models for notes", "LLM chart review.", pis=(("Zed", "Other"),))]
    run(world)
    assert client.calls == 0
    ids = [x["source_id"] for x in rows(world, "shown")]
    assert set(ids) == {"9004", "9005"} and all(x["relevance"] is None for x in rows(world, "shown"))


def test_dry_run_prints_samples_and_records_nothing(world, capsys):
    world[2]["by_label"]["llm_ai"] = [rec(2, "Large language models for chart review", "LLM extraction. Machine learning.")]
    run(world, dry_run=True)
    out = capsys.readouterr().out
    assert "Dry run [reporter]" in out and "1R01DA02-01" in out and "$500,000" in out
    assert not world[0].db_path.exists()


# ------------------------------------------------------------------ people pinned to a RePORTER profile id
def test_profile_id_entries_split_from_name_entries_and_parse_profile_ids():
    people = [{"name": "Rivera, Elena", "profile_id": 9000002}, {"name": "Hartley, Logan"}, {"name": "Novak, Rita", "profile_id": "9000001"}]
    assert rp.profile_ids(people) == [9000002, 9000001]
    assert rp.pi_criteria(people) == [{"last_name": "Hartley", "first_name": "Logan"}]            # pinned people are not name-searched
    r = rp.parse_record(dict(RAW[0], principal_investigators=[{"first_name": "A", "last_name": "B", "full_name": "A B",
                                                               "is_contact_pi": True, "profile_id": 42}]))
    assert r["pis"][0]["profile_id"] == 42


def test_pinned_entry_matches_only_that_person_even_with_a_namesake():
    pins = [{"name": "Hale, Andrew", "profile_id": 111}]
    right = [{"first": "Andrew", "last": "Hale", "name": "Andrew Hale", "contact": True, "profile_id": 111}]
    namesake = [{"first": "Andrew", "last": "Hale", "name": "Andrew Hale", "contact": True, "profile_id": 222}]
    assert rp.match_watch(right, "Vanderbilt", pins) == ["Hale, Andrew"]
    assert rp.match_watch(namesake, "Vanderbilt", pins) == []                                    # same name, different person
    assert rp.match_watch(namesake, "x", [{"name": "Hale, Andrew"}]) == ["Hale, Andrew"]          # name-only still matches by name
    assert rp.match_watch(right, "x", [{"name": "Hale, Andrew", "profile_id": "111"}]) == ["Hale, Andrew"]   # str/int ids tolerated


def test_stream_issues_separate_name_and_profile_id_watch_queries_and_flags_pinned_awards(world):
    s, cfg, st, client = world
    cfg["people_to_watch"] = [{"name": "Lee, Pat"}, {"name": "Wu, Ann", "profile_id": 777}]
    st["by_label"]["watch_ids"] = [dict(rec(6, "Some project by a pinned PI about HIV", "HIV PrEP study.", pis=(("Ann", "Wu"),)),
                                        principal_investigators=[{"first_name": "Ann", "last_name": "Wu", "full_name": "Ann Wu",
                                                                  "is_contact_pi": True, "profile_id": 777}])]
    run(world)
    by = {q[0]: q[1] for q in st["queries"]}
    assert by["watch"]["pi_names"] == [{"last_name": "Lee", "first_name": "Pat"}] and "pi_profile_ids" not in by["watch"]
    assert by["watch_ids"]["pi_profile_ids"] == [777] and "pi_names" not in by["watch_ids"]
    assert by["watch_ids"]["award_types"] == ["1", "2", "3"]
    shown = rows(world, "shown")
    assert json.loads(shown[0]["stream_specific"])["watched"] == ["Wu, Ann"]
