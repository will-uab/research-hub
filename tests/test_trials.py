import json

import pytest

from litsearch.core import pipeline
from litsearch.core.context import RunOptions
from litsearch.core.db import DB
from litsearch.sources import clinicaltrials as ct
from tests.helpers import FIXTURES, ScriptedClient, patch_client, settings_for

FX = FIXTURES / "trials"
load = lambda n: json.loads((FX / f"{n}.json").read_text())


# ------------------------------------------------------------------ parsers (saved real responses)
def test_parse_new_registration():
    p = ct.parse_study(load("studies_new")["studies"][0])
    assert p["nct_id"].startswith("NCT") and p["title"] and p["status"] and p["first_post"]
    assert p["interventions"] and {"type", "name", "description"} <= set(p["interventions"][0])
    assert p["sponsor"] and p["enrollment"] and p["enrollment_type"] in ("estimated", "actual")
    assert p["phases"] == ["N/A"] and p["has_results"] is False and "  " not in p["summary"]
    assert p["primary_outcomes"] == [] or all(isinstance(o, str) for o in p["primary_outcomes"])


def test_parse_trial_with_posted_results():
    ps = [ct.parse_study(s) for s in load("studies_results")["studies"]]
    assert all(p["has_results"] and p["results_first_post"] for p in ps)


def test_status_check_parse(monkeypatch):
    data = load("status_check")
    monkeypatch.setattr(ct, "request", lambda *a, **k: type("R", (), {"json": lambda self: data})())
    out = ct.fetch_status(["x"])
    assert set(out) == {s["protocolSection"]["identificationModule"]["nctId"] for s in data["studies"]}
    assert all({"status", "has_results", "results_first_post", "last_update", "why_stopped"} <= set(v) for v in out.values())


def test_term_query_and_range():
    assert ct.term_query(["large language model*", "LLM", "LLM", "HIV-1"]) == '("large language model" OR LLM OR "HIV-1")'
    assert ct.range_filter("StudyFirstPostDate", "2026-11-01") == "AREA[StudyFirstPostDate]RANGE[2026-11-01,MAX]"


# ------------------------------------------------------------------ the stream (ClinicalTrials.gov mocked)
def study(nct, title, summary="", status="RECRUITING", interventions=(("DRUG", "Drug X"),), conditions=("Opioid Use Disorder",),
          results=False, first="2026-11-02", rdate="2026-11-02"):
    st = {"overallStatus": status, "studyFirstPostDateStruct": {"date": first}, "lastUpdatePostDateStruct": {"date": first}}
    if results:
        st["resultsFirstPostDateStruct"] = {"date": rdate}
    return {"hasResults": results, "protocolSection": {
        "identificationModule": {"nctId": nct, "briefTitle": title, "officialTitle": title},
        "statusModule": st, "descriptionModule": {"briefSummary": summary},
        "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Some University", "class": "OTHER"}},
        "designModule": {"phases": ["PHASE2"], "enrollmentInfo": {"count": 120, "type": "ESTIMATED"}},
        "conditionsModule": {"conditions": list(conditions)},
        "armsInterventionsModule": {"interventions": [{"type": t, "name": n, "description": ""} for t, n in interventions]},
        "outcomesModule": {"primaryOutcomes": [{"measure": "Retention at 6 months"}]}}}


@pytest.fixture
def world(tmp_path, monkeypatch, cfg2):
    state = {"registered": [], "results": [], "status": {}, "searches": []}

    def fake_search(term, advanced, limiter=None, max_records=600):
        mode = "registered" if "StudyFirstPostDate" in advanced else "results"
        state["searches"].append((mode, term, advanced))
        seen = [s for s in state[mode] if any(w in json.dumps(s).lower() for w in ("opioid", "hiv", "artificial", "chatbot", "large language", "sepsis"))]
        return seen, len(seen)

    monkeypatch.setattr(ct, "search", fake_search)
    monkeypatch.setattr(ct, "fetch_status", lambda ids, limiter=None: {i: state["status"][i] for i in ids if i in state["status"]})
    client = ScriptedClient(relevance=8, fit="relevant")
    patch_client(monkeypatch, client)
    return settings_for(tmp_path), json.loads(json.dumps(cfg2)), state, client


def run(world, **kw):
    s, cfg, st, client = world
    return pipeline.run(s, cfg, RunOptions(**kw), stream="trials")


def rows(world, status=None):
    db = DB(world[0].db_path)
    out = [dict(r) for r in db.items("trials", statuses=(status,) if status else None)]
    db.close()
    return out


def events(world):
    db = DB(world[0].db_path)
    out = [(e["kind"], json.loads(e["detail"])) for e in db.events()]
    db.close()
    return out


def test_new_registration_stored_with_all_fields(world):
    world[2]["registered"] = [study("NCT00000001", "Buprenorphine extended release for OUD", "Opioid use disorder retention.")]
    assert run(world) == 0
    r = rows(world, "shown")[0]
    ss = json.loads(r["stream_specific"])
    assert r["source_id"] == "NCT00000001" and r["url"] == "https://clinicaltrials.gov/study/NCT00000001"
    assert ss["sponsor"] == "Some University" and ss["phases"] == ["PHASE2"] and ss["status"] == "RECRUITING"
    assert ss["enrollment"] == 120 and ss["interventions"][0]["name"] == "Drug X" and ss["primary_outcomes"] == ["Retention at 6 months"]
    assert ss["ai_intervention"] is False and "hiv_addiction" in json.loads(r["domains"])
    assert r["takeaway"] == "funds X" and r["fit_note"] == "relevant"
    assert "Sponsor: Some University" in world[3].last_user and "AI intervention: no" in world[3].last_user


def test_ai_intervention_flagged_even_without_a_domain_keyword(world):
    world[2]["registered"] = [study("NCT00000002", "Chatbot coaching for diabetes self-management", "A conversational agent.",
                                    interventions=(("BEHAVIORAL", "AI chatbot coach"),), conditions=("Diabetes",))]
    run(world)
    r = rows(world, "shown")[0]
    assert json.loads(r["stream_specific"])["ai_intervention"] is True and "llm_ai" in json.loads(r["domains"])


def test_irrelevant_trial_never_reaches_the_llm(world):
    world[2]["registered"] = [study("NCT00000003", "Knee replacement rehab", "Physio.", interventions=(("OTHER", "Physio"),),
                                    conditions=("Knee",))]
    run(world)
    assert rows(world) == [] and world[3].calls == 0


def test_second_run_does_not_reshow_known_trials(world):
    world[2]["registered"] = [study("NCT00000001", "Buprenorphine for OUD", "Opioid use disorder.")]
    run(world); calls = world[3].calls
    run(world)
    assert world[3].calls == calls and len(rows(world, "shown")) == 1


def test_status_change_recorded_for_tracked_trial_only_when_meaningful(world):
    st = world[2]
    st["registered"] = [study("NCT00000001", "Buprenorphine for OUD", "Opioid use disorder.", status="NOT_YET_RECRUITING")]
    run(world)
    st["registered"] = []
    st["status"]["NCT00000001"] = {"status": "RECRUITING", "why_stopped": "", "has_results": False, "results_first_post": "", "last_update": "2026-11-05"}
    run(world)
    assert events(world) == [("status_change", {"from": "NOT_YET_RECRUITING", "to": "RECRUITING", "why_stopped": ""})]
    st["status"]["NCT00000001"] = {"status": "TERMINATED", "why_stopped": "Funding", "has_results": False, "results_first_post": "", "last_update": "x"}
    run(world); run(world)                                     # second poll: no duplicate event
    assert [e[1]["to"] for e in events(world)] == ["RECRUITING", "TERMINATED"] and events(world)[1][1]["why_stopped"] == "Funding"
    ss = json.loads(rows(world, "shown")[0]["stream_specific"])
    assert ss["status"] == "TERMINATED"


def test_unlisted_status_change_updates_item_but_no_event(world):
    st = world[2]
    st["registered"] = [study("NCT00000001", "Buprenorphine for OUD", "Opioid use disorder.")]
    run(world); st["registered"] = []
    st["status"]["NCT00000001"] = {"status": "UNKNOWN", "why_stopped": "", "has_results": False, "results_first_post": "", "last_update": "x"}
    run(world)
    assert events(world) == [] and json.loads(rows(world)[0]["stream_specific"])["status"] == "UNKNOWN"


def test_results_posted_for_tracked_trial(world):
    st = world[2]
    st["registered"] = [study("NCT00000001", "Buprenorphine for OUD", "Opioid use disorder.")]
    run(world); st["registered"] = []
    st["status"]["NCT00000001"] = {"status": "COMPLETED", "why_stopped": "", "has_results": True, "results_first_post": "2026-11-09", "last_update": "x"}
    run(world)
    kinds = [e[0] for e in events(world)]
    assert kinds == ["status_change", "results_posted"] and events(world)[1][1] == {"date": "2026-11-09"}
    assert json.loads(rows(world)[0]["stream_specific"])["has_results"] is True


def test_finished_trials_with_results_are_no_longer_polled(world, monkeypatch):
    st = world[2]
    st["registered"] = [study("NCT00000001", "Buprenorphine for OUD", "Opioid use disorder.", status="COMPLETED", results=True)]
    run(world); st["registered"] = []
    polled = []
    monkeypatch.setattr(ct, "fetch_status", lambda ids, limiter=None: polled.append(ids) or {})
    run(world)
    assert polled == []


def test_new_trial_arriving_with_results_gets_a_results_event(world):
    world[2]["results"] = [study("NCT00000009", "Old OUD trial reports results", "Opioid use disorder.", status="COMPLETED", results=True,
                                 first="2019-01-01", rdate="2026-11-01")]
    run(world)
    r = rows(world, "shown")[0]
    assert json.loads(r["stream_specific"])["has_results"] and r["date"] == "2026-11-01"
    assert events(world) == [("results_posted", {"date": "2026-11-01", "on_arrival": True})]


def test_screened_out_trial_gets_no_event_and_is_not_polled(world):
    world[3].relevance = 1
    world[2]["results"] = [study("NCT00000009", "Opioid use disorder trial", "Opioid use disorder.", status="COMPLETED", results=True)]
    run(world)
    assert rows(world, "shown") == [] and len(rows(world, "screened")) == 1 and events(world) == []


def test_queries_cover_domains_ai_and_both_event_types(world):
    run(world)
    labels = [(m, t) for m, t, _ in world[2]["searches"]]
    assert {m for m, _ in labels} == {"registered", "results"} and len(labels) == 8          # 4 term sets x 2 event types
    assert any('"large language model"' in t for _, t in labels) and any("chatbot" in t for _, t in labels)
    assert all(a.startswith("AREA[") and a.endswith(",MAX]") for _, _, a in world[2]["searches"])


def test_all_queries_failing_fails_the_run_partial_does_not(world, monkeypatch):
    calls = {"n": 0}
    def flaky(term, adv, limiter=None, max_records=600):
        calls["n"] += 1
        if calls["n"] % 2:
            raise RuntimeError("503")
        return [], 0
    monkeypatch.setattr(ct, "search", flaky)
    assert run(world) == 0
    def dead(*a, **k): raise RuntimeError("down")
    monkeypatch.setattr(ct, "search", dead)
    with pytest.raises(Exception, match="all ClinicalTrials.gov queries failed"):
        run(world)


def test_budget_trip_still_shows_trials_keyword_ranked(world):
    world[0].env["DAILY_BUDGET_USD"] = "0.0000001"
    world[2]["registered"] = [study("NCT00000001", "Buprenorphine for OUD", "Opioid use disorder.")]
    run(world)
    r = rows(world, "shown")
    assert world[3].calls == 0 and len(r) == 1 and r[0]["relevance"] is None


def test_dry_run_prints_samples_and_records_nothing(world, capsys):
    world[2]["registered"] = [study("NCT00000002", "Chatbot coaching", "AI conversational agent for OUD.",
                                    interventions=(("BEHAVIORAL", "AI chatbot"),))]
    run(world, dry_run=True)
    out = capsys.readouterr().out
    assert "Dry run [trials]" in out and "NCT00000002" in out and "AI" in out
    assert not world[0].db_path.exists()
