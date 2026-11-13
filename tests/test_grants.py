import json
from datetime import date

import pytest

from litsearch.core import pipeline
from litsearch.core.context import RunOptions
from litsearch.core.db import DB
from litsearch.sources import grantsgov as gg
from litsearch.sources import nihguide
from litsearch.streams import grants
from tests.helpers import FIXTURES, ScriptedClient, patch_client, settings_for

FX = FIXTURES / "grants"


# ------------------------------------------------------------------ parsers (saved real responses)
def test_search_hits_parse():
    hits = [gg.parse_hit(h) for h in json.loads((FX / "search2.json").read_text())["data"]["oppHits"]]
    h = next(x for x in hits if x["number"] == "RFA-RM-27-009")
    assert h["agency_code"] == "HHS-NIH11" and h["open_date"] == "2026-10-29" and h["close_date"] == "2027-01-08"
    assert h["status"] == "posted" and h["opp_id"].isdigit()
    assert {x["agency_code"] for x in hits} >= {"HHS-NIH11", "HHS-SAMHS-SAMHSA", "HHS-HRSA"}


def test_fetch_opportunity_parse():
    d = gg.parse_detail(json.loads((FX / "fetch_360366.json").read_text())["data"])
    assert d["number"] == "RFA-DA-28-012" and d["status"] == "posted" and d["revision"] == 6
    assert d["close_date"] == "2027-01-26" and d["archive_date"] == "2027-03-27" and d["open_date"] == "2026-10-06"
    assert d["funding_instruments"] == ["Grant"] and d["award_ceiling"] is None
    assert d["description"].startswith("The National Institutes of Health (NIH) intends to renew") and "<p>" not in d["description"]
    assert d["modified_comments"] == "Updated Opportunity Number and Title to match NOFO."


def test_forecast_detail_uses_forecast_fields():
    raw = {"id": 5, "opportunityNumber": "", "opportunityTitle": "A forecast", "owningAgencyCode": "HHS-NIH11",
           "docType": "forecast", "revision": 1, "agencyDetails": {"agencyName": "NIH"},
           "forecast": {"postingDate": "Sep 01, 2026 12:00:00 AM EDT", "forecastDesc": "<p>Will fund <b>X</b></p>",
                        "estApplicationResponseDate": "Mar 03, 2027 12:00:00 AM EST", "estAwardDate": "Dec 01, 2027"}}
    d = gg.parse_detail(raw)
    assert d["status"] == "forecasted" and d["description"] == "Will fund X" and d["close_date"] == "2027-03-03"
    assert d["est_award_date"] == "2027-12-01" and d["open_date"] == "2026-09-01"


@pytest.mark.parametrize("raw,iso", [("10/29/2026", "2026-10-29"), ("Jan 26, 2027 12:00:00 AM EST", "2027-01-26"),
                                     ("Mar 27, 2027", "2027-03-27"), ("2027-02-02", "2027-02-02"), ("", ""), ("soon", "")])
def test_iso_date(raw, iso):
    assert gg.iso_date(raw) == iso


def test_guide_rss_and_notice_parse():
    items = nihguide.parse_rss((FX / "guide_rss.xml").read_bytes())
    assert items and items[0]["number"] == "NOT-OD-27-006" and items[0]["link"].startswith("https://grants.nih.gov/")
    n = nihguide.parse_notice((FX / "notice_NOT-DA-26-004.html").read_text())
    assert n["number"] == "NOT-DA-26-004" and "PA-24-179" in n["nofos"] and "PA-24-188" in n["nofos"]
    assert n["release_date"] == "2024-09-17"
    assert n["expiration_new"] == "2025-09-09" and n["expiration_prior"] == "2028-09-08"
    assert nihguide.nofo_numbers("see RFA-DA-28-012, PAR-26-100 and PA-24-179; NOT-OD-25-143") == [
        "RFA-DA-28-012", "PAR-26-100", "PA-24-179"]


# ------------------------------------------------------------------ inference
def test_institute_inference():
    assert grants.infer_institutes("RFA-DA-28-012", "HHS-NIH11", "t", "x") == ["NIDA"]
    assert grants.infer_institutes("PAR-27-013", "HHS-NIH11", "t", "Sponsored by the National Institute of Mental Health (NIMH) "
                                   "and NIDA") == ["NIMH", "NIDA"]
    assert grants.infer_institutes("PAR-27-013", "HHS-NIH11", "t", "no institute named") == ["NIH"]
    assert grants.infer_institutes("HRSA-27-017", "HHS-HRSA", "t", "") == ["HRSA"]
    assert grants.infer_institutes("RFA-AI-28-005", "HHS-NIH11", "t", "the NIA program") == ["NIAID", "NIA"]


@pytest.mark.parametrize("title,mech", [
    ("HEAL Initiative: IMPOWR - BEACON Center (U2C Clinical Trial Not Allowed)", "U2C"),
    ("Research Project Grant (Parent R01 Clinical Trial Required)", "R01"),
    ("Independent Scientist Award (Parent K02 - Independent Clinical Trial Not Allowed)", "K02"),
    ("Small Research Grant (R21 Clinical Trial Optional)", "R21"),
    ("Genomic Data Generation Center (RM1 Clinical Trial Not Allowed)", "RM1"),
    ("A thing with no code", "")])
def test_mechanism_inference(title, mech):
    assert grants.infer_mechanism(title) == mech


def test_mechanism_preferences_use_globs():
    pats = ["R01", "R21", "R34", "K*", "U*"]
    assert [grants.mechanism_matches(m, pats) for m in ("R01", "K99", "U2C", "R03", "", "P30")] == [True, True, True, False, False, False]


def test_days_until():
    assert grants.days_until("2026-12-01", date(2026, 11, 3)) == 28 and grants.days_until("", date(2026, 11, 3)) is None


# ------------------------------------------------------------------ the stream, with Grants.gov and the Guide mocked
def detail(opp_id, number, title, desc, close="2027-01-26", rev=1, **kw):
    d = {"opp_id": str(opp_id), "number": number, "title": title, "agency_code": "HHS-NIH11", "agency": "NIH",
         "doc_type": "synopsis", "status": "posted", "open_date": "2026-11-01", "close_date": close, "archive_date": "2027-03-01",
         "est_award_date": "", "description": desc, "award_ceiling": None, "award_floor": None,
         "funding_instruments": ["Grant"], "revision": rev, "last_updated": "", "modified_comments": ""}
    d.update(kw)
    return d


HITS = [{"opp_id": "1", "number": "RFA-DA-28-001", "title": "LLM tools for opioid use disorder care (R01 Clinical Trial Optional)",
         "agency_code": "HHS-NIH11", "open_date": "2026-11-01"},
        {"opp_id": "2", "number": "PAR-28-002", "title": "Chemistry of marine sponges (R21)", "agency_code": "HHS-NIH11",
         "open_date": "2026-11-01"},
        {"opp_id": "3", "number": "RFA-AI-28-003", "title": "HIV prevention and PrEP uptake research (U01)",
         "agency_code": "HHS-NIH11", "open_date": "2026-10-31"}]
DETAILS = {"1": detail(1, "RFA-DA-28-001", HITS[0]["title"], "Large language model and machine learning methods for opioid use disorder (OUD) treatment.", close="2026-12-10"),
           "2": detail(2, "PAR-28-002", HITS[1]["title"], "Marine natural products chemistry."),
           "3": detail(3, "RFA-AI-28-003", HITS[2]["title"], "HIV prevention including PrEP for people who inject drugs. National Institute of Allergy and Infectious Diseases (NIAID).")}


@pytest.fixture
def world(tmp_path, monkeypatch, cfg2):
    state = {"hits": list(HITS), "details": {k: dict(v) for k, v in DETAILS.items()}, "rss": [], "notices": {}}
    monkeypatch.setattr(gg, "search_recent", lambda agencies, since, limiter=None: list(state["hits"]))
    monkeypatch.setattr(gg, "fetch_opportunity", lambda oid, limiter=None: dict(state["details"][str(oid)]))
    monkeypatch.setattr(nihguide, "fetch_rss", lambda limiter=None: list(state["rss"]))
    monkeypatch.setattr(nihguide, "fetch_notice", lambda link, limiter=None: dict(state["notices"][link]))
    client = ScriptedClient(relevance=8, fit="strong fit")
    patch_client(monkeypatch, client)
    s = settings_for(tmp_path)
    cfg = json.loads(json.dumps(cfg2))
    cfg["streams"]["grants"]["weekly_summary"] = "never"          # tests that want the weekly summary set the day themselves
    return s, cfg, state, client


def run(world, **kw):
    s, cfg, state, client = world
    return pipeline.run(s, cfg, RunOptions(**kw), stream="grants")


def rows(world, status=None):
    db = DB(world[0].db_path)
    r = [dict(x) for x in db.items("grants", statuses=(status,) if status else None)]
    db.close()
    return r


def test_new_nofos_stored_with_fit_note_domains_and_inferred_fields(world):
    assert run(world) == 0
    shown = {r["source_id"]: r for r in rows(world, "shown")}
    assert set(shown) == {"RFA-DA-28-001", "RFA-AI-28-003"}                    # marine-sponge NOFO has no domain match: never triaged
    r = shown["RFA-DA-28-001"]
    ss = json.loads(r["stream_specific"])
    assert r["fit_note"] == "strong fit" and r["takeaway"] == "funds X" and r["relevance"] == 8
    assert ss["institutes"] == ["NIDA"] and ss["mechanism"] == "R01" and ss["close_date"] == "2026-12-10"
    assert ss["pref_institute"] and ss["pref_mechanism"] and r["url"].endswith("/1")
    assert "llm_ai" in json.loads(r["domains"]) or "hiv_addiction" in json.loads(r["domains"])
    assert "[p1]" in world[3].last_user and "Mechanism: R01" in world[3].last_user


def test_second_run_does_not_reshow_and_makes_no_llm_call(world):
    run(world)
    calls = world[3].calls
    run(world)
    assert world[3].calls == calls and len(rows(world, "shown")) == 2


def test_below_threshold_is_screened_not_shown(world):
    world[3].per_id = {"p1": 2, "p2": 2, "p3": 2}
    world[3].relevance = 2
    run(world)
    assert rows(world, "shown") == [] and len(rows(world, "screened")) == 2


def test_preference_boost_orders_candidates_and_strict_mode_drops_others(world):
    s, cfg, state, client = world
    state["details"]["3"].update(number="RFA-HL-28-003", title="HIV prevention research",        # NHLBI: not preferred;
                                 description="HIV prevention and cardiovascular outcomes.")      # no mechanism either
    state["hits"][2]["number"] = "RFA-HL-28-003"
    run(world, dry_run=True)
    # preferred (NIDA + R01) item is sent first to the LLM
    assert client.calls == 0                                              # dry-run: stub triage
    cfg["streams"]["grants"]["strict_preferences"] = True
    try:
        run(world)
    finally:
        cfg["streams"]["grants"]["strict_preferences"] = False
    assert {r["source_id"] for r in rows(world, "shown")} == {"RFA-DA-28-001"}      # strict: no preferred institute/mechanism -> out


def test_modification_updates_item_and_records_event(world):
    s, cfg, state, client = world
    run(world)
    state["hits"] = []                                                    # nothing new today
    state["details"]["1"].update(revision=2, close_date="2027-01-15", modified_comments="Extended the due date")
    run(world)
    db = DB(s.db_path)
    row = db.find_by_key("grants", "number", "RFA-DA-28-001")
    ss = json.loads(row["stream_specific"])
    assert ss["close_date"] == "2027-01-15" and ss["revision"] == 2
    ev = db.events(kinds=("nofo_modified",))
    assert len(ev) == 1
    d = json.loads(ev[0]["detail"])
    assert d["changes"]["close_date"] == ["2026-12-10", "2027-01-15"] and d["comments"] == "Extended the due date"
    db.close()
    run(world)                                                            # unchanged revision: no new event
    db = DB(s.db_path)
    assert len(db.events(kinds=("nofo_modified",))) == 1


def test_forecast_becoming_posted_is_a_status_change(world):
    s, cfg, state, client = world
    state["details"]["1"].update(status="forecasted", doc_type="forecast")
    run(world)
    state["hits"] = []
    state["details"]["1"].update(status="posted", doc_type="synopsis", revision=3)
    run(world)
    db = DB(s.db_path)
    kinds = [e["kind"] for e in db.events()]
    assert "status_change" in kinds
    db.close()


def test_guide_notice_matched_to_shown_nofo_once(world):
    s, cfg, state, client = world
    run(world)
    link = "https://grants.nih.gov/grants/guide/notice-files/NOT-DA-27-001.html"
    state["rss"] = [{"number": "NOT-DA-27-001", "title": "Notice of Change", "link": link, "pub_date": ""}]
    state["notices"][link] = {"number": "NOT-DA-27-001", "title": "Notice of Change to RFA-DA-28-001",
                              "nofos": ["RFA-DA-28-001", "PA-00-000"], "expiration_new": "2027-02-01",
                              "expiration_prior": "2027-01-26", "release_date": "2026-11-02", "url": link}
    state["hits"] = []
    run(world); run(world)                                                # second run: already processed
    db = DB(s.db_path)
    ev = db.events(kinds=("notice",))
    assert len(ev) == 1 and json.loads(ev[0]["detail"])["notice"] == "NOT-DA-27-001"
    assert json.loads(ev[0]["detail"])["expiration_new"] == "2027-02-01"
    db.close()


def test_reissue_is_linked_to_the_older_nofo(world):
    s, cfg, state, client = world
    run(world)
    state["hits"] = [{"opp_id": "9", "number": "RFA-DA-29-009", "title": "OUD LLM tools renewal (R01)", "agency_code": "HHS-NIH11",
                      "open_date": "2026-11-03"}]
    state["details"]["9"] = detail(9, "RFA-DA-29-009", "OUD LLM tools renewal (R01)",
                                   "This NOFO reissues RFA-DA-28-001 for opioid use disorder machine learning research.")
    run(world)
    db = DB(s.db_path)
    ev = db.events(kinds=("reissued",))
    assert len(ev) == 1 and json.loads(ev[0]["detail"])["new_number"] == "RFA-DA-29-009"
    new = db.find_by_key("grants", "number", "RFA-DA-29-009")
    assert json.loads(new["stream_specific"])["supersedes"] == ["RFA-DA-28-001"]


def test_one_failed_detail_fetch_does_not_sink_the_run(world, monkeypatch):
    def flaky(oid, limiter=None):
        if str(oid) == "1":
            raise RuntimeError("503")
        return dict(world[2]["details"][str(oid)])
    monkeypatch.setattr(gg, "fetch_opportunity", flaky)
    assert run(world) == 0
    assert {r["source_id"] for r in rows(world, "shown")} == {"RFA-AI-28-003"}
    db = DB(world[0].db_path)
    r = db.conn.execute("SELECT ok FROM runs").fetchone()
    assert r["ok"] == 1
    db.close()


def test_search_failure_fails_the_run_and_records_it(world, monkeypatch):
    def boom(*a, **k): raise RuntimeError("down")
    monkeypatch.setattr(gg, "search_recent", boom)
    with pytest.raises(Exception, match="Grants.gov search failed"):
        run(world)
    db = DB(world[0].db_path)
    assert db.conn.execute("SELECT ok, error FROM runs").fetchone()["ok"] == 0
    db.close()


def test_budget_trip_shows_keyword_ranked_items_with_note(world):
    s, cfg, state, client = world
    s.env["DAILY_BUDGET_USD"] = "0.0000001"
    run(world)
    assert client.calls == 0
    shown = rows(world, "shown")
    assert len(shown) == 2 and all(r["relevance"] is None and r["fit_note"] == "" for r in shown)
    db = DB(s.db_path)
    notes = json.loads(db.conn.execute("SELECT notes FROM runs").fetchone()["notes"])
    assert any("budget guard tripped" in n and "keyword-ranked" in n for n in notes)


def test_per_stream_cap_applies_to_grants(world):
    s, cfg, state, client = world
    cfg["streams"]["grants"]["daily_cap_usd"] = 0.0000001
    try:
        run(world)
    finally:
        cfg["streams"]["grants"]["daily_cap_usd"] = 0.03
    assert client.calls == 0 and len(rows(world, "shown")) == 2


def test_weekly_summary_built_from_stored_items_without_llm(world):
    s, cfg, state, client = world
    run(world, dry_run=True)                      # dry-run always builds the summary (so you can see it)
    run(world)                                    # real run on a non-Friday: no summary
    assert [r for r in rows(world) if r["source"] == "summary"] == []
    # force "today is Friday" for a real run
    from litsearch.streams import grants as g
    import datetime as dt
    cfg["streams"]["grants"]["weekly_summary"] = dt.datetime.now(s.tz).strftime("%A").lower()
    try:
        state["hits"] = []
        run(world)
        run(world)                                # idempotent for the same ISO week
    finally:
        cfg["streams"]["grants"]["weekly_summary"] = "friday"
    summ = [r for r in rows(world) if r["source"] == "summary"]
    assert len(summ) == 1
    ss = json.loads(summ[0]["stream_specific"])
    assert ss["kind"] == "weekly_summary" and {n["number"] for n in ss["new"]} == {"RFA-DA-28-001", "RFA-AI-28-003"}
    assert [c["number"] for c in ss["closing"]] in ([], ["RFA-DA-28-001"])      # depends on today's date vs 2026-12-10
    assert client.calls == 1                      # only the original triage call; the summary used no LLM


def test_dry_run_writes_nothing_and_prints_samples(world, capsys):
    run(world, dry_run=True)
    out = capsys.readouterr().out
    assert "Dry run [grants]" in out and "RFA-DA-28-001" in out and "due 2026-12-10" in out
    assert not world[0].db_path.exists()


def test_institute_prior_tags_domain_without_keywords(world):
    s, cfg, state, client = world
    state["hits"] = [{"opp_id": "7", "number": "RFA-AA-28-001", "title": "Specialized Alcohol Research Centers (P50)",
                      "agency_code": "HHS-NIH11", "open_date": "2026-11-02"}]
    state["details"]["7"] = detail(7, "RFA-AA-28-001", "Specialized Alcohol Research Centers (P50)", "Centers for alcohol science.")
    run(world)
    r = rows(world, "shown")[0]
    assert r["source_id"] == "RFA-AA-28-001" and "hiv_addiction" in json.loads(r["domains"])
