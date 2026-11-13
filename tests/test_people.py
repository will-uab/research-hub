import json

import pytest

from litsearch.core import pipeline, watch
from litsearch.core.context import RunOptions
from litsearch.core.db import DB
from litsearch.core.models import Paper
from litsearch.sources import openalex as oa
from litsearch.streams import literature, people
from tests.helpers import FIXTURES, ScriptedClient, patch_client, settings_for

FX = FIXTURES / "openalex"
load = lambda n: json.loads((FX / f"{n}.json").read_text())


# ------------------------------------------------------------------ OpenAlex parsing (saved real responses)
def test_decode_abstract_rebuilds_text_in_order():
    assert oa.decode_abstract({"Hello": [0, 3], "world": [1], "again": [4], "big": [2]}) == "Hello world big Hello again"
    assert oa.decode_abstract(None) == "" and oa.decode_abstract({}) == ""


def test_parse_real_works_and_skip_types():
    ws = load("works")["results"]
    out = [oa.parse_work(w, {"A5000000001": "Rivera, Elena"}) for w in ws]
    kept = [o for o in out if o]
    assert len(kept) == len(ws) and all(o["watched"] == ["Rivera, Elena"] for o in kept)
    art = next(o for o in kept if o["type"] == "article" and o["abstract"])
    assert art["title"] and art["date"] and art["url"].startswith("https://doi.org/") and art["doi"] and "  " not in art["abstract"]
    assert any(o["pmid"] and o["pmid"].isdigit() for o in kept)
    assert oa.parse_work({"id": "https://openalex.org/W1", "title": "x", "type": "erratum"}, {}) is None
    assert oa.parse_work({"id": "https://openalex.org/W1", "title": " ", "type": "article"}, {}) is None


def test_parse_authors_search_fixture():
    r = load("authors")["results"][0]
    assert oa.short_id(r["id"]).startswith("A") and r["display_name"] == "Elena Rivera"


def test_fetch_works_batches_authors_50_at_a_time_and_pages(monkeypatch):
    calls = []
    def fake_request(method, url, limiter, **kw):
        calls.append(kw["params"])
        page = kw["params"]["page"]
        results = [{"id": f"https://openalex.org/W{page}{len(calls)}", "title": f"Paper {page}-{len(calls)}", "type": "article",
                    "authorships": [{"author": {"id": "https://openalex.org/A1", "display_name": "A B"}}]}]
        return type("R", (), {"json": lambda self: {"meta": {"count": 150 if len(calls) == 1 else 1}, "results": results}})()
    monkeypatch.setattr(oa, "request", fake_request)
    from datetime import date
    watched = {f"A{i}": f"P{i}" for i in range(60)}
    out = oa.fetch_works(watched, date(2026, 10, 1))
    assert len(calls) >= 3 and calls[0]["filter"].count("|") == 49 and "from_publication_date:2026-10-01" in calls[0]["filter"]
    assert calls[1]["page"] == 2 and len(out) >= 2


# ------------------------------------------------------------------ watch module
def test_tidy_name_fixes_reporter_capitals_and_word_order():
    t = watch.tidy_name
    assert t("SPRINGER, SANDRA") == "Springer, Sandra" and t("SANDRA Ann SPRINGER") == "Springer, Sandra"
    assert t("Rivera, Elena") == "Rivera, Elena" and t("  Elena   Rivera ") == "Rivera, Elena" and t("Hartley") == "Hartley"
    assert t("McDonald, Emily") == "McDonald, Emily"                                                      # mixed case untouched


def test_add_stores_a_tidied_name(wsettings):
    people, added = watch.add(wsettings, {"name": "SPRINGER, SANDRA", "profile_id": 7094439})
    assert added and people[-1]["name"] == "Springer, Sandra"


def test_normalize_names():
    assert watch.normalize_name("Elena F Rivera") == ("Rivera", "Elena") and watch.normalize_name("Rivera, Elena F") == ("Rivera", "Elena")
    assert watch.normalize_name("Cher") == ("Cher", "") and watch.normalize_name("  ") == ("", "")
    assert watch.display_name("RIVERA", "ELENA") == "Rivera, Elena"


def test_name_ok_is_strict_about_last_name_and_first_initial():
    ok = watch.name_ok
    assert ok("Elena Rivera", "Rivera", "Elena") and ok("Elena F. Rivera", "Rivera", "Elena") and ok("É. Lindqvist", "Lindqvist", "Elana")
    assert not ok("Richard Kemper Leuchter", "Spellberg", "Brad") and not ok("Bob Smith", "Smith", "Elena")    # wrong person
    assert not ok("Brad Spellberg", "Spellberg", "Alan") and ok("Brad Spellberg", "Spellberg", "")


def test_pick_openalex_returns_all_split_records_at_the_institution_and_never_guesses():
    c = lambda i, name, inst, w=10: {"id": i, "name": name, "works": w, "institutions": inst}
    pick = watch.pick_openalex
    uab = ["University of Alabama at Birmingham"]
    got, why = pick([c("A1", "Elena Rivera", uab, 121), c("A2", "Elena Rivera", uab, 17), c("A3", "Elena Rivera", ["Duke University"])],
                    "UNIVERSITY OF ALABAMA AT BIRMINGHAM", "Rivera", "Elena")
    assert [x["id"] for x in got] == ["A1", "A2"] and "2 records" in why                                  # one person split in two
    assert pick([c("A1", "Brad Spellberg", ["University of Southern California"], 300),
                 c("A9", "Richard Leuchter", ["University of California, Los Angeles"], 37)], "UNIVERSITY OF SOUTHERN CALIFORNIA",
                "Spellberg", "Brad")[0][0]["id"] == "A1"                                                  # the 'California' namesake is rejected by name
    assert pick([c("A1", "Pat Lee", ["Duke University"])], "UNIVERSITY OF ALABAMA", "Lee", "Pat")[0] == []   # wrong institution
    assert pick([c("A1", "Logan Hartley", ["Duke"], 121), c("A2", "Logan Hartley", ["Stanford"], 1)], "", "Hartley", "Logan")[0][0]["id"] == "A1"   # dominant record
    assert pick([c("A1", "Pat Lee", ["X"], 30), c("A2", "Pat Lee", ["Y"], 25)], "", "Lee", "Pat")[0] == []   # no hint, comparable: ask a human
    assert pick([c("A1", "Pat Lee", ["X"], 0)], "", "Lee", "Pat")[0] == []                                     # no works


def test_multiple_openalex_ids_are_all_watched_and_merged_on_add(wsettings):
    people, _ = watch.add(wsettings, {"name": "Rivera, Elena", "openalex_id": ["A1", "A2"]})
    assert people[0]["openalex_id"] == ["A1", "A2"] and watch.oa_ids(people[0]) == ["A1", "A2"]
    people, added = watch.add(wsettings, {"name": "Rivera, Elena", "openalex_id": "A3"})
    assert added is False and people[0]["openalex_id"] == ["A1", "A2", "A3"]


@pytest.fixture
def wsettings(tmp_path, cfg2):
    from litsearch.core import config
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "interests.yaml").write_text(config.render_v2_yaml(
        {**cfg2, "people_to_watch": [{"name": "Rivera, Elena", "profile_id": 5}, {"name": "Lee, Pat"}]}))
    return settings_for(tmp_path)


def test_add_enriches_existing_entries_and_does_not_duplicate(wsettings):
    people, added = watch.add(wsettings, {"name": "Rivera, Elena", "openalex_id": "A5000000001"})
    assert added is False and people[0] == {"name": "Rivera, Elena", "profile_id": 5, "openalex_id": "A5000000001"}
    people, added = watch.add(wsettings, {"name": "Lee, Pat", "profile_id": "9"})
    assert added is False and people[1] == {"name": "Lee, Pat", "profile_id": 9}                      # name-only entry got pinned
    people, added = watch.add(wsettings, {"name": "Lee, Pat", "profile_id": 10})                       # a different Pat Lee (other id)
    assert added is True and len(people) == 3
    assert len(watch.load(wsettings)) == 3                                                              # persisted
    people, n = watch.remove(wsettings, profile_id=10)
    assert n == 1 and len(people) == 2


def test_resolve_missing_attaches_ids_and_reports_the_rest(wsettings, monkeypatch):
    monkeypatch.setattr(watch.openalex, "search_authors", lambda q, *a, **k: (
        [{"id": "A9", "name": "Elena Rivera", "works": 50, "institutions": ["University of Alabama at Birmingham"]},
         {"id": "A8", "name": "Elena F Rivera", "works": 4, "institutions": ["UAB"]},
         {"id": "A7", "name": "Elena F Rivera", "works": 3, "institutions": ["University of Alabama at Birmingham"]}] if "Rivera" in q
        else [{"id": "A1", "name": "Pat Lee", "works": 5, "institutions": ["X"]}, {"id": "A2", "name": "Pat Lee", "works": 9, "institutions": ["Y"]}]))
    monkeypatch.setattr(watch, "hint_for", lambda e, *a, **k: "UNIVERSITY OF ALABAMA AT BIRMINGHAM" if e["name"].startswith("Rivera") else "")
    rep = {r["name"]: r for r in watch.resolve_missing(wsettings)}
    assert rep["Rivera, Elena"]["status"] == "resolved" and rep["Lee, Pat"]["status"] == "unresolved"
    ppl = watch.load(wsettings)
    assert ppl[0]["openalex_id"] == ["A9", "A7"] and "openalex_id" not in ppl[1]                       # both UAB records ("UAB" alone has no token overlap)


# ------------------------------------------------------------------ people stream
def work(i, title, watched=("Rivera, Elena",), doi=None, pmid=None, abstract="An abstract about HIV and opioid use disorder care.", typ="article"):
    return {"id": f"W{i}", "title": title, "doi": doi, "pmid": pmid, "date": "2026-10-30", "type": typ, "journal": "A Journal",
            "authors": ["A One", "B Two", "C Three", "D Four"], "abstract": abstract, "url": f"https://doi.org/{doi}" if doi else f"https://openalex.org/W{i}",
            "watched": list(watched), "is_preprint": typ == "preprint"}


@pytest.fixture
def world(tmp_path, monkeypatch, cfg2):
    st = {"works": [], "asked": []}
    monkeypatch.setattr(oa, "fetch_works", lambda watched, since, limiter=None, max_records=300: (st["asked"].append((dict(watched), since)), list(st["works"]))[1])
    client = ScriptedClient(relevance=3, fit="")
    patch_client(monkeypatch, client)
    cfg = json.loads(json.dumps(cfg2))
    cfg["people_to_watch"] = [{"name": "Rivera, Elena", "openalex_id": "A1"}, {"name": "Lee, Pat", "openalex_id": "A2"}, {"name": "No, Id"}]
    return settings_for(tmp_path), cfg, st, client


def run(world, **kw):
    s, cfg, st, client = world
    return pipeline.run(s, cfg, RunOptions(**kw), stream="people")


def rows(world, stream="people", status=None):
    db = DB(world[0].db_path)
    r = [dict(x) for x in db.items(stream, statuses=(status,) if status else None)]
    db.close()
    return r


def test_new_works_are_shown_whatever_the_llm_score_and_carry_who_you_watch(world):
    world[2]["works"] = [work(1, "HIV prevention in hospitalized patients: a cohort", doi="10.1/a"), work(2, "Unrelated plant biology finding paper", watched=("Lee, Pat",))]
    assert run(world) == 0
    r = {x["source_id"]: x for x in rows(world)}
    assert set(r) == {"W1", "W2"} and all(x["status"] == "shown" for x in r.values())                 # relevance 3 would fail any threshold
    ss = json.loads(r["W1"]["stream_specific"])
    assert ss["watched"] == ["Rivera, Elena"] and ss["authors"] == "A One, B Two, C Three et al." and ss["doi"] == "10.1/a" and ss["kind"] == "paper"
    assert r["W1"]["url"] == "https://doi.org/10.1/a" and r["W1"]["takeaway"] == "funds X"
    assert world[2]["asked"][0][0] == {"A1": "Rivera, Elena", "A2": "Lee, Pat"}                          # only people with an OpenAlex id are asked about


def test_people_without_openalex_id_are_reported_not_silently_dropped(world):
    run(world)
    db = DB(world[0].db_path)
    notes = json.loads(db.conn.execute("SELECT notes FROM runs").fetchone()["notes"])
    assert any("1 watched person(s) have no OpenAlex id" in n and "No, Id" in n for n in notes)


def test_second_run_does_not_repeat_and_default_window_is_14_days(world):
    world[2]["works"] = [work(1, "HIV prevention in hospitalized patients: a cohort")]
    run(world); calls = world[3].calls
    run(world)
    assert world[3].calls == calls and len(rows(world)) == 1
    from datetime import date
    assert (date.today() - world[2]["asked"][0][1]).days in (14, 13, 15)


def test_paper_already_in_literature_feed_is_flagged_there_not_duplicated(world):
    s = world[0]
    db = DB(s.db_path)
    lit = db.record(Paper("pubmed", "HIV prevention in hospitalized patients: a cohort", "abstract", pmid="123", doi="10.1/a", relevance=8), "sent")
    scr = db.record(Paper("pubmed", "A screened-out paper about opioid use disorder trends", "abstract", pmid="456", relevance=2), "screened")
    db.close()
    world[2]["works"] = [work(1, "HIV prevention in hospitalized patients: a cohort", doi="10.1/a", pmid="123"),
                         work(2, "A screened-out paper about opioid use disorder trends", pmid="456")]
    run(world)
    assert [x["source_id"] for x in rows(world)] == ["W2"]                                              # sent paper: flagged in literature only
    ss = json.loads(next(x for x in rows(world, "literature") if x["id"] == lit)["stream_specific"])
    assert ss["watched_authors"] == ["Rivera, Elena"]
    run(world)
    ss = json.loads(next(x for x in rows(world, "literature") if x["id"] == lit)["stream_specific"])
    assert ss["watched_authors"] == ["Rivera, Elena"]                                                    # idempotent


def test_literature_stream_skips_papers_the_people_feed_already_shows(world, monkeypatch):
    world[2]["works"] = [work(1, "HIV prevention in hospitalized patients: a cohort", doi="10.1/a", pmid="123")]
    run(world)
    s, cfg, st, client = world
    lit = literature.LiteratureStream.__new__(literature.LiteratureStream)
    db = DB(s.db_path)
    lit.ctx = type("C", (), {"db": db})()
    same = Paper("pubmed", "HIV prevention in hospitalized patients: a cohort", "x", pmid="123", doi="10.1/a")
    other = Paper("pubmed", "A completely different paper about overdose deaths", "x", pmid="999")
    assert lit.is_new(same) is False and lit.is_new(other) is True


def test_no_watched_ids_means_no_openalex_call(world):
    world[1]["people_to_watch"] = [{"name": "No, Id"}]
    run(world)
    assert world[2]["asked"] == [] and rows(world) == []


def test_openalex_failure_fails_the_run(world, monkeypatch):
    def boom(*a, **k): raise RuntimeError("503")
    monkeypatch.setattr(oa, "fetch_works", boom)
    with pytest.raises(Exception, match="OpenAlex failed"):
        run(world)


def test_dry_run_prints_and_records_nothing(world, capsys):
    world[2]["works"] = [work(1, "HIV prevention in hospitalized patients: a cohort")]
    run(world, dry_run=True)
    assert "Dry run [people]" in capsys.readouterr().out and not world[0].db_path.exists()
