import json

import pytest
import yaml

from litsearch.core import config, learned, pagedata, pipeline
from litsearch.core.context import RunOptions
from litsearch.core.db import DB
from litsearch.core.models import Item, Paper
from litsearch.streams import literature
from tests.helpers import ScriptedClient, patch_client, settings_for

F = lambda **kw: learned.features(kw.pop("title", "A title"), kw.pop("summary", ""), kw)
L = lambda prefer=(), deprio=(), prefs="": learned.normalise({"preferences": prefs, "prefer": list(prefer), "deprioritize": list(deprio)})


# ------------------------------------------------------------------ rule matching and the bounded adjustment
def test_every_rule_kind_matches_the_right_field():
    f = F(title="Veterinary imaging classifiers: a benchmark of X-ray models", summary="sepsis appears only in the abstract", journal="Vet Radiol", authors=["Smith J", "Lee K"],
          institutes=["NIDA"], mechanism="R01", sponsor="University of Alabama at Birmingham")
    m = lambda **r: learned.rule_matches(r, f)
    assert m(kind="term", value="veterinary") and m(kind="term", value="benchmark*") and not m(kind="term", value="sepsis")   # titles only, not abstracts
    assert m(kind="journal", value="vet radiol") and not m(kind="journal", value="Vet")
    assert m(kind="author", value="smith j") and not m(kind="author", value="Smith")
    assert m(kind="institute", value="nida") and not m(kind="institute", value="NIAID")
    assert m(kind="mechanism", value="R*") and m(kind="mechanism", value="R01") and not m(kind="mechanism", value="K*")
    assert m(kind="sponsor", value="alabama") and not m(kind="sponsor", value="duke")
    assert not m(kind="nonsense", value="x") and not m(kind="term", value="")


def test_adjustment_sums_clips_and_explains():
    f = F(title="Veterinary benchmark of LLM agents")
    adj, why = learned.adjust(f, L(prefer=[{"kind": "term", "value": "benchmark*", "weight": 1}],
                                   deprio=[{"kind": "term", "value": "veterinary", "weight": -2}]))
    assert adj == -1 and why == ["↑ benchmark*", "↓ veterinary"]
    big = L(deprio=[{"kind": "term", "value": "veterinary", "weight": -3}, {"kind": "term", "value": "benchmark*", "weight": -3}])
    assert learned.adjust(f, big)[0] == learned.ADJ_MIN                                   # never below -3
    up = L(prefer=[{"kind": "term", "value": "veterinary", "weight": 2}, {"kind": "term", "value": "llm", "weight": 2}])
    assert learned.adjust(f, up)[0] == learned.ADJ_MAX                                    # never above +2
    assert learned.adjust(f, L()) == (0.0, [])
    assert learned.adjust(F(title="Unrelated"), L(deprio=[{"kind": "journal", "value": "x", "weight": -2}]))[1] == []
    assert learned.adjust(f, L(deprio=[{"kind": "journal", "value": "J", "weight": -1}]))[1] == []        # journal 'J' != this item


def test_reasons_label_non_term_kinds():
    adj, why = learned.adjust(F(title="t", journal="Lancet"), L(deprio=[{"kind": "journal", "value": "Lancet", "weight": -1}]))
    assert why == ["↓ journal: Lancet"]


# ------------------------------------------------------------------ the prompt paragraph
def test_paragraph_goes_before_output_rules_and_is_absent_when_empty():
    sp = "You triage.\n\nTASK\nstuff\n\nOUTPUT\nRespond with JSON."
    assert learned.insert_into_prompt(sp, L()) == sp
    out = learned.insert_into_prompt(sp, L(prefs="Values validation studies."))
    assert out.index("LEARNED READER TENDENCIES") < out.index("\n\nOUTPUT\n") and "the profile above takes precedence" in out
    assert out.endswith("Respond with JSON.") and "Values validation studies." in out
    assert learned.insert_into_prompt("no output section", L(prefs="x")).endswith("\n\nx") or "LEARNED" in learned.insert_into_prompt("no output section", L(prefs="x"))


# ------------------------------------------------------------------ config round trip
def test_learned_block_round_trips_through_the_yaml_and_defaults_to_empty(cfg2, tmp_path):
    assert set(config.load_interests(config.ROOT / "config" / "interests.example.yaml")["learned"]) == set(learned.EMPTY)
    f = tmp_path / "interests.yaml"
    f.write_text(config.render_v2_yaml({**cfg2, "learned": {"preferences": "Likes bias studies.\nDislikes imaging.", "prefer": [], "deprioritize": [{"kind": "term", "value": "veterinary", "weight": -2}]}}))
    got = config.load_interests(f)["learned"]
    assert got["preferences"].startswith("Likes bias studies.") and got["deprioritize"] == [{"kind": "term", "value": "veterinary", "weight": -2}]
    config.save_key(f, "learned", {"preferences": "New.", "prefer": [], "deprioritize": []})
    assert config.load_interests(f)["learned"]["preferences"] == "New." and config.load_interests(f)["version"] == 2
    assert "people_to_watch" in yaml.safe_load(f.read_text())                                           # other keys untouched


# ------------------------------------------------------------------ ranking effect in the streams
@pytest.fixture
def world(tmp_path, monkeypatch, cfg2):
    cfg = json.loads(json.dumps(cfg2))
    client = ScriptedClient(relevance=8, fit="ok", track="mine")
    patch_client(monkeypatch, client)
    mk = lambda i, title: Paper("pubmed", title, "LLM clinical note chart review abstract", pmid=str(i), journal="JAMIA", kw_score=10)
    monkeypatch.setattr(literature, "retrieve", lambda *a, **k: ([mk(1, "Large language models for chart review of notes one"), mk(2, "Veterinary large language models for chart review two"),
                                                                  mk(3, "Large language models for chart review of notes three")], [], {"pubmed": 3}))
    return settings_for(tmp_path), cfg, client


def test_deprioritised_candidates_rank_lower_but_are_still_triaged_and_stored(world):
    s, cfg, client = world
    cfg["learned"] = L(deprio=[{"kind": "term", "value": "veterinary", "weight": -3}], prefs="Dislikes veterinary work.")
    cfg["streams"]["literature"]["max_candidates"] = 60
    pipeline.run(s, cfg, RunOptions())
    assert client.calls == 1 and "Dislikes veterinary work." in client.last_system
    user = client.last_user
    assert user.index("Veterinary") > user.index("notes one") and user.index("Veterinary") > user.index("notes three")   # sorted last, not dropped
    assert "[p3]" in user                                                                                       # all three were still sent
    db = DB(s.db_path)
    assert db.conn.execute("SELECT COUNT(*) FROM items WHERE stream='literature' AND status='sent'").fetchone()[0] == 3


def test_the_learned_paragraph_reaches_the_model_prompt(world, monkeypatch):
    s, cfg, client = world
    cfg["learned"] = L(prefs="Values validation studies of LLMs on clinical text.")
    pipeline.run(s, cfg, RunOptions())
    sysmsg = client.last_system
    assert "Values validation studies of LLMs" in sysmsg and sysmsg.index("LEARNED READER") < sysmsg.index("\n\nOUTPUT\n")


def test_candidate_cap_prefers_boosted_items_over_deprioritised_ones(world):
    s, cfg, client = world
    cfg["streams"]["literature"]["max_candidates"] = 2
    cfg["learned"] = L(deprio=[{"kind": "term", "value": "veterinary", "weight": -3}])
    pipeline.run(s, cfg, RunOptions())
    assert "Veterinary" not in client.last_user and "[p2]" in client.last_user and "[p3]" not in client.last_user      # the cap kept 2 of 3: not the deprioritised one


# ------------------------------------------------------------------ page data carries the adjustment
def test_pagedata_emits_adj_and_reasons_only_for_items_that_match(tmp_path, db, cfg2):
    from datetime import datetime, timezone
    db.record(Paper("pubmed", "Veterinary LLM imaging paper for clinics", "abs", pmid="1", relevance=8, domains=["llm_ai"]), "sent")
    db.record(Paper("pubmed", "Clinical NLP benchmark on discharge summaries", "abs", pmid="2", relevance=8, domains=["llm_ai"]), "sent")
    cfg = {**cfg2, "learned": L(prefer=[{"kind": "term", "value": "benchmark*", "weight": 1}], deprio=[{"kind": "term", "value": "veterinary", "weight": -2}])}
    s = config.Settings(env={}, root=tmp_path)
    pagedata.regenerate(s, cfg, db, tmp_path / "site", datetime(2026, 11, 3, 15, tzinfo=timezone.utc))
    its = {i["title"][:9]: i for i in json.loads((tmp_path / "site" / "data" / "items-90d.json").read_text())["items"]}
    assert its["Veterinar"]["adj"] == -2 and its["Veterinar"]["adj_why"] == ["↓ veterinary"]
    assert its["Clinical "]["adj"] == 1 and its["Clinical "]["adj_why"] == ["↑ benchmark*"]
