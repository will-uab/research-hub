import json
from datetime import datetime, timedelta, timezone

import pytest

from litsearch.core import config, learned, pipeline, tuning
from litsearch.core.context import RunOptions
from litsearch.core.db import DB
from litsearch.core.models import Item, Paper
from litsearch.streams import tuning as ts
from tests.helpers import settings_for

NOW = datetime(2026, 11, 5, tzinfo=timezone.utc)
EMPTY = learned.normalise({})


def r(i, sign, title="A paper", reason=None, days=1, **kw):
    return {"fid": i, "item_id": i, "sign": sign, "reason": reason, "ts": (NOW - timedelta(days=days)).isoformat(), "title": title,
            "journal": kw.get("journal", ""), "authors": kw.get("authors", []), "institutes": [], "mechanism": "", "sponsor": kw.get("sponsor", ""),
            "relevance": kw.get("relevance"), "stream": "literature", "takeaway": ""}


def cands(rated, cur=EMPTY, **kw):
    return {(c["kind"], c["value"]): c for c in tuning.candidate_rules(rated, cur, now=NOW, **kw)}


# ------------------------------------------------------------------ candidate rules: thresholds and direction
def test_a_single_tap_or_two_can_never_create_a_rule():
    rated = [r(1, -1, "Veterinary imaging classifier"), r(2, -1, "Veterinary radiology deep learning")]
    assert ("term", "veterinary") not in cands(rated)                                    # support 2 < 3
    rated.append(r(3, -1, "Veterinary pathology slides model"))
    c = cands(rated)[("term", "veterinary")]
    assert c["dislikes"] == 3 and c["likes"] == 0 and c["weight"] < 0 and c["evidence"] == [1, 2, 3]


def test_mixed_signals_do_not_make_a_rule_but_a_consistent_majority_does():
    mixed = [r(i, 1 if i % 2 else -1, "Benchmark of LLM extraction") for i in range(1, 7)]
    assert ("term", "benchmark") not in cands(mixed)                                      # 3 liked / 3 disliked: share 0.5
    mostly = [r(i, 1, "Benchmark of LLM extraction") for i in range(1, 6)] + [r(6, -1, "Benchmark of something")]
    c = cands(mostly)[("term", "benchmark")]
    assert c["weight"] > 0 and c["likes"] == 5 and c["share"] >= 0.7


def test_weight_scales_with_support_and_stays_within_bounds():
    assert tuning.suggest_weight(3, 0, 0.8) == 1.0 and tuning.suggest_weight(0, 5, 0.1) == -2.0
    assert tuning.suggest_weight(0, 10, 0.05) == -3.0 and tuning.suggest_weight(7, 1, 0.8) == 2.0


SECOND = ["imaging", "pathology", "radiology", "surgery", "dentistry", "nutrition"]


def vet(i, sign=-1, **kw):                                  # titles share only the word 'veterinary' (so no phrase swallows it)
    return r(i, sign, f"Veterinary {SECOND[i % 6]} paper", **kw)


def test_reasons_decide_what_a_dislike_says():
    base = lambda reason: [vet(i, reason=reason, journal="Vet Radiol") for i in range(1, 5)]
    c = cands(base("not_topic"))
    assert ("term", "veterinary") in c and ("journal", "Vet Radiol") in c                # a topic dislike teaches both
    c = cands(base("weak_study"))
    assert ("term", "veterinary") not in c and ("journal", "Vet Radiol") in c            # a quality dislike teaches the source, not the topic
    assert not cands(base("already_knew")) and not cands(base("too_basic"))               # "not new to me" teaches nothing
    assert ("term", "veterinary") in cands(base(None))                                    # no reason given: counts as a topic signal


def test_old_ratings_fade_but_still_count_toward_support():
    assert ("term", "veterinary") in cands([vet(i, days=180) for i in range(1, 4)], min_share=0.55)    # half a year old: counts at half strength
    assert ("term", "veterinary") not in cands([vet(i, days=900) for i in range(1, 4)], min_share=0.55)  # 2.5 years old: faded to ~nothing
    mix = [vet(i, days=900) for i in range(1, 4)] + [vet(i, 1, days=1) for i in range(4, 7)]
    assert cands(mix, min_share=0.55)[("term", "veterinary")]["weight"] > 0                          # recent likes outweigh ancient dislikes


def test_specific_phrase_replaces_its_own_unigram_and_existing_rules_are_not_reproposed():
    rated = [r(i, -1, "Veterinary imaging classifier") for i in range(1, 5)]
    c = cands(rated)
    assert ("term", "veterinary imaging") in c and ("term", "veterinary") not in c        # same evidence: keep the more specific phrase
    wide = [vet(i) for i in range(1, 5)]
    assert ("term", "veterinary") in cands(wide)                                          # different second words: the unigram has wider evidence, so it stays
    have = learned.normalise({"deprioritize": [{"kind": "term", "value": "veterinary imaging", "weight": -2}]})
    assert ("term", "veterinary imaging") not in cands(rated, have)


def test_stopwords_and_generic_terms_never_become_rules():
    rated = [r(i, -1, "A study of clinical outcomes in patients using large language models") for i in range(1, 6)]
    assert not [c for c in cands(rated) if c["kind"] == "term" and c["value"] in tuning.STOP]


# ------------------------------------------------------------------ follow suggestions and disagreements
def test_follow_candidates_need_two_liked_papers_or_a_tag_and_skip_people_already_followed():
    rated = [r(1, 1, authors=["Smith J", "Lee K"]), r(2, 1, authors=["Smith J"]), r(3, 1, authors=["Lee K"], reason="follow_person"),
             r(4, -1, authors=["Hate H"]), r(5, 1, authors=["Rivera, Elena", "Rivera E"]), r(6, 1, authors=["Rivera E"])]
    names = {c["name"]: c for c in tuning.follow_candidates(rated, ["Rivera, Elena"])}
    assert set(names) == {"Smith J", "Lee K"} and names["Smith J"]["likes"] == 2 and names["Lee K"]["tagged"] is True
    assert "Hate H" not in names and not [n for n in names if n.startswith("Rivera")]       # Rivera is followed: matched across name styles


def test_name_keys_match_across_pubmed_and_plain_styles():
    k = tuning._name_key
    assert k("Smith J") == k("Smith, John") == k("John Smith") == ("smith", "j")


def test_disagreements_surface_the_biggest_gaps_only():
    rated = [r(1, 1, "liked but scored low", relevance=2), r(2, -1, "disliked but scored high", relevance=9),
             r(3, 1, "liked and scored high", relevance=9), r(4, -1, "disliked and scored low", relevance=2), r(5, 1, "no score", relevance=None)]
    d = tuning.disagreements(rated)
    assert [x["item_id"] for x in d] == [1, 2] and d[0]["llm_score"] == 2


# ------------------------------------------------------------------ backtest + apply/undo
@pytest.fixture
def db_items(tmp_path, cfg2):
    s = settings_for(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "interests.yaml").write_text(config.render_v2_yaml({**cfg2, "learned": {"preferences": "old text", "prefer": [], "deprioritize": []}}))
    db = DB(s.db_path)
    ids = {n: db.record(Paper("pubmed", f"{t} paper number {n} for testing rules", "abs", pmid=str(n), relevance=8), "sent")
           for n, t in ((1, "Veterinary imaging"), (2, "Veterinary pathology"), (3, "Clinical NLP"), (4, "Veterinary radiology"))}
    return s, db, ids


def test_backtest_counts_matches_and_how_they_compare_with_your_taps(db_items):
    s, db, ids = db_items
    bt = tuning.backtest(db, {"kind": "term", "value": "veterinary", "weight": -2}, {ids[1]: -1, ids[2]: -1, ids[3]: 1})
    assert bt == {"days": 30, "matches": 3, "liked": 0, "disliked": 2, "unrated": 1}
    assert tuning.backtest(db, {"kind": "term", "value": "sepsis", "weight": -1}, {})["matches"] == 0


def test_accept_writes_the_yaml_and_undo_restores_it_exactly(db_items):
    s, db, ids = db_items
    pref = {"kind": "preferences", "payload": {"new": "Values validation studies."}, "why": ""}
    rule = {"kind": "rule", "payload": {"kind": "term", "value": "veterinary", "weight": -2}, "why": "disliked 4", "evidence": [1, 2]}
    b1, b2 = tuning.apply_proposal(s, pref), tuning.apply_proposal(s, rule)
    l = learned.normalise(config.load_interests(s.interests_path)["learned"])
    assert l["preferences"] == "Values validation studies." and l["deprioritize"][0]["value"] == "veterinary" and l["deprioritize"][0]["weight"] == -2
    assert l["deprioritize"][0]["evidence"] == [1, 2]
    flip = {"kind": "rule", "payload": {"kind": "term", "value": "Veterinary", "weight": 1}, "why": "changed my mind"}
    b3 = tuning.apply_proposal(s, flip)                                                  # the same rule in the other direction replaces it
    l = learned.normalise(config.load_interests(s.interests_path)["learned"])
    assert l["deprioritize"] == [] and l["prefer"][0]["weight"] == 1
    tuning.undo_proposal(s, {**flip, "before": b3})
    l = learned.normalise(config.load_interests(s.interests_path)["learned"])
    assert l["prefer"] == [] and l["deprioritize"][0]["weight"] == -2
    tuning.undo_proposal(s, {**rule, "before": b2}); tuning.undo_proposal(s, {**pref, "before": b1})
    l = learned.normalise(config.load_interests(s.interests_path)["learned"])
    assert l["preferences"] == "old text" and l["prefer"] == [] and l["deprioritize"] == []
    assert tuning.apply_proposal(s, {"kind": "follow", "payload": {"name": "X"}}) is None  # follow: nothing written


# ------------------------------------------------------------------ response validation: the model can't invent things
def test_parse_only_accepts_offered_ids_and_caps_weights_by_the_evidence():
    cands_ = {"c1": {"id": "c1", "kind": "term", "value": "veterinary", "weight": -2.0}, "c2": {"id": "c2", "kind": "journal", "value": "J", "weight": 1.0}}
    follows = {"f1": {"id": "f1", "name": "Smith J"}}
    txt = json.dumps({"preferences": "Skips veterinary work.", "notes": "n", "rules": [
        {"id": "c1", "weight": -9, "why": "w"}, {"id": "c2", "weight": -3, "why": "wrong direction"}, {"id": "c99", "weight": -1, "why": "invented"}, {"id": "c1", "weight": -1}],
        "follow": [{"id": "f1", "why": "liked"}, {"id": "f7", "why": "invented"}]})
    out = ts.parse_response(txt, cands_, follows, "old")
    assert [(x["cand"]["id"], x["weight"]) for x in out["rules"]] == [("c1", -2.0), ("c2", 1.0)]      # capped; wrong sign overridden; invented/duplicate ids dropped
    assert [f["cand"]["id"] for f in out["follow"]] == ["f1"] and out["preferences"] == "Skips veterinary work."
    assert ts.parse_response(json.dumps({"preferences": "old"}), {}, {}, "old")["preferences"] == ""     # unchanged text is not a proposal
    assert ts.parse_response(json.dumps({"preferences": "x" * 2000}), {}, {}, "old")["preferences"] == ""  # over-long paragraphs are ignored
    with pytest.raises(Exception):
        ts.parse_response("nope", {}, {}, "")


# ------------------------------------------------------------------ the stream
class TuneClient:
    def __init__(self, bad=0): self.calls, self.messages, self.bad, self.users = 0, self, bad, []
    def create(self, **kw):
        from tests.test_budget import _R
        self.calls += 1; self.users.append(kw["messages"][0]["content"])
        if self.calls <= self.bad:
            return _R("not json", i=2000, o=50)
        import re
        m = re.search(r'^(c\d+) \| term \| "veterinary"', kw["messages"][0]["content"], flags=re.M)          # choose by content, like a real model
        body = {"preferences": "Tends to skip veterinary and animal-health work and values LLM validation studies.", "notes": "Taps and model scores mostly agree.",
                "rules": [{"id": m.group(1), "weight": -3, "why": "Disliked every veterinary paper."}] if m else [], "follow": []}
        return _R(json.dumps(body), i=3000, o=500)


TUNE_ENV = {"LLM_MODEL": "claude-haiku-5-5", "LLM_MODEL_TUNING": "claude-sonnet-5-5", "PRICE_TUNING_INPUT_PER_MTOK": "2", "PRICE_TUNING_OUTPUT_PER_MTOK": "10"}


@pytest.fixture
def world(tmp_path, monkeypatch, cfg2):
    s = settings_for(tmp_path, **TUNE_ENV)
    (tmp_path / "config").mkdir()
    cfg = json.loads(json.dumps(cfg2)); cfg["learned"] = learned.normalise({}); cfg["people_to_watch"] = []
    (tmp_path / "config" / "interests.yaml").write_text(config.render_v2_yaml(cfg))
    db = DB(s.db_path)
    ids = [db.record(Paper("pubmed", f"Veterinary {SECOND[i % 6]} paper number {i}", "abs", pmid=str(i), relevance=8, domains=["llm_ai"],
                           authors=["Smith J"]), "sent") for i in range(1, 13)]
    db.close()
    client = TuneClient()
    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", lambda **k: client)
    return s, cfg, client, ids


def rate(world, n, sign=-1, reason=None):
    s, cfg, client, ids = world
    db = DB(s.db_path)
    for i in ids[:n]:
        db.set_rating(i, sign, reason)
    db.close()


def run(world, **kw):
    return pipeline.run(world[0], world[1], RunOptions(**kw), stream="tuning")


def proposals(world, status="pending"):
    db = DB(world[0].db_path)
    out = db.proposals((status,)); db.close(); return out


def test_nothing_happens_below_the_first_threshold_then_one_report_at_ten(world):
    rate(world, 9)
    run(world)
    assert world[2].calls == 0 and proposals(world) == []                                # 9 < 10
    rate(world, 10)
    run(world)
    assert world[2].calls == 1
    kinds = {p["kind"] for p in proposals(world)}
    assert {"preferences", "rule"} <= kinds
    run(world)
    assert world[2].calls == 1                                                           # no new ratings since the report: nothing to do


def test_next_report_needs_20_new_ratings_after_the_first(world):
    rate(world, 10); run(world)
    assert world[2].calls == 1
    s = world[0]
    db = DB(s.db_path)
    more = [db.record(Paper("pubmed", f"Another veterinary paper {i} with models", "abs", pmid=str(100 + i), relevance=8), "sent") for i in range(25)]
    for i in more[:19]:
        db.set_rating(i, -1)
    db.close()
    run(world)
    assert world[2].calls == 1                                                           # 19 < 20
    db = DB(s.db_path); db.set_rating(more[19], -1); db.close()
    run(world)
    assert world[2].calls == 2


def test_proposals_carry_evidence_backtest_and_are_pending_until_you_decide(world):
    rate(world, 12)
    run(world)
    rule = next(p for p in proposals(world) if p["kind"] == "rule" and p["payload"]["value"] == "veterinary")
    assert rule["payload"]["weight"] == -3.0 and rule["status"] == "pending" and len(rule["evidence"]) == 12
    assert rule["backtest"]["matches"] == 12 and rule["backtest"]["disliked"] == 12 and rule["why"].startswith("Disliked")
    l = learned.normalise(config.load_interests(world[0].interests_path)["learned"])
    assert l["deprioritize"] == [] and l["preferences"] == ""                            # NOTHING is written until accepted
    db = DB(world[0].db_path); rep = db.latest_report(); db.close()
    assert rep["n_ratings"] == 12 and rep["notes"].startswith("Taps and model scores") and rep["model"] == "claude-sonnet-5-5"


def test_prompt_contains_ratings_reasons_scores_candidates_and_disagreements(world):
    rate(world, 11, reason="not_topic")
    run(world)
    user = world[2].users[0]
    assert "reason: not_topic" in user and "model score then: 8" in user and "CANDIDATE RULES:" in user and "c1 | " in user and '"veterinary"' in user
    assert "WHERE THE MODEL'S SCORE AND THE TAP DISAGREE" in user and "disliked but model scored 8" in user
    assert "CURRENT LEARNED PARAGRAPH:\n(none yet)" in user


def test_force_runs_below_threshold_and_bad_json_is_retried_once(world, monkeypatch):
    rate(world, 3)
    run(world, force=True)
    assert world[2].calls == 1
    import anthropic
    c2 = TuneClient(bad=5); monkeypatch.setattr(anthropic, "Anthropic", lambda **k: c2)
    rate(world, 12)
    before = len(proposals(world)); run(world, force=True)
    assert c2.calls == 2 and len(proposals(world)) == before                               # two bad replies: skipped, nothing stored


def test_budget_guard_skips_the_call_and_stores_nothing(world):
    world[0].env["DAILY_BUDGET_USD"] = "0.0000001"
    rate(world, 12)
    run(world)
    assert world[2].calls == 0 and proposals(world) == []


def test_dry_run_proposes_from_evidence_alone_without_the_api(world, capsys):
    rate(world, 12)
    run(world, dry_run=True)
    out = capsys.readouterr().out
    assert "Dry run [tuning]" in out and "veterinary" in out and world[2].calls == 0
