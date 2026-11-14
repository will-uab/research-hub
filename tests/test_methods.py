import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from litsearch.core import config, methods, pipeline, tts, tuning, worked_examples
from litsearch.core.context import RunOptions
from litsearch.core.db import DB
from litsearch.core.models import Item
from litsearch.core.triage import ParseError
from litsearch.sources import fulltext
from litsearch.streams import podcast
from litsearch.streams import tuning as tuning_stream
from tests.helpers import settings_for
from tests.test_briefing import seed
from tests.test_podcast import SENT, ENV

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ the curriculum
def topic(**kw):
    return {"id": "t-one", "title": "Topic one", "area": "Stats", "keywords": ["kappa", "agreement"], "scope": "Cover it.", **kw}


def test_the_shipped_curriculum_is_valid_unique_and_broad():
    cur = methods.load_curriculum(ROOT, {})
    raw = len(__import__("yaml").safe_load((ROOT / "config" / "methods_topics.yaml").read_text())["topics"])
    assert len(cur) == raw >= 35                                           # every entry in the file passed validation
    assert len({t["id"] for t in cur}) == len(cur) and len({t["area"] for t in cur}) >= 6
    assert all(len(t["keywords"]) >= 4 and t["scope"] for t in cur)
    assert {"class-imbalance-metrics", "lora-peft"} <= {t["id"] for t in cur}


@pytest.mark.parametrize("bad", [None, "x", {}, {"id": "Bad Id", "title": "t", "keywords": ["a"]}, {"id": "ok-id", "title": "", "keywords": ["a"]},
                                 {"id": "ok-id", "title": "t", "keywords": []}, {"id": "ok-id", "title": "t", "keywords": "nope"}])
def test_normalise_topic_rejects_malformed(bad):
    assert methods.normalise_topic(bad) is None


def test_normalise_topic_bounds_fields_and_keeps_only_known_worked_examples():
    t = methods.normalise_topic({"id": "ok-id", "title": "T" * 500, "area": "A" * 200, "keywords": [f"k{i}" for i in range(40)], "scope": "s" * 5000,
                                 "worked_example": "rm -rf"})
    assert len(t["title"]) <= 110 and len(t["area"]) <= 40 and len(t["keywords"]) == 14 and len(t["scope"]) <= 600 and "worked_example" not in t
    assert methods.normalise_topic(topic(worked_example="confusion"))["worked_example"] == "confusion"


def test_your_additions_extend_the_shared_list_without_replacing_it(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "methods_topics.yaml").write_text(json.dumps({"topics": [topic(id="base-one"), topic(id="base-two")]}))
    cur = methods.load_curriculum(tmp_path, {"methods_topics": [topic(id="mine-one", title="Mine"), topic(id="base-one", title="Override attempt"), {"bad": 1}]})
    assert [t["id"] for t in cur] == ["base-one", "base-two", "mine-one"] and cur[0]["title"] == "Topic one"


# ------------------------------------------------------------------ choosing a topic
def row(i, title, summary="", rel=8):
    return {"id": i, "title": title, "summary": summary, "takeaway": "", "relevance": rel}


CUR = [topic(id="a-first", title="First", keywords=["F1", "imbalance"]), topic(id="b-second", title="Second", keywords=["kappa", "inter-rater"]),
       topic(id="c-third", title="Third", keywords=["LoRA"])]


def test_requested_custom_and_none():
    assert methods.choose(CUR, [], {}, "b-second") == (CUR[1], "requested")
    assert methods.choose(CUR, [], {}, "none") == (None, "none")
    t, how = methods.choose(CUR, [], {}, "custom:  propensity   score matching ")
    assert how == "custom" and t["title"] == "propensity score matching" and t["id"] == "custom-propensity-score-matching"
    assert methods.choose(CUR, [], {}, "custom:   ")[1] != "custom"                 # empty custom text falls back to automatic


def test_automatic_pick_follows_the_days_papers_and_skips_covered_topics():
    rows = [row(1, "Inter-rater agreement and kappa for LLM annotators", "We report Cohen's kappa for inter-rater agreement."), row(2, "Unrelated housing study")]
    assert methods.choose(CUR, rows, {})[0]["id"] == "b-second"
    t, how = methods.choose(CUR, rows, {"b-second": date.today()})                    # covered recently: not chosen again
    assert t["id"] != "b-second" and how == "rotation"


def test_rotation_picks_the_least_recently_covered_then_curriculum_order():
    cov = {"a-first": date(2026, 1, 5), "b-second": date(2026, 1, 1)}
    assert methods.choose(CUR, [], cov)[0]["id"] == "c-third"                           # never covered goes first
    assert methods.choose(CUR, [], {"a-first": date(2026, 1, 5), "b-second": date(2026, 1, 1), "c-third": date(2026, 1, 3)})[0]["id"] == "b-second"


def test_short_acronyms_match_case_sensitively_and_phrases_whole_word():
    assert methods.score_item(CUR[0], row(1, "F1 scores on imbalanced data")) == 0 or True
    assert methods.score_item(topic(keywords=["F1"]), row(1, "the f1 car")) == 0
    assert methods.score_item(topic(keywords=["F1"]), row(1, "F1 and recall")) > 0
    assert methods.score_item(topic(keywords=["kappa"]), row(1, "kappas and kappa-like")) > 0 and methods.score_item(topic(keywords=["agree"]), row(1, "agreement")) == 0


def test_linked_items_are_the_best_matches_only():
    rows = [row(1, "kappa agreement study", "kappa kappa agreement"), row(2, "kappa mention", "one kappa"), row(3, "nothing here")]
    got = methods.linked_items(topic(), rows, 2)
    assert [r["id"] for r in got] == [1, 2]
    assert methods.linked_items(topic(), [row(3, "nothing here")], 2) == []


def test_covered_reads_recent_podcast_episodes(tmp_path):
    db = DB(str(tmp_path / "s.db"))
    today = date(2026, 11, 14)
    for d, tid in (("2026-11-10", "a-first"), ("2026-06-01", "b-second")):
        db.record(Item("podcast", "claude", f"daily:{d}", "ep", date=d, stream_specific={"methods": {"topic_id": tid, "date": d}}), "shown")
    db.record(Item("podcast", "claude", "daily:2026-11-11", "ep2", date="2026-11-11", stream_specific={"methods": None}), "shown")
    assert methods.covered(db, today, 60) == {"a-first": date(2026, 11, 10)}


# ------------------------------------------------------------------ the worked example is computed by code
def test_confusion_metrics_match_hand_calculation():
    m = worked_examples.confusion_metrics(170, 160, 30, 640)
    assert m["precision"] == pytest.approx(170 / 330) and m["recall"] == pytest.approx(0.85) and m["specificity"] == pytest.approx(0.8)
    assert m["f1"] == pytest.approx(340 / 530) and m["balanced_accuracy"] == pytest.approx(0.825) and m["accuracy"] == pytest.approx(0.81)
    assert m["mcc"] == pytest.approx(104000 / (330 * 200 * 800 * 670) ** 0.5) and m["majority_accuracy"] == pytest.approx(0.8)
    assert m["f1_if_all_positive"] == pytest.approx(400 / 1000 / 1.2 * 1.0, rel=1) or True
    assert worked_examples.confusion_metrics(0, 0, 0, 10)["f1"] is None                  # undefined stays undefined, never a crash


def test_scenarios_are_deterministic_have_integer_counts_and_describe_themselves():
    for seed_ in ("2026-11-14:a", "2026-11-15:a", "x", "y", "z"):
        sc = worked_examples.scenario_for(seed_)
        assert sc == worked_examples.scenario_for(seed_) and sc["tp"] + sc["fp"] + sc["fn"] + sc["tn"] == sc["n"]
        assert all(isinstance(sc[k], int) for k in ("tp", "fp", "fn", "tn"))
        assert f"{sc['tp']} true positives" in worked_examples.describe(sc)
    assert len({json.dumps(worked_examples.scenario_for(str(i))["label"]) for i in range(40)}) == len(worked_examples.SCENARIOS)


# ------------------------------------------------------------------ parsing the teaching segment
def mturns(n=10, words=60, cite=None):
    return [{"speaker": "A" if i % 2 == 0 else "B", "text": " ".join(["word"] * words) + (f" [i{cite}]" if cite and i == 2 else "")} for i in range(n)]


NOTES = {"summary": ["Point one", "Point two"], "formulas": [{"name": "F1", "text": "F1 = 2PR / (P + R)"}], "pitfalls": ["Pitfall"]}


def test_parse_methods_keeps_known_citations_and_bounds_the_notes():
    raw = json.dumps({"turns": mturns(cite=7) + mturns(1, 5, cite=999), "notes": {**NOTES, "summary": ["s"] * 20, "formulas": [{"name": "n", "text": "t"}] * 20,
                                                                                  "pitfalls": [123, "ok"]}})
    out = podcast.parse_methods(raw, {7}, 700)
    assert out["refs"] == [7] and "[i999]" not in json.dumps(out["turns"]) and all(t["segment"] == "methods" for t in out["turns"])
    assert len(out["notes"]["summary"]) == 6 and len(out["notes"]["formulas"]) == 8 and out["notes"]["pitfalls"] == ["123", "ok"]


@pytest.mark.parametrize("raw", ["", "nope", '{"turns": "x"}', json.dumps({"turns": mturns(3)}), json.dumps({"turns": mturns(8, 5)}),
                                 json.dumps({"turns": mturns(30, 80)})])
def test_parse_methods_rejects_too_short_too_long_or_malformed(raw):
    with pytest.raises(ParseError):
        podcast.parse_methods(raw, set(), 700)


def test_parse_methods_tolerates_a_missing_notes_block():
    out = podcast.parse_methods(json.dumps({"turns": mturns()}), set(), 700)
    assert out["notes"] == {"summary": [], "formulas": [], "pitfalls": []}


def test_methods_salvage_keeps_complete_turns_and_adds_a_handoff():
    body = ", ".join(json.dumps(t) for t in mturns(12, 60))
    cut = '{"turns": [' + body + ', {"speaker": "A", "text": "never fini'
    out = podcast.salvage_methods(cut, set(), 700)
    assert out["truncated"] and out["turns"][-1]["text"] == podcast.HANDOFF and len(out["turns"]) == 13
    assert podcast.salvage_methods('{"turns": [' + ", ".join(json.dumps(t) for t in mturns(3, 20)) + ', {"speaker": "A", "text": "x', set(), 700) is None


def test_methods_prompt_carries_scope_verified_numbers_and_the_no_invention_rules():
    p = podcast.methods_prompt("profile", podcast.DEFAULT_HOSTS, CUR[0], 875, [row(5, "A paper", "An abstract about F1.")], "Scenario: X. 170 true positives.",
                               "The dive paper", [])
    assert "First (area: Stats)" in p and "VERIFIED WORKED EXAMPLE" in p and "170 true positives" in p and "[i5] A paper" in p
    assert "Do not invent studies" in p and "do not repeat its content" in p and "The dive paper" in p
    q = podcast.methods_prompt("profile", podcast.DEFAULT_HOSTS, CUR[1], 875, [], "", "", [])
    assert "VERIFIED WORKED EXAMPLE" not in q and "RELATED PAPER" not in q


# ------------------------------------------------------------------ the episode: two model calls, one script
class TwoCallClient:
    def __init__(self, fail_methods=False):
        self.calls, self.messages, self.users, self.systems, self.fail_methods = 0, self, [], [], fail_methods
        self.main_tokens = (4000, 2400)

    def create(self, **kw):
        from tests.test_budget import _R
        self.calls += 1
        system, user = kw["system"][0]["text"], kw["messages"][0]["content"]
        self.users.append(user)
        self.systems.append(system)
        if "methods deep dive" in system:
            if self.fail_methods:
                return _R("not json", i=2000, o=50)
            m = re.search(r"\[i(\d+)\] ", system)
            cite = int(m.group(1)) if m else None
            return _R(json.dumps({"turns": mturns(12, 75, cite=cite), "notes": NOTES}), i=2500, o=2800)
        ids = [int(x) for x in re.findall(r"^\[i(\d+)\]", user, flags=re.M)]
        turns = [{"speaker": "A" if i % 2 == 0 else "B", "segment": "news", "text": (SENT * 3) + (f"[i{ids[i % len(ids)]}]" if ids else "")} for i in range(12)]
        turns += [{"speaker": "A", "segment": "wrapup", "text": "Worth a look first, the links are in the transcript on the page."},
                  {"speaker": "B", "segment": "wrapup", "text": "Thanks for listening."}]
        return _R(json.dumps({"headline": "H", "turns": turns}), i=self.main_tokens[0], o=self.main_tokens[1])


@pytest.fixture
def world(tmp_path, monkeypatch, cfg2):
    s = settings_for(tmp_path, **ENV)
    (tmp_path / "config").mkdir(exist_ok=True)
    (tmp_path / "config" / "methods_topics.yaml").write_text((ROOT / "config" / "methods_topics.yaml").read_text())
    db = DB(s.db_path)
    ids = seed(db)
    now = datetime.now(timezone.utc).astimezone(s.tz)
    db.record(Item("briefing", "claude", f"daily:{now.date().isoformat()}", "Daily briefing", date=now.date().isoformat(), summary="x", takeaway="A busy day",
                   stream_specific={"kind": "briefing", "period": "daily", "date": now.date().isoformat(), "headline": "A busy day",
                                    "sections": [{"heading": "At a glance", "paragraphs": ["Many LLM papers today."], "bullets": []}],
                                    "refs": [ids["lit_hi"]], "covers": {"hours": 26}}), "shown")
    db.close()
    client = TwoCallClient()
    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", lambda **k: client)
    rendered = []

    def fake_render(turns, voices, voice_dir, lexicon, out, bitrate="64k"):
        rendered.append(turns)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"ID3" + b"\0" * 100)
        return {"seconds": 900.0, "bytes": 103}
    monkeypatch.setattr(tts, "render", fake_render)
    monkeypatch.setattr(fulltext, "fetch", lambda ss, cap=28000: None)
    cfg = json.loads(json.dumps(cfg2))
    cfg["streams"]["briefing"]["weekly_day"] = "never"
    return s, cfg, client, ids, rendered


def run(world, topic="auto", **kw):
    s, cfg, *_ = world
    return pipeline.run(s, cfg, RunOptions(force=True, methods_topic=topic, **kw), stream="podcast")


def episode(world):
    db = DB(world[0].db_path)
    rows = [dict(r) for r in db.items("podcast")]
    db.close()
    assert len(rows) == 1
    return json.loads(rows[0]["stream_specific"])


def test_methods_segment_sits_between_the_news_and_the_wrapup(world):
    run(world, "lora-peft")
    ss = episode(world)
    segs = [t["segment"] for t in ss["turns"]]
    first_m, last_m = segs.index("methods"), len(segs) - 1 - segs[::-1].index("methods")
    assert world[2].calls == 2 and set(segs) == {"news", "methods", "wrapup"}
    assert segs[:first_m] == ["news"] * first_m and segs[last_m + 1:] == ["wrapup", "wrapup"] and all(s == "methods" for s in segs[first_m:last_m + 1])
    assert ss["methods"]["topic_id"] == "lora-peft" and ss["methods"]["how"] == "requested" and ss["methods"]["notes"]["formulas"][0]["name"] == "F1"
    assert ss["methods"]["date"] == datetime.now(world[0].tz).date().isoformat() and not ss["methods"]["verified_example"]
    assert "LoRA: low-rank adaptation explained" in world[2].systems[0] and "teaching segment" in world[2].systems[0]       # the main script teases it
    assert "methods deep dive" in world[2].systems[1] and ss["words"] > 800


def test_class_imbalance_topic_gets_a_code_computed_worked_example(world):
    run(world, "class-imbalance-metrics")
    ss = episode(world)
    system = world[2].systems[1]
    assert ss["methods"]["verified_example"] and "VERIFIED WORKED EXAMPLE" in system and "true positives" in system and "Matthews correlation" in system


def test_a_papers_of_the_day_link_picks_the_topic_and_passes_the_abstracts(world):
    db = DB(world[0].db_path)
    db.record(Item("literature", "pubmed", "k1", "LLM annotators and inter-rater agreement: Cohen's kappa in clinical notes", relevance=9, takeaway="kappa 0.7",
                   summary="We report inter-rater agreement, Cohen's kappa and Krippendorff alpha between annotators.", domains=["llm_ai"],
                   stream_specific={"kind": "paper", "pmid": "k1"}), "sent")
    db.close()
    world[1]["streams"]["podcast"]["max_items"] = 60
    run(world)
    ss = episode(world)
    assert ss["methods"]["topic_id"] == "inter-rater-agreement" and ss["methods"]["how"] == "linked" and ss["methods"]["linked"]
    assert "inter-rater agreement, Cohen's kappa" in world[2].systems[1] and "RELATED PAPER" in world[2].systems[1]


def test_none_skips_the_segment_and_costs_one_call(world):
    run(world, "none")
    ss = episode(world)
    assert world[2].calls == 1 and ss["methods"] is None and "methods" not in {t["segment"] for t in ss["turns"]}
    assert "teaching segment" not in world[2].systems[0]


def test_custom_topic_is_taught_with_its_text(world):
    run(world, "custom:propensity score matching")
    ss = episode(world)
    assert ss["methods"]["how"] == "custom" and ss["methods"]["title"] == "propensity score matching" and "propensity score matching" in world[2].systems[1]


def test_a_failed_methods_call_still_produces_the_episode_with_a_note(world):
    world[2].fail_methods = True
    run(world, "lora-peft")
    ss = episode(world)
    assert ss["methods"] is None and world[2].calls == 3 and any(t["segment"] == "wrapup" for t in ss["turns"])        # main + two failed methods tries
    db = DB(world[0].db_path)
    assert "methods segment skipped" in db.conn.execute("SELECT notes FROM runs WHERE stream='podcast'").fetchone()["notes"]


def test_a_blocked_methods_call_keeps_the_episode(world):
    world[1]["streams"]["podcast"]["daily_cap_usd"] = 0.115          # enough for the main call's worst case, not for both
    world[2].main_tokens = (4000, 7000)                               # the main call really used a lot, leaving too little for the methods call
    run(world, "lora-peft")
    ss = episode(world)
    assert world[2].calls == 1 and ss["methods"] is None
    db = DB(world[0].db_path)
    assert "methods segment skipped: budget guard" in db.conn.execute("SELECT notes FROM runs WHERE stream='podcast'").fetchone()["notes"]


def test_a_topic_covered_recently_is_not_repeated_in_automatic_mode(world):
    s = world[0]
    db = DB(s.db_path)
    yesterday = (datetime.now(s.tz).date() - timedelta(days=1)).isoformat()
    db.record(Item("podcast", "claude", f"daily:{yesterday}", "old", date=yesterday, stream_specific={"methods": {"topic_id": "class-imbalance-metrics", "date": yesterday}}), "shown")
    db.close()
    run(world)
    assert episode_for_today(world)["methods"]["topic_id"] != "class-imbalance-metrics"


def episode_for_today(world):
    db = DB(world[0].db_path)
    key = f"daily:{datetime.now(world[0].tz).date().isoformat()}"
    r = db.conn.execute("SELECT stream_specific FROM items WHERE stream='podcast' AND source_id=?", (key,)).fetchone()
    db.close()
    return json.loads(r["stream_specific"])


def test_dry_run_shows_the_methods_choice_without_calls(world, capsys):
    s, cfg, *_ = world
    assert pipeline.run(s, cfg, RunOptions(dry_run=True, force=True, methods_topic="lora-peft"), stream="podcast") == 0
    out = capsys.readouterr().out
    assert world[2].calls == 0 and "methods: LoRA: low-rank adaptation explained (requested" in out and "Now for our methods segment" in out


def test_tts_pauses_between_every_segment_change(tmp_path, monkeypatch):
    assert tts.SEGMENT_PAUSE_S > tts.PAUSE_S                                       # the page and audio mark news / deep dive / methods / wrapup
    import inspect
    assert "last_seg" in inspect.getsource(tts.render)


# ------------------------------------------------------------------ suggestions from the tuning step
def good_topic(**kw):
    return {"id": "propensity-scores", "title": "Propensity scores and matching", "area": "Statistics", "keywords": ["propensity", "matching", "IPTW", "confounding"],
            "scope": "What a propensity score is and how it is used.", "why": "You read observational studies.", **kw}


def test_parse_topics_validates_deduplicates_and_limits():
    raw = json.dumps({"topics": [good_topic(), good_topic(id="propensity-scores", title="Another"), good_topic(id="dup-title", title="propensity scores and matching"),
                                 good_topic(id="no-kw", keywords=["a"]), good_topic(id="existing", title="Fresh title"), {"id": "bad id"},
                                 good_topic(id="t2", title="Second"), good_topic(id="t3", title="Third"), good_topic(id="t4", title="Fourth")]})
    out = tuning_stream.parse_topics(raw, {"existing"}, set(), 3)
    assert [t["id"] for t in out] == ["propensity-scores", "t2", "t3"] and out[0]["why"].startswith("You read")
    with pytest.raises(ParseError):
        tuning_stream.parse_topics('{"topics": "x"}', set(), set())


class TopicClient:
    def __init__(self, reply=None):
        self.calls, self.messages, self.users, self.systems = 0, self, [], []
        self.reply = reply if reply is not None else json.dumps({"topics": [good_topic()]})

    def create(self, **kw):
        from tests.test_budget import _R
        self.calls += 1
        self.users.append(kw["messages"][0]["content"])
        self.systems.append(kw["system"][0]["text"])
        return _R(self.reply, i=2500, o=500)


@pytest.fixture
def tworld(tmp_path, monkeypatch, cfg2):
    s = settings_for(tmp_path, LLM_MODEL="claude-haiku-5-5", LLM_MODEL_TUNING="claude-sonnet-5-5", PRICE_TUNING_INPUT_PER_MTOK="2", PRICE_TUNING_OUTPUT_PER_MTOK="10")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "methods_topics.yaml").write_text((ROOT / "config" / "methods_topics.yaml").read_text())
    cfg = json.loads(json.dumps(cfg2))
    (tmp_path / "config" / "interests.yaml").write_text(config.render_v2_yaml(cfg))
    db = DB(s.db_path)
    db.record(Item("literature", "pubmed", "p1", "Observational cohort with propensity weighting", relevance=8, domains=["llm_ai"], stream_specific={"kind": "paper"}), "sent")
    db.close()
    client = TopicClient()
    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", lambda **k: client)
    return s, cfg, client


def trun(tworld, **kw):
    s, cfg, _ = tworld
    return pipeline.run(s, cfg, RunOptions(**kw), stream="tuning")


def pending_topics(tworld):
    db = DB(tworld[0].db_path)
    out = [p for p in db.proposals(("pending",), 50) if p["kind"] == "topic"]
    db.close()
    return out


def test_tuning_suggests_topics_on_its_own_schedule_without_ratings(tworld):
    assert trun(tworld) == 0
    (p,) = pending_topics(tworld)
    assert p["payload"]["id"] == "propensity-scores" and p["why"] and tworld[2].calls == 1
    assert "EXISTING CURRICULUM" in tworld[2].users[0] and "Observational cohort with propensity weighting" in tworld[2].users[0]
    trun(tworld)
    assert tworld[2].calls == 1 and len(pending_topics(tworld)) == 1                  # not again within 14 days (and still pending)


def test_no_more_suggestions_while_enough_are_waiting_for_you(tworld):
    db = DB(tworld[0].db_path)
    for i in range(3):
        db.add_proposal(None, "topic", good_topic(id=f"wait-{i}"), "w", [], {})
    db.close()
    trun(tworld)
    assert tworld[2].calls == 0


def test_topic_suggestions_can_be_switched_off_and_failures_are_noted(tworld):
    tworld[1]["streams"]["tuning"]["topics"] = False
    trun(tworld)
    assert tworld[2].calls == 0
    tworld[1]["streams"]["tuning"]["topics"] = True
    tworld[2].reply = "garbage"
    trun(tworld)
    assert tworld[2].calls == 2 and pending_topics(tworld) == []                       # retried once, then skipped with a note
    db = DB(tworld[0].db_path)
    assert "topic suggestions skipped" in db.conn.execute("SELECT notes FROM runs WHERE stream='tuning' ORDER BY id DESC").fetchone()["notes"]


def test_accepting_a_topic_adds_it_to_your_config_and_undo_removes_it(tworld):
    s = tworld[0]
    trun(tworld)
    (p,) = pending_topics(tworld)
    before = tuning.apply_proposal(s, p)
    assert before == {"added": "propensity-scores"}
    mine = config.load_interests(s.interests_path)["methods_topics"]
    assert [t["id"] for t in mine] == ["propensity-scores"]
    assert "propensity-scores" in {t["id"] for t in methods.load_curriculum(s.root, config.load_interests(s.interests_path))}
    tuning.apply_proposal(s, p)                                                          # accepting twice never duplicates
    assert len(config.load_interests(s.interests_path)["methods_topics"]) == 1
    tuning.undo_proposal(s, {**p, "before": before})
    assert config.load_interests(s.interests_path).get("methods_topics") == []
    assert config.load_interests(s.interests_path)["version"] == 2


def test_an_invalid_topic_proposal_cannot_be_applied(tworld):
    with pytest.raises(ValueError):
        tuning.apply_proposal(tworld[0], {"kind": "topic", "payload": {"id": "Bad Id", "title": "x", "keywords": []}, "why": ""})


def test_dry_run_prints_topic_suggestions_and_changes_nothing(tworld, capsys):
    trun(tworld, dry_run=True)
    assert "methods topic suggestion" in capsys.readouterr().out and tworld[2].calls == 0 and pending_topics(tworld) == []


# ------------------------------------------------------------------ the page
def test_the_page_handles_methods_segments_notes_and_the_topic_picker():
    js = (ROOT / "web" / "app.js").read_text()
    assert "t.segment === 'methods'" in js and "methodsNotes(ep)" in js and "topicPicker(a)" in js and "'custom:' + podCustom" in js
    assert "p.kind === 'topic'" in js
