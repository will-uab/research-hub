import json

import pytest

from litsearch.core.budget import Budget, Prices
from litsearch.core.models import Paper
from litsearch.core.triage import ParseError, Triager, format_batch, parse_response, truncate_words
from litsearch.streams.literature import TRACKS
from tests.helpers import DOMAINS, make_spec
from tests.test_budget import FakeClient, _good, _papers

DOMS = tuple(DOMAINS)


def parse(text, ids=None):
    return parse_response(text, ids or IDS, TRACKS, DOMS)

IDS = ["p1", "p2"]
ITEM = {"id": "p1", "relevance": 7.4, "track": "Mine", "one_sentence_takeaway": " ok. ", "why_relevant": "w"}


def test_parse_plain_array():
    out = parse(json.dumps([ITEM]))
    assert out["p1"] == {"relevance": 7.4, "track": "mine", "takeaway": "ok.", "fit_note": "w", "domains": []}


def test_parse_fenced_and_wrapped_in_prose():
    txt = "Here you go:\n```json\n" + json.dumps([ITEM]) + "\n```"
    assert "p1" in parse(txt)


def test_parse_object_wrapping_list():
    assert "p1" in parse(json.dumps({"results": [ITEM]}))


def test_unknown_ids_invalid_items_ignored_and_clamped():
    items = [dict(ITEM, id="p9"), dict(ITEM, id="p2", relevance=15), dict(ITEM, id="p1", track="maybe"),
             "junk", dict(ITEM, id="p1", relevance="n/a")]
    out = parse(json.dumps(items))
    assert list(out) == ["p2"] and out["p2"]["relevance"] == 10.0


@pytest.mark.parametrize("bad", ["", "not json at all", '{"a": 1}', "[1, 2", "```"])
def test_malformed_raises(bad):
    with pytest.raises(ParseError):
        parse(bad)


def test_truncate_words():
    assert truncate_words("a b c d", 2) == "a b …"
    assert truncate_words("a b", 5) == "a b"


def test_format_batch_uses_short_ids_and_truncates():
    p = Paper("pubmed", "T", " ".join(["w"] * 500), journal="JAMA", is_major=True)
    s = format_batch([p], ["p1"], make_spec())
    assert s.startswith("Papers:") and "[p1] T" in s and "major-journal: yes" in s
    assert s.count(" w") < 300


def test_system_prompt_includes_feedback_only_when_given():
    assert "Recent reader feedback" not in make_spec().system_prompt
    s = make_spec([(1, "Liked title"), (-1, "Bad title")]).system_prompt
    assert "Liked:\n- Liked title" in s and "Disliked:\n- Bad title" in s


def _triager(db, texts):
    prices = Prices.from_env({"PRICE_INPUT_PER_MTOK": "0.1", "PRICE_OUTPUT_PER_MTOK": "0.5"})
    client = FakeClient(texts)
    return Triager(client, "m", "", Budget(db, "d", 1.0, prices), make_spec(), 15), client


def test_malformed_then_valid_retries_once(db):
    t, client = _triager(db, ["garbage", _good(["p1", "p2"])])
    res = t.run(_papers(2))
    assert client.calls == 2 and len(res.scored) == 2 and not res.parse_failed
    assert res.scored[0].llm_scored and res.scored[0].track == "mine"


def test_malformed_twice_skips_without_third_call(db):
    t, client = _triager(db, ["garbage", "still {bad", _good(["p1"])])
    res = t.run(_papers(2))
    assert client.calls == 2 and not res.scored and len(res.parse_failed) == 2
    assert any("not valid JSON" in e for e in res.errors)


def test_partial_output_retries_only_missing(db):
    t, client = _triager(db, [_good(["p1"]), _good(["p2"])])    # retry asks only for p2, keeping its original id
    res = t.run(_papers(2))
    assert client.calls == 2 and len(res.scored) == 2


def test_domains_parsed_filtered_and_ordered():
    item = dict(ITEM, domains=["hiv_addiction", "bogus", "llm_ai"])
    assert parse(json.dumps([item]))["p1"]["domains"] == ["llm_ai", "hiv_addiction"]    # config order, unknown dropped
    assert parse(json.dumps([dict(ITEM, domains="llm_ai")]))["p1"]["domains"] == []      # wrong type tolerated


def test_streams_without_tracks_need_no_track_field():
    item = {"id": "p1", "relevance": 5, "one_sentence_takeaway": "t", "why_relevant": "fit"}
    out = parse_response(json.dumps([item]), ["p1"], tracks=None, domain_ids=DOMS)
    assert out["p1"]["track"] is None and out["p1"]["fit_note"] == "fit"


def test_llm_domains_override_keyword_tags(db):
    import json as _j
    t, client = _triager(db, [_j.dumps([{"id": "p1", "relevance": 8, "track": "mine", "domains": ["hiv_addiction"],
                                         "one_sentence_takeaway": "t", "why_relevant": "w"}])])
    ps = _papers(1)
    ps[0].domains = ["llm_ai"]
    res = t.run(ps)
    assert res.scored[0].domains == ["hiv_addiction"]
    t2, _ = _triager(db, [_good(["p1"])])           # _good has no domains -> keyword tags kept
    ps2 = _papers(1)
    ps2[0].domains = ["llm_ai"]
    assert t2.run(ps2).scored[0].domains == ["llm_ai"]
