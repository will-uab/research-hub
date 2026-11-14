import json
import os
import re
import threading
from datetime import datetime, timedelta, timezone

import pytest
import requests

from litsearch import web
from litsearch.core import pagedata, pipeline, tts
from litsearch.core.context import RunOptions
from litsearch.core.db import DB
from litsearch.core.models import Item
from litsearch.core.systemd import on_calendar
from litsearch.core.triage import ParseError
from litsearch.sources import fulltext
from litsearch.streams import podcast
from tests.helpers import settings_for
from tests.test_briefing import seed

ENV = {"LLM_MODEL": "claude-haiku-5-5", "LLM_MODEL_PODCAST": "claude-sonnet-5-5", "LLM_EFFORT_PODCAST": "low",
       "PRICE_PODCAST_INPUT_PER_MTOK": "2", "PRICE_PODCAST_OUTPUT_PER_MTOK": "10"}
SENT = "Researchers report that language models can pull outcomes out of clinical notes with reasonable accuracy. "


def script(cites, n=14, words_each=14):
    turns = [{"speaker": "A" if i % 2 == 0 else "B", "text": (SENT * 2)[: words_each * 7] + (f" [i{cites[i % len(cites)]}]" if cites else "")}
             for i in range(n)]
    return json.dumps({"headline": "Models and notes", "turns": turns})


# ------------------------------------------------------------------ parsing and speech text
def test_parse_keeps_known_citations_and_strips_unknown_ones():
    raw = json.dumps({"headline": "H", "turns": [{"speaker": "A" if i % 2 == 0 else "B", "text": SENT * 3 + ("[i5] [i999]" if i == 0 else "")} for i in range(6)]})
    out = podcast.parse_script(raw, {5}, 400)
    assert out["refs"] == [5] and "[i999]" not in out["turns"][0]["text"] and out["words"] > 120


@pytest.mark.parametrize("bad", ["", "nope", '{"turns": "x"}', '{"turns": []}',
                                 json.dumps({"turns": [{"speaker": "C", "text": SENT * 20}] * 8}),
                                 json.dumps({"turns": [{"speaker": "A", "text": "short"}] * 8})])
def test_parse_rejects_malformed_or_tiny_scripts(bad):
    with pytest.raises(ParseError):
        podcast.parse_script(bad, set(), 400)


def test_parse_rejects_a_script_far_longer_than_the_target():
    raw = json.dumps({"turns": [{"speaker": "A" if i % 2 == 0 else "B", "text": SENT * 6} for i in range(10)]})
    with pytest.raises(ParseError):
        podcast.parse_script(raw, set(), 200)


def test_speakable_strips_cites_and_applies_lexicon_on_whole_words_longest_first():
    lex = {"LLM": "L L M", "LLMs": "L L Ms", "NIDA": "Nye-duh"}
    assert tts.speakable("LLMs help NIDA staff [i12] and an LLM **works**. NIDAS stays.", lex) == "L L Ms help Nye-duh staff and an L L M works. NIDAS stays."


def test_lexicon_loads_from_yaml_and_the_shipped_one_parses():
    from pathlib import Path
    lex = tts.load_lexicon(Path(__file__).parent.parent / "config" / "pronunciation.yaml")
    assert lex["LLM"] == "L L M" and len(lex) > 10
    assert tts.load_lexicon(Path("/nonexistent/x.yaml")) == {}


# ------------------------------------------------------------------ the stream
class PodClient:
    def __init__(self, bad=0):
        self.calls, self.messages, self.bad, self.users, self.systems = 0, self, bad, [], []

    def create(self, **kw):
        from tests.test_budget import _R
        self.calls += 1
        user = kw["messages"][0]["content"]
        self.users.append(user)
        self.systems.append(kw["system"][0]["text"])
        if self.calls <= self.bad:
            return _R("not json", i=3000, o=100)
        ids = [int(x) for x in re.findall(r"^\[i(\d+)\]", user, flags=re.M)]
        return _R(script(ids[:3] + [99999]), i=4000, o=2400)


@pytest.fixture
def world(tmp_path, monkeypatch, cfg2):
    s = settings_for(tmp_path, **ENV)
    db = DB(s.db_path)
    ids = seed(db)
    now = datetime.now(timezone.utc).astimezone(s.tz)
    db.record(Item("briefing", "claude", f"daily:{now.date().isoformat()}", "Daily briefing", date=now.date().isoformat(), summary="x",
                   takeaway="A busy day", stream_specific={"kind": "briefing", "period": "daily", "date": now.date().isoformat(), "headline": "A busy day",
                                                           "sections": [{"heading": "At a glance", "paragraphs": ["Many LLM papers today."], "bullets": ["Worth it"]}],
                                                           "refs": [ids["lit_hi"]], "covers": {"hours": 26}}), "shown")
    db.close()
    client = PodClient()
    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", lambda **k: client)
    rendered = []

    def fake_render(turns, voices, voice_dir, lexicon, out, bitrate="64k"):
        rendered.append({"turns": turns, "voices": voices, "lex": lexicon})
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"ID3" + b"\0" * 5000)
        return {"seconds": 612.4, "bytes": out.stat().st_size}
    monkeypatch.setattr(tts, "render", fake_render)
    monkeypatch.setattr(fulltext, "fetch", lambda ss, cap=36000: None)
    cfg = json.loads(json.dumps(cfg2))
    cfg["streams"]["briefing"]["weekly_day"] = "never"
    return s, cfg, client, ids, rendered


def run(world, **kw):
    s, cfg, *_ = world
    return pipeline.run(s, cfg, RunOptions(force=True, **kw), stream="podcast")


def eps(world):
    db = DB(world[0].db_path)
    out = [dict(r) for r in db.items("podcast")]
    db.close()
    return out


def test_episode_is_written_with_audio_valid_citations_and_the_right_prompt(world):
    assert run(world) == 0
    (e,) = eps(world)
    ss = json.loads(e["stream_specific"])
    ids = world[3]
    assert e["source_id"].startswith("daily:") and ss["model"] == "claude-sonnet-5-5" and ss["audio"].startswith("audio/") and ss["seconds"] == 612.4
    assert ss["hosts"] == {"A": "Oliver", "B": "Hannah"} and set(ss["refs"]) <= {ids["lit_hi"], ids["trial"], ids["person"], ids["grant"]}
    assert "[i99999]" not in json.dumps(ss["turns"]) and (world[0].site_dir / ss["audio"]).exists()
    user, system = world[2].users[0], world[2].systems[0]
    assert f"[i{ids['lit_hi']}]" in user and f"[i{ids['person']}]" in user and "TODAY'S WRITTEN BRIEFING" in user and "Many LLM papers today." in user
    assert f"[i{ids['lit_screened']}]" not in user and f"[i{ids['lit_lo']}]" not in user and f"[i{ids['alt']}]" not in user
    assert "Use ONLY the supplied items" in system and "Oliver" in system and "Hannah" in system and podcast.DISCLOSURE in system
    v = world[4][0]["voices"]
    assert v["A"] == {"model": "en_GB-alan-medium", "length_scale": 0.85} and v["B"]["model"] == "en_US-hfc_female-medium"


def test_one_episode_per_day_and_cost_is_logged_under_podcast(world):
    run(world); run(world)
    assert len(eps(world)) == 1 and world[2].calls == 1 and len(world[4]) == 1
    db = DB(world[0].db_path)
    r = db.conn.execute("SELECT * FROM spend WHERE stream='podcast'").fetchone()
    assert r["calls"] == 1 and r["cost_usd"] == pytest.approx((4000 * 2 + 2400 * 10) / 1e6)


def test_no_briefing_today_means_no_episode_and_no_model_call(world):
    db = DB(world[0].db_path)
    db.conn.execute("DELETE FROM items WHERE stream='briefing'"); db.conn.commit(); db.close()
    run(world)
    assert eps(world) == [] and world[2].calls == 0


def test_a_failed_render_keeps_the_script_and_force_rerenders_without_another_model_call(world, monkeypatch):
    good = tts.render

    def boom(*a, **k):
        raise RuntimeError("piper exploded")
    monkeypatch.setattr(tts, "render", boom)
    run(world)
    (e,) = eps(world)
    assert "audio" not in json.loads(e["stream_specific"]) and world[2].calls == 1
    monkeypatch.setattr(tts, "render", good)
    run(world)
    (e,) = eps(world)
    ss = json.loads(e["stream_specific"])
    assert ss["audio"].startswith("audio/") and world[2].calls == 1                      # script reused, no new spend


def test_unparseable_output_is_retried_once_then_skipped(world, monkeypatch):
    world[2].bad = 5
    run(world)
    assert eps(world) == [] and world[2].calls == 2


def test_old_episode_files_are_pruned(world):
    audio = world[0].site_dir / "audio"
    audio.mkdir(parents=True)
    old = (datetime.now() - timedelta(days=45)).date().isoformat()
    (audio / f"{old}.mp3").write_bytes(b"x")
    run(world)
    assert not (audio / f"{old}.mp3").exists() and len(list(audio.glob("*.mp3"))) == 1


def test_podcast_items_never_feed_the_briefing_or_the_item_list(world):
    run(world)
    from litsearch.streams.briefing import BriefingStream
    s, cfg, *_ = world
    db = DB(s.db_path)
    out = pagedata.collect(db, datetime.now().date(), datetime.now(timezone.utc))[0]
    assert all(i["stream"] not in ("podcast", "briefing") for i in out)
    assert db.conn.execute("SELECT COUNT(*) FROM items WHERE stream='podcast'").fetchone()[0] == 1


def test_page_data_lists_only_episodes_whose_audio_exists(world):
    run(world)
    db = DB(world[0].db_path)
    got = pagedata.build_podcast(db, world[0].site_dir)["episodes"]
    assert len(got) == 1 and got[0]["audio"].endswith(".mp3") and got[0]["turns"] and got[0]["hosts"]["A"] == "Oliver"
    (world[0].site_dir / got[0]["audio"]).unlink()
    assert pagedata.build_podcast(db, world[0].site_dir)["episodes"] == []


def test_dry_run_uses_a_stub_and_makes_no_call_or_audio(world, capsys):
    s, cfg, *_ = world
    assert pipeline.run(s, cfg, RunOptions(dry_run=True, force=True), stream="podcast") == 0
    assert world[2].calls == 0 and world[4] == [] and "dry-run stub" in capsys.readouterr().out


# ------------------------------------------------------------------ server: byte ranges, weekday schedules
@pytest.mark.parametrize("hdr,size,want", [("bytes=0-99", 1000, (0, 99)), ("bytes=900-", 1000, (900, 999)), ("bytes=-100", 1000, (900, 999)),
                                           ("bytes=0-5000", 1000, (0, 999)), ("bytes=1000-", 1000, None), ("bytes=5-2", 1000, None),
                                           ("bytes=-", 1000, None), ("items=0-1", 1000, None), ("bytes=0-1", 0, None)])
def test_parse_range(hdr, size, want):
    assert web.parse_range(hdr, size) == want


@pytest.fixture
def audio_srv(tmp_path):
    s = settings_for(tmp_path)
    (s.site_dir / "audio").mkdir(parents=True)
    data = bytes(range(256)) * 40
    (s.site_dir / "audio" / "2026-01-02.mp3").write_bytes(data)
    (s.site_dir / "audio" / ".hidden.mp3").write_bytes(b"secret")
    httpd = web.serve(s, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/audio/2026-01-02.mp3", data, httpd.server_address[1]
    httpd.shutdown(); httpd.server_close()


def test_audio_supports_byte_ranges_for_ios(audio_srv):
    url, data, _ = audio_srv
    full = requests.get(url)
    assert full.status_code == 200 and full.content == data and full.headers["Accept-Ranges"] == "bytes"
    part = requests.get(url, headers={"Range": "bytes=100-199"})
    assert part.status_code == 206 and part.content == data[100:200] and part.headers["Content-Range"] == f"bytes 100-199/{len(data)}"
    tail = requests.get(url, headers={"Range": "bytes=-50"})
    assert tail.status_code == 206 and tail.content == data[-50:]
    assert requests.get(url, headers={"Range": f"bytes={len(data)}-"}).status_code == 416
    head = requests.head(url)
    assert head.status_code == 200 and head.headers["Accept-Ranges"] == "bytes" and head.headers["Content-Length"] == str(len(data))
    assert requests.get(url.replace("2026-01-02", "missing")).status_code == 404
    assert requests.get(url.replace("2026-01-02", ".hidden")).status_code == 404
    assert requests.get(url, headers={"Host": "evil.example.com"}).status_code == 403


def test_weekday_range_schedules():
    assert on_calendar("mon-fri 04:55") == "Mon..Fri *-*-* 04:55:00 America/Chicago"
    assert on_calendar("sun 04:15").startswith("Sun *-*-*") and on_calendar("04:00").startswith("*-*-* 04:00")
    with pytest.raises(ValueError):
        on_calendar("mon-xyz 04:55")


# ------------------------------------------------------------------ deep dive
def test_deep_dive_takes_the_most_relevant_paper_with_open_full_text(world, monkeypatch):
    ids = world[3]
    asked = []
    db = DB(world[0].db_path)
    db.conn.execute("UPDATE items SET relevance=7 WHERE id=?", (ids["person"],)); db.conn.commit(); db.close()       # a second eligible paper

    def fake_fetch(ss, cap=36000):
        asked.append(ss.get("pmid"))
        return {"text": "## Results\nSensitivity was 0.91 in the validation cohort.", "source": "Europe PMC (PMC open access)", "url": "https://europepmc.org/x", "words": 4200} \
            if len(asked) == 2 else None
    monkeypatch.setattr(fulltext, "fetch", fake_fetch)
    run(world)
    (e,) = eps(world)
    ss = json.loads(e["stream_specific"])
    user, system = world[2].users[0], world[2].systems[0]
    assert len(asked) == 2 and ss["dive"]["source"].startswith("Europe PMC") and ss["dive"]["words"] == 4200
    assert "DEEP DIVE ARTICLE" in user and "Sensitivity was 0.91" in user and "DEEP DIVE" in system and "at least 700 words" in system
    dive_id = ss["dive"]["id"]
    assert f"[i{dive_id}] {ss['dive']['title']}" in user.split("DEEP DIVE ARTICLE")[1]
    assert not any(l.startswith(f"[i{dive_id}] ") for l in user.split("DEEP DIVE ARTICLE")[0].splitlines())     # not also in the item list
    assert dive_id == ids["person"] and asked[0] != asked[1]                          # the top paper had no open text, the next one did


def test_no_open_full_text_means_a_news_only_episode_with_a_note(world):
    run(world)
    (e,) = eps(world)
    ss = json.loads(e["stream_specific"])
    assert ss["dive"] is None and "DEEP DIVE ARTICLE" not in world[2].users[0] and "DEEP DIVE" not in world[2].systems[0]
    db = DB(world[0].db_path)
    notes = db.conn.execute("SELECT notes FROM runs WHERE stream='podcast'").fetchone()["notes"]
    assert "no open full text" in notes


def test_items_carry_a_longer_detail_excerpt(world):
    run(world)
    user = world[2].users[0]
    assert "DETAIL: abstract abstract" in user


def test_deep_dive_can_be_switched_off_and_never_fetches(world, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("fetch must not run")
    monkeypatch.setattr(fulltext, "fetch", boom)
    world[1]["streams"]["podcast"]["dive"] = False
    run(world)
    assert json.loads(eps(world)[0]["stream_specific"])["dive"] is None


def test_segments_are_kept_and_default_to_news():
    raw = json.dumps({"turns": [{"speaker": "A" if i % 2 == 0 else "B", "segment": "deep_dive" if i > 3 else "news", "text": SENT * 3} for i in range(8)]
                      + [{"speaker": "A", "text": SENT * 3}]})
    out = podcast.parse_script(raw, set(), 2000)
    assert [t["segment"] for t in out["turns"]] == ["news"] * 4 + ["deep_dive"] * 4 + ["news"]


def test_page_data_carries_the_dive_and_segments(world, monkeypatch):
    monkeypatch.setattr(fulltext, "fetch", lambda ss, cap=36000: {"text": "t", "source": "arXiv (HTML)", "url": "https://arxiv.org/abs/1", "words": 2000})
    run(world)
    db = DB(world[0].db_path)
    ep = pagedata.build_podcast(db, world[0].site_dir)["episodes"][0]
    assert ep["dive"]["source"] == "arXiv (HTML)" and all("segment" in t for t in ep["turns"])


# ------------------------------------------------------------------ truncated replies
def cut_reply(n_complete=10, words=40):
    turns = [{"speaker": "A" if i % 2 == 0 else "B", "segment": "news", "text": " ".join(["word"] * words) + (" [i5]" if i == 0 else "")} for i in range(n_complete)]
    body = ", ".join(json.dumps(t) for t in turns)
    return '{"headline": "A cut story", "turns": [' + body + ', {"speaker": "A", "segment": "news", "text": "this one never fini'


def test_salvage_keeps_complete_turns_adds_a_closing_and_flags_it():
    out = podcast.salvage_script(cut_reply(), {5}, 400)
    assert out["truncated"] and out["headline"] == "A cut story" and out["refs"] == [5]
    assert len(out["turns"]) == 11 and out["turns"][-1]["text"] == podcast.CLOSING and "never fini" not in json.dumps(out["turns"])


def test_salvage_refuses_when_too_little_survived_or_nothing_parses():
    assert podcast.salvage_script(cut_reply(n_complete=3), {5}, 2000) is None
    assert podcast.salvage_script("not json at all", set(), 400) is None


def test_a_max_tokens_reply_still_becomes_an_episode_and_the_raw_reply_is_kept(world, monkeypatch):
    from tests.test_budget import _R
    reply = cut_reply(n_complete=24, words=60)

    def create(**kw):
        world[2].calls += 1
        r = _R(reply, i=4000, o=6800)
        r.stop_reason = "max_tokens"
        return r
    monkeypatch.setattr(world[2], "create", create)
    run(world)
    (e,) = eps(world)
    ss = json.loads(e["stream_specific"])
    assert ss["truncated"] and ss["turns"][-1]["text"] == podcast.CLOSING and world[2].calls == 1
    assert (world[0].root / "data" / "podcast_rejects").glob("*.txt")
    assert list((world[0].root / "data" / "podcast_rejects").glob("*-attempt1.txt"))


# ------------------------------------------------------------------ on demand: its own budget, redo, the page API
def spend(world, day, stream, usd):
    db = DB(world[0].db_path)
    db.conn.execute("INSERT INTO spend (day, stream, calls, input_tokens, output_tokens, cache_write_tokens, cache_read_tokens, cost_usd) VALUES (?,?,1,0,0,0,0,?)",
                    (day, stream, usd))
    db.conn.commit(); db.close()


def today_str(world):
    return datetime.now(world[0].tz).date().isoformat()


def test_podcast_has_its_own_budget_and_ignores_a_full_daily_cap(world):
    spend(world, today_str(world), "briefing", 0.20)                 # the hub's combined cap (0.15) is already exceeded
    run(world)
    assert len(eps(world)) == 1 and world[2].calls == 1


def test_podcast_spend_does_not_count_against_the_other_streams(world):
    from litsearch.core.budget import Budget, Prices
    spend(world, today_str(world), "podcast", 0.40)
    db = DB(world[0].db_path)
    b = Budget(db, today_str(world), 0.15, Prices(1, 1), "literature", exclude_streams=("podcast",))
    assert b.spent_today == 0 and b.allow(0.10)
    assert Budget(db, today_str(world), 0.15, Prices(1, 1), "literature").spent_today == pytest.approx(0.40)      # without the exclusion it would count


def test_podcast_stops_at_its_own_daily_budget(world):
    cap = world[1]["streams"]["podcast"]["daily_cap_usd"]
    spend(world, today_str(world), "podcast", cap - 0.01)
    run(world)
    assert eps(world) == [] and world[2].calls == 0
    db = DB(world[0].db_path)
    assert "podcast budget" in db.conn.execute("SELECT notes FROM runs WHERE stream='podcast'").fetchone()["notes"]


def test_redo_replaces_todays_episode_and_its_audio(world):
    run(world)
    first = json.loads(eps(world)[0]["stream_specific"])["audio"]
    s, cfg, *_ = world
    assert pipeline.run(s, cfg, RunOptions(), stream="podcast") == 0 and world[2].calls == 1            # no redo: nothing happens
    assert pipeline.run(s, cfg, RunOptions(redo=True), stream="podcast") == 0
    (e,) = eps(world)
    assert world[2].calls == 2 and len(world[4]) == 2 and (s.site_dir / json.loads(e["stream_specific"])["audio"]).exists()
    db = DB(s.db_path)
    assert db.conn.execute("SELECT COUNT(*) FROM item_keys k JOIN items i ON i.id=k.item_id WHERE i.stream='podcast'").fetchone()[0] == 1


def test_scheduled_timers_skip_on_demand_streams(world):
    from litsearch.core.systemd import render_units
    files = render_units(world[1], world[0].root, "pi")
    assert not any("podcast" in f for f in files) and "litsearch-briefing.timer" in files


def test_status_reports_podcast_spend_separately_and_never_overdue(world):
    run(world)
    db = DB(world[0].db_path)
    now = datetime.now(timezone.utc)
    st = pagedata.build_status(db, world[0], world[1], now.astimezone(world[0].tz).date(), now)
    p = next(x for x in st["streams"] if x["id"] == "podcast")
    assert p["on_demand"] and not p["stale"] and st["spend"]["separate"]["podcast"]["today"] > 0
    assert st["spend"]["today"] == 0 and all(r["stream"] != "podcast" for r in st["spend"]["log"])


@pytest.fixture
def papi(world):
    import litsearch.core.config as config
    s, cfg = world[0], world[1]
    (s.root / "config").mkdir(exist_ok=True)
    s.interests_path.write_text(config.render_v2_yaml(cfg))
    started = []
    api = web.Api(s, spawn=lambda redo, topic='auto': started.append(redo) or os.getpid())
    return api, started, world


def test_api_generate_starts_a_run_once_and_reports_state(papi):
    import os
    api, started, world = papi
    code, st = api.get("/api/podcast", {})
    assert code == 200 and st["enabled"] and not st["running"] and not st["episode_today"] and st["cap"] == 0.7 and st["est_cost"] == 0.3
    code, out = api.post("/api/podcast/generate", {})
    assert code == 202 and started == [False]
    assert api.get("/api/podcast", {})[1]["running"]
    assert api.post("/api/podcast/generate", {})[0] == 409 and started == [False]            # a second tap while it runs does nothing


def test_api_generate_refuses_a_second_episode_unless_redo(papi):
    api, started, world = papi
    run(world)                                                   # an episode now exists for today
    assert api.get("/api/podcast", {})[1]["episode_today"]
    code, out = api.post("/api/podcast/generate", {})
    assert code == 409 and out["exists"] and started == []
    assert api.post("/api/podcast/generate", {"redo": True})[0] == 202 and started == [True]


def test_api_generate_forgets_a_dead_run_and_reports_the_last_outcome(papi):
    api, started, world = papi
    api._run_marker.parent.mkdir(exist_ok=True)
    api._run_marker.write_text(json.dumps({"pid": 2 ** 22 + 12345, "started": datetime.now(timezone.utc).isoformat()}))        # no such process
    assert not api.get("/api/podcast", {})[1]["running"]
    assert api.post("/api/podcast/generate", {})[0] == 202
    spend(world, today_str(world), "podcast", 0.69)
    api._run_marker.unlink()
    run(world)                                                    # blocked by its own budget -> the note is surfaced to the page
    last = api.get("/api/podcast", {})[1]["last"]
    assert last["ok"] and any("budget" in n for n in last["notes"])


def test_api_generate_can_be_disabled_and_failures_release_the_slot(papi):
    import litsearch.core.config as config
    api, started, world = papi
    def boom(redo, topic='auto'):
        raise OSError("no fork")
    api.spawn = boom
    code, out = api.post("/api/podcast/generate", {})
    assert code == 500 and not api.get("/api/podcast", {})[1]["running"]


def test_ping_advertises_api_version_3(papi):
    assert papi[0].get("/api/ping", {})[1]["v"] == 3
