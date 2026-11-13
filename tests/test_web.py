import json
import threading

import pytest
import requests

from litsearch import web
from litsearch.core import config
from litsearch.core.db import DB
from litsearch.core.models import Item
from tests.helpers import settings_for

HDR = {"Content-Type": "application/json", "X-LH": "1"}


@pytest.fixture
def srv(tmp_path, cfg2):
    s = settings_for(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "interests.yaml").write_text(config.render_v2_yaml({**cfg2, "people_to_watch": [{"name": "Rivera, Elena", "profile_id": 5}]}))
    s.site_dir.mkdir()
    (s.site_dir / "index.html").write_text("<html>hub</html>")
    (s.site_dir / ".lock").write_text("secret")
    (tmp_path / "outside.txt").write_text("nope")
    db = DB(s.db_path)
    a = db.record(Item("grants", "g", "A1", "First item title"), "shown")
    b = db.record(Item("grants", "g", "A2", "Second item title"), "shown")
    db.close()
    calls = []
    def fake_search(q):
        calls.append(q)
        if not q.strip():
            raise ValueError("enter a name")
        return {"query": q, "reporter": [{"profile_id": 9, "name": "Pat Lee", "awards": 3, "institutions": ["X UNIV"]}], "openalex": [], "errors": []}
    httpd = web.serve(s, "127.0.0.1", 0)
    httpd.RequestHandlerClass = web.make_handler(s.site_dir, web.Api(s, fake_search))
    t = threading.Thread(target=httpd.serve_forever, daemon=True); t.start()
    yield {"url": f"http://127.0.0.1:{httpd.server_address[1]}", "ids": (a, b), "settings": s, "calls": calls}
    httpd.shutdown(); httpd.server_close()


def post(srv, path, body, headers=HDR):
    return requests.post(srv["url"] + path, data=json.dumps(body), headers=headers, timeout=10)


def test_static_pages_served_and_dotfiles_and_traversal_blocked(srv):
    assert requests.get(srv["url"] + "/index.html", timeout=10).text == "<html>hub</html>"
    assert requests.get(srv["url"] + "/.lock", timeout=10).status_code == 404
    assert requests.get(srv["url"] + "/%2e%2e/outside.txt", timeout=10).status_code in (400, 404)
    r = requests.get(srv["url"] + "/api/ping", timeout=10)
    assert r.json() == {"ok": True, "v": 3} and r.headers["Cache-Control"] == "no-store"


def test_rate_save_read_roundtrip_via_state(srv):
    a, b = srv["ids"]
    assert post(srv, "/api/rate", {"id": a, "sign": 1}).json()["sign"] == 1
    assert post(srv, "/api/rate", {"id": b, "sign": -1}).status_code == 200
    assert post(srv, "/api/mark", {"id": a, "saved": True}).json() == {"ok": True, "id": a, "saved": True}
    assert post(srv, "/api/mark", {"id": a, "read": True}).status_code == 200
    st = requests.get(srv["url"] + "/api/state", timeout=10).json()
    assert st["ratings"] == {str(a): 1, str(b): -1} and st["saved"] == [a] and st["read"] == [a]
    post(srv, "/api/rate", {"id": a, "sign": 0}); post(srv, "/api/mark", {"id": a, "saved": False})
    st = requests.get(srv["url"] + "/api/state", timeout=10).json()
    assert st["ratings"] == {str(b): -1} and st["saved"] == []
    db = DB(srv["settings"].db_path)
    assert db.recent_feedback(10, "grants") == [(-1, "Second item title")]               # ratings reach the triage prompt
    db.close()


@pytest.mark.parametrize("body,code", [({"sign": 1}, 400), ({"id": "x", "sign": 1}, 400), ({"id": 9999, "sign": 1}, 404),
                                       ({"id": 1, "sign": 5}, 400), ({"id": 1, "sign": "1"}, 400)])
def test_rate_validation(srv, body, code):
    assert post(srv, "/api/rate", body).status_code == code


def test_mark_validation(srv):
    a = srv["ids"][0]
    assert post(srv, "/api/mark", {"id": a}).status_code == 400
    assert post(srv, "/api/mark", {"id": a, "saved": "yes"}).status_code == 400


def test_csrf_protections(srv):
    a = srv["ids"][0]
    body = {"id": a, "sign": 1}
    assert post(srv, "/api/rate", body, headers={"Content-Type": "application/json"}).status_code == 403          # no X-LH
    assert post(srv, "/api/rate", body, headers={"Content-Type": "text/plain", "X-LH": "1"}).status_code == 403   # form-style post
    assert post(srv, "/api/rate", body, headers={**HDR, "Origin": "https://evil.example"}).status_code == 403     # other website
    assert post(srv, "/api/rate", body, headers={**HDR, "Origin": "https://hub.example-tailnet.ts.net"}).status_code == 200  # your tailnet
    assert post(srv, "/api/rate", body, headers={**HDR, "Origin": srv["url"]}).status_code == 200                  # same origin
    assert requests.get(srv["url"] + "/api/state", headers={"Host": "evil.example"}, timeout=10).status_code == 403  # DNS rebinding
    assert post(srv, "/api/rate", body, headers={**HDR, "Host": "evil.example"}).status_code == 403
    assert post(srv, "/api/rate", "x" * 9000).status_code in (413, 403, 400)
    big = requests.post(srv["url"] + "/api/rate", data=b"{" + b" " * 9000 + b"}", headers=HDR, timeout=10)
    assert big.status_code == 413
    assert requests.post(srv["url"] + "/api/rate", data=b"not json", headers=HDR, timeout=10).status_code == 400
    assert requests.post(srv["url"] + "/index.html", data=b"{}", headers=HDR, timeout=10).status_code == 404


def test_watch_search_add_remove_persist_to_yaml(srv):
    r = requests.get(srv["url"] + "/api/watch/search", params={"q": "Lee, Pat"}, timeout=10).json()
    assert r["reporter"][0]["profile_id"] == 9 and srv["calls"] == ["Lee, Pat"]
    assert requests.get(srv["url"] + "/api/watch/search", params={"q": " "}, timeout=10).status_code == 400
    assert requests.get(srv["url"] + "/api/watch", timeout=10).json()["people"] == [{"name": "Rivera, Elena", "profile_id": 5}]
    r = post(srv, "/api/watch/add", {"name": "Lee, Pat", "profile_id": 9, "openalex_id": "A1"}).json()
    assert r["added"] is True and [p["name"] for p in r["people"]] == ["Rivera, Elena", "Lee, Pat"]
    assert post(srv, "/api/watch/add", {"name": "Lee, Pat", "profile_id": 9}).json()["added"] is False       # idempotent
    text = (srv["settings"].root / "config" / "interests.yaml").read_text()
    assert "Lee, Pat" in text and config.load_interests(srv["settings"].interests_path)["version"] == 2       # still a valid v2 file
    assert post(srv, "/api/watch/add", {"profile_id": 3}).status_code == 400                                  # name required
    r = post(srv, "/api/watch/remove", {"name": "lee, pat"}).json()
    assert r["removed"] == 1 and [p["name"] for p in r["people"]] == ["Rivera, Elena"]
    assert post(srv, "/api/watch/remove", {"name": "nobody"}).json()["removed"] == 0


def test_host_allowlist():
    assert web.host_allowed("localhost:8081") and web.host_allowed("127.0.0.1") and web.host_allowed("hub.example-tailnet.ts.net")
    assert not web.host_allowed("evil.example") and not web.host_allowed("") and not web.host_allowed("ts.net.evil.com")
    assert web.host_allowed("my-pi.lan", ("my-pi.lan",))


def test_ack_roundtrip_validation_and_state(srv):
    assert requests.get(srv["url"] + "/api/state", timeout=10).json()["acked"] == []
    assert post(srv, "/api/ack", {"key": "daily:2026-11-05"}).json() == {"ok": True, "key": "daily:2026-11-05"}
    assert post(srv, "/api/ack", {"key": "weekly:2026-W46"}).status_code == 200
    assert sorted(requests.get(srv["url"] + "/api/state", timeout=10).json()["acked"]) == ["daily:2026-11-05", "weekly:2026-W46"]
    for bad in ({}, {"key": "monthly:2026-11"}, {"key": "daily:tomorrow"}, {"key": "daily:2026-11-05; DROP"}):
        assert post(srv, "/api/ack", bad).status_code == 400
    assert post(srv, "/api/ack", {"key": "daily:2026-11-05"}, headers={"Content-Type": "application/json"}).status_code == 403   # CSRF rules apply


def test_static_files_are_revalidated_not_blindly_cached(srv):
    for path in ("/index.html", "/data/x.json", "/api/ping"):
        (srv["settings"].site_dir / "data").mkdir(exist_ok=True)
        (srv["settings"].site_dir / "data" / "x.json").write_text("{}")
        r = requests.get(srv["url"] + path, timeout=10)
        assert r.status_code == 200 and r.headers["Cache-Control"] in ("no-cache", "no-store"), path
    first = requests.get(srv["url"] + "/index.html", timeout=10)
    again = requests.get(srv["url"] + "/index.html", headers={"If-Modified-Since": first.headers["Last-Modified"]}, timeout=10)
    assert again.status_code == 304                                      # cheap revalidation: no re-download when unchanged


def test_reason_requires_a_rating_validates_and_attaches_to_the_latest_one(srv):
    a, b = srv["ids"]
    assert post(srv, "/api/reason", {"id": a, "reason": "not_topic"}).status_code == 409            # nothing rated yet
    post(srv, "/api/rate", {"id": a, "sign": -1})
    assert post(srv, "/api/reason", {"id": a, "reason": "bogus"}).status_code == 400
    assert post(srv, "/api/reason", {"id": a, "reason": "weak_study"}).json() == {"ok": True, "id": a, "reason": "weak_study"}
    assert post(srv, "/api/reason", {"id": 9999, "reason": "weak_study"}).status_code == 404
    db = DB(srv["settings"].db_path)
    row = db.conn.execute("SELECT sign, reason, snapshot FROM feedback WHERE item_id=? ORDER BY id DESC LIMIT 1", (a,)).fetchone()
    snap = json.loads(row["snapshot"])
    assert row["sign"] == -1 and row["reason"] == "weak_study"
    assert snap["title"] == "First item title" and snap["stream"] == "grants" and "relevance" in snap and "domains" in snap
    post(srv, "/api/rate", {"id": a, "sign": 0})                                                       # cleared: no reason can hang on it
    assert post(srv, "/api/reason", {"id": a, "reason": "too_basic"}).status_code == 409
    db.close()


# ------------------------------------------------------------------ tuning API: proposals, accept / reject / undo
@pytest.fixture
def tsrv(srv):
    s = srv["settings"]
    db = DB(s.db_path)
    rid = db.add_report(12, 12, "Notes about patterns.", "claude-sonnet-5-5")
    ids = {"pref": db.add_proposal(rid, "preferences", {"old": "", "new": "Skips veterinary work."}, "Rewritten", [], {}),
           "rule": db.add_proposal(rid, "rule", {"kind": "term", "value": "veterinary", "weight": -2.0, "likes": 0, "dislikes": 4, "share": 0.17},
                                   "Disliked 4 of 4", list(srv["ids"]), {"matches": 6, "liked": 0, "disliked": 4, "unrated": 2, "days": 30}),
           "follow": db.add_proposal(rid, "follow", {"name": "Smith J", "likes": 2}, "Liked twice", list(srv["ids"]), {})}
    db.close()
    calls = []
    srv_httpd_api = srv  # noqa
    return {**srv, "pids": ids, "regen": calls}


def api_for(srv_dict, monkeypatch):
    from litsearch import web as w
    calls = srv_dict["regen"]
    monkeypatch.setattr(w.Api, "_regenerate", lambda self: calls.append(1))


def test_tuning_state_lists_pending_with_evidence_titles_and_progress(tsrv):
    st = requests.get(tsrv["url"] + "/api/tuning", timeout=10).json()
    assert [p["kind"] for p in st["pending"]] == ["preferences", "rule", "follow"] and st["notes"] == "Notes about patterns."
    rule = st["pending"][1]
    assert rule["payload"]["value"] == "veterinary" and rule["backtest"]["matches"] == 6
    assert [e["title"] for e in rule["evidence_items"]] == ["First item title", "Second item title"]
    assert st["need"] == 20 and "ratings" in st and st["history"] == []


def test_accept_writes_yaml_reject_does_not_undo_restores_and_each_only_once(tsrv, monkeypatch):
    api_for(tsrv, monkeypatch)
    pids = tsrv["pids"]
    yaml_path = tsrv["settings"].interests_path
    learned_now = lambda: config.load_interests(yaml_path)["learned"]
    assert post(tsrv, "/api/tuning/decide", {"id": pids["pref"], "action": "accept"}).json()["ok"] is True
    assert post(tsrv, "/api/tuning/decide", {"id": pids["rule"], "action": "accept"}).status_code == 200
    l = learned_now()
    assert l["preferences"] == "Skips veterinary work." and l["deprioritize"][0]["value"] == "veterinary" and l["deprioritize"][0]["weight"] == -2.0
    assert post(tsrv, "/api/tuning/decide", {"id": pids["rule"], "action": "accept"}).status_code == 409                # not twice
    assert post(tsrv, "/api/tuning/decide", {"id": pids["follow"], "action": "reject"}).status_code == 200
    assert learned_now()["prefer"] == []                                                                                  # rejecting changes nothing
    assert post(tsrv, "/api/tuning/decide", {"id": pids["follow"], "action": "undo"}).status_code == 409                # only accepted ones can be undone
    assert post(tsrv, "/api/tuning/decide", {"id": pids["rule"], "action": "undo"}).status_code == 200
    assert learned_now()["deprioritize"] == [] and learned_now()["preferences"] == "Skips veterinary work."            # only that proposal rolled back
    st = requests.get(tsrv["url"] + "/api/tuning", timeout=10).json()
    assert st["pending"] == [] and [(h["kind"], h["status"]) for h in st["history"]][:3] == [("follow", "rejected"), ("rule", "undone"), ("preferences", "accepted")]
    assert len(tsrv["regen"]) == 3                                                                                        # 2 accepts + 1 undo re-rank the page; reject doesn't


def test_tuning_decide_validation_and_csrf(tsrv):
    assert post(tsrv, "/api/tuning/decide", {"id": 99999, "action": "accept"}).status_code == 404
    assert post(tsrv, "/api/tuning/decide", {"id": tsrv["pids"]["rule"], "action": "delete"}).status_code == 400
    assert post(tsrv, "/api/tuning/decide", {"action": "accept"}).status_code == 400
    assert post(tsrv, "/api/tuning/decide", {"id": tsrv["pids"]["rule"], "action": "accept"}, headers={"Content-Type": "application/json"}).status_code == 403
