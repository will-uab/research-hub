import pytest

from litsearch import digest as D
from litsearch.core import pipeline
from litsearch.core.config import Settings
from litsearch.core.context import RunOptions
from litsearch.core.db import DB
from litsearch.core.models import Paper
from litsearch.streams import literature

from tests.test_budget import FakeClient, _good

LONGT = "Large language models for chart review of HIV clinical notes number {}"


def mk(n, **kw):
    return Paper("pubmed", LONGT.format(n), "LLM clinical note chart review abstract", pmid=str(n),
                 journal="JAMIA", kw_score=10, **kw)


def scored(p, rel, track="mine"):
    p.relevance, p.track, p.llm_scored, p.takeaway = rel, track, True, "t"
    return p


def test_select_thresholds_caps_and_sort(cfg):
    ps = [scored(mk(1), 9), scored(mk(2), 5), scored(mk(3), 7), scored(mk(4), 8, "major"),
          scored(mk(5), 3, "major"), scored(mk(6), 9, "drop")]
    mine, major = D.select(ps, [], cfg)
    assert [p.pmid for p in mine] == ["1", "3"] and [p.pmid for p in major] == ["4"]


def test_select_fallback_keyword_ranked_when_unscored(cfg):
    un = [mk(i) for i in range(30)]
    for i, p in enumerate(un):
        p.kw_score = i
        p.is_major = i % 2 == 0
    mine, major = D.select([], un, cfg)
    assert len(mine) == cfg["digest"]["fallback_max_items"] and mine[0].kw_score > mine[-1].kw_score
    assert all(not p.is_major for p in mine) and all(p.is_major for p in major)


def test_render_subject_empty_fallback_and_escaping(cfg):
    st = D.Stats(retrieved=5, by_source="pubmed 5", screened=2, sent=1, budget=0.1)
    p = scored(mk(1), 8)
    p.title = "<script>x</script> & stuff"
    p.is_preprint = True
    d = D.render("2026-11-03", [p], [], [], st, ["PubMed efetch: boom"], "daily budget reached", 1)
    assert d.subject == "Lit digest 2026-11-03: 1 for you, 0 major"
    assert "&lt;script&gt;" in d.html and "<script>" not in d.html
    assert "PREPRINT" in d.html and "<details" in d.html and "Budget guard tripped" in d.html
    assert "boom" in d.html and "boom" in d.text and "lit feedback +1" in d.text
    e = D.render("2026-11-03", [], [], [], st, [])
    assert e.subject == "Lit digest 2026-11-03: nothing notable today" and e.empty


# ------------------------------------------------------------------ integration (network + SMTP + API mocked)
@pytest.fixture
def env(tmp_path, monkeypatch, cfg2):
    cfg = cfg2
    settings = Settings(env={"DAILY_BUDGET_USD": "0.10", "ANTHROPIC_API_KEY": "k", "PRICE_INPUT_PER_MTOK": "0.1",
                             "PRICE_OUTPUT_PER_MTOK": "0.5", "SMTP_HOST": "h", "SMTP_USER": "u",
                             "SMTP_PASSWORD": "p", "MAIL_FROM": "a@b", "MAIL_TO": "c@d", "DELIVERY": "email"},
                        root=tmp_path, has_env_file=True)
    sent = []
    monkeypatch.setattr(literature, "retrieve", lambda *a, **k: ([mk(1), mk(2), mk(3)], [], {"pubmed": 3}))
    monkeypatch.setattr(literature.mailer, "send", lambda s, subj, html, text: sent.append(subj))
    import anthropic
    client = FakeClient([_good(["p1", "p2", "p3"])])
    monkeypatch.setattr(anthropic, "Anthropic", lambda **k: client)
    return settings, cfg, sent, client


def test_end_to_end_sends_records_and_never_resends(env):
    settings, cfg, sent, client = env
    assert pipeline.run(settings, cfg, RunOptions()) == 0
    assert sent == ["Lit digest %s: 3 for you, 0 major" % sent[0].split()[2].rstrip(":")]
    db = DB(settings.db_path)
    assert db.conn.execute("SELECT COUNT(*) FROM items WHERE status='sent'").fetchone()[0] == 3
    assert db.spent(sent[0].split()[2].rstrip(":")) > 0
    db.close()
    # second run: same papers retrieved again -> all deduped -> "nothing notable" and no LLM call
    calls = client.calls
    pipeline.run(settings, cfg, RunOptions())
    assert client.calls == calls and sent[1].endswith("nothing notable today")


def test_failed_send_records_nothing(env, monkeypatch):
    settings, cfg, sent, client = env
    def boom(*a): raise OSError("smtp down")
    monkeypatch.setattr(literature.mailer, "send", boom)
    with pytest.raises(OSError):
        pipeline.run(settings, cfg, RunOptions())
    db = DB(settings.db_path)
    assert db.conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 0
    db.close()


def test_budget_trip_sends_keyword_ranked_with_note(env):
    settings, cfg, sent, client = env
    settings.env["DAILY_BUDGET_USD"] = "0.0000001"
    captured = {}
    orig = literature.digest_mod.render
    def spy(*a, **k):
        d = orig(*a, **k); captured["d"] = d; return d
    literature.digest_mod.render = spy
    try:
        pipeline.run(settings, cfg, RunOptions())
    finally:
        literature.digest_mod.render = orig
    assert client.calls == 0
    assert "Budget guard tripped" in captured["d"].html and len(captured["d"].mine) == 3


def test_dry_run_records_nothing_and_needs_no_key(env, tmp_path):
    settings, cfg, sent, client = env
    settings.env.pop("ANTHROPIC_API_KEY"); settings.env.pop("SMTP_HOST")
    (tmp_path / "out").mkdir(exist_ok=True)
    assert pipeline.run(settings, cfg, RunOptions(dry_run=True)) == 0
    assert client.calls == 0 and sent == [] and not settings.db_path.exists()
    assert list((tmp_path / "out").glob("digest-*.html"))


def test_env_parsing_and_permission_warning(tmp_path):
    from litsearch.core import config
    f = tmp_path / ".env"
    f.write_text("# c\nA=1\nB='two words'\nEMPTY=\nC=x=y\n")
    f.chmod(0o644)
    assert config.parse_env_file(f) == {"A": "1", "B": "two words", "EMPTY": "", "C": "x=y"}
    assert "chmod 600" in config.check_env_permissions(f)
    f.chmod(0o600)
    assert config.check_env_permissions(f) is None


# ------------------------------------------------------------------ web delivery
def _web(env):
    settings, cfg, sent, client = env
    settings.env["DELIVERY"] = "web"
    settings.env.pop("SMTP_HOST")          # web-only must not need SMTP
    return settings, cfg, sent, client


def test_web_delivery_publishes_page_and_index_without_smtp(env):
    settings, cfg, sent, client = _web(env)
    assert pipeline.run(settings, cfg, RunOptions()) == 0
    assert sent == []
    pages = [p for p in settings.site_dir.glob("*.html") if p.name not in ("index.html", "digests.html")]
    assert len(pages) == 1
    page = pages[0].read_text()
    assert "Daily digests" in page and "<details" in page and "lit feedback +1" in page
    index = (settings.site_dir / "digests.html").read_text()
    assert pages[0].name in index and "3 for you" in index
    db = DB(settings.db_path)
    assert db.conn.execute("SELECT COUNT(*) FROM items WHERE status='sent'").fetchone()[0] == 3
    db.close()


def test_second_run_same_day_does_not_overwrite_first_page(env, monkeypatch):
    settings, cfg, sent, client = _web(env)
    pipeline.run(settings, cfg, RunOptions())
    monkeypatch.setattr(literature, "retrieve", lambda *a, **k: ([mk(7)], [], {"pubmed": 1}))
    client.texts.append(_good(["p1"]))
    pipeline.run(settings, cfg, RunOptions())
    pages = {p.name for p in settings.site_dir.glob("*.html") if p.name not in ("index.html", "digests.html")}
    assert len(pages) == 2 and any(n.endswith("-2.html") for n in pages)
    import json
    assert len(json.loads((settings.site_dir / "manifest.json").read_text())) == 2


def test_both_delivery_emails_without_back_link(env):
    settings, cfg, sent, client = env
    settings.env["DELIVERY"] = "both"
    captured = {}
    literature.mailer.send = lambda s, subj, html, text: captured.update(html=html)
    pipeline.run(settings, cfg, RunOptions())
    assert "Daily digests" not in captured["html"]
    assert any("Daily digests" in p.read_text() for p in settings.site_dir.glob("2*.html"))


def test_web_failure_page_and_prune(tmp_path):
    from datetime import datetime, timedelta
    from litsearch import publish
    now = datetime(2026, 11, 3, 5, 30)
    st = D.Stats(budget=0.1)
    old = D.render("2026-07-01", [], [], [], st, [])
    publish.publish(tmp_path, old, "2026-07-01", now - timedelta(days=125))
    publish.publish_failure(tmp_path, "2026-11-03", "<b>boom</b>", now)
    publish.publish(tmp_path, D.render("2026-11-03", [], [], [], st, []), "2026-11-03", now)  # prunes July
    assert not (tmp_path / "2026-07-01.html").exists()
    idx = (tmp_path / "digests.html").read_text()
    assert "FAILED" in idx and "nothing notable" in idx and "2026-07-01" not in idx
    assert "&lt;b&gt;boom" in (tmp_path / "failed-2026-11-03.html").read_text()


def test_invalid_delivery_value_rejected(tmp_path):
    with pytest.raises(ValueError):
        Settings(env={"DELIVERY": "carrier-pigeon"}, root=tmp_path).delivery
    assert Settings(env={}, root=tmp_path).delivery == {"web"}
    assert Settings(env={"DELIVERY": "both"}, root=tmp_path).delivery == {"web", "email"}


# ------------------------------------------------------------------ core: runs table, stream registry, per-stream budget
def test_run_is_recorded_for_the_status_banner(env):
    settings, cfg, sent, client = env
    pipeline.run(settings, cfg, RunOptions())
    db = DB(settings.db_path)
    r = db.conn.execute("SELECT * FROM runs WHERE stream='literature'").fetchone()
    import json
    assert r["ok"] == 1 and r["finished"] and json.loads(r["counts"])["retrieved"] == 3
    db.close()


def test_failed_run_is_recorded_with_error(env, monkeypatch):
    settings, cfg, sent, client = env
    def boom(*a, **k): raise RuntimeError("all sources failed")
    monkeypatch.setattr(literature, "retrieve", boom)
    with pytest.raises(RuntimeError):
        pipeline.run(settings, cfg, RunOptions())
    db = DB(settings.db_path)
    r = db.conn.execute("SELECT ok, error FROM runs").fetchone()
    assert r["ok"] == 0 and "all sources failed" in r["error"]
    db.close()


def test_unknown_and_disabled_streams(env):
    settings, cfg, sent, client = env
    with pytest.raises(ValueError):
        pipeline.run(settings, cfg, RunOptions(), stream="nope")
    cfg["streams"]["literature"]["enabled"] = False
    try:
        assert pipeline.run(settings, cfg, RunOptions()) == 0 and client.calls == 0
    finally:
        cfg["streams"]["literature"]["enabled"] = True


# ------------------------------------------------------------------ golden: refactor must not change literature output
def test_literature_output_identical_to_v1(tmp_path, cfg2, monkeypatch):
    """expected_v1.txt was produced by the pre-refactor (v1) code on the same fixture and an empty database.
    Only the first line (date) and the cost-estimate line (the prompt got longer on purpose) are masked."""
    import json, re
    from pathlib import Path
    fx = Path(__file__).parent / "fixtures" / "literature_golden"
    raw = json.loads((fx / "papers.json").read_text())

    def mk_paper(d):
        d = dict(d); d["fit_note"] = d.pop("why", "")
        return Paper(**d)

    monkeypatch.setattr(literature, "retrieve", lambda *a, **k: ([mk_paper(d) for d in raw["papers"]], raw["errors"], raw["counts"]))
    (tmp_path / ".env.example").write_text((Path(__file__).parent.parent / ".env.example").read_text())
    settings = Settings(env={"DAILY_BUDGET_USD": "0.10"}, root=tmp_path)
    assert pipeline.run(settings, cfg2, RunOptions(dry_run=True, backfill=3)) == 0
    got = next((tmp_path / "out").glob("digest-*.txt")).read_text()

    def mask(t):
        lines = t.splitlines()
        return [re.sub(r"\d{4}-\d{2}-\d{2}", "DATE", lines[0]), *[re.sub(r"estimated cost for this run \$[\d.]+", "COST", l) for l in lines[1:]]]
    exp = (fx / "expected_v1.txt").read_text()
    assert mask(got) == mask(exp)
    assert got.count("Feedback: lit feedback") >= 20      # the fixture is non-trivial
