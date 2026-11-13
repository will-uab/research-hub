import pytest

from litsearch.sources import fulltext

BODY = "The cohort included adults with opioid use disorder who started treatment. " * 40
JATS = f"""<?xml version="1.0"?><article xmlns:xlink="http://www.w3.org/1999/xlink"><front><article-meta><title-group><article-title>An LLM for notes</article-title></title-group>
<abstract><p>We tested a model.</p></abstract></article-meta></front>
<body><sec><title>Introduction</title><p>Background {BODY} <xref ref-type="bibr" rid="b1">[1]</xref> more.</p></sec>
<sec><title>Methods</title><p>{BODY}</p><sec><title>Statistics</title><p>We used bootstrap intervals.</p></sec><fig><caption><p>FIGURE CAPTION TEXT</p></caption></fig>
<table-wrap><p>TABLE CELL TEXT</p></table-wrap></sec>
<sec><title>Results</title><p>Sensitivity was 0.91. {BODY}</p></sec></body>
<back><ref-list><ref><mixed-citation>REFERENCE TEXT</mixed-citation></ref></ref-list></back></article>""".encode()


def test_jats_sections_keep_prose_and_drop_citations_figures_tables_and_references():
    title, secs = fulltext.jats_sections(JATS)
    text = " ".join(t for _, t in secs)
    assert title == "An LLM for notes" and [h for h, _ in secs] == ["Abstract", "Introduction", "Methods", "Results"]
    assert "Sensitivity was 0.91" in text and "bootstrap intervals" in text and "Statistics." in text
    assert "FIGURE CAPTION" not in text and "TABLE CELL" not in text and "REFERENCE TEXT" not in text and "[1]" not in text


def test_condense_fits_the_cap_and_weights_results_over_intro():
    secs = [("Introduction", "i " * 5000), ("Methods", "m " * 5000), ("Results", "r " * 5000), ("Discussion", "d " * 5000)]
    out = fulltext.condense(secs, 8000)
    assert len(out) < 8600 and out.count("[...]") == 4 and out.count("r ") > out.count("i ")
    assert fulltext.condense([("Results", "short text")], 8000) == "## Results\nshort text"


def test_arxiv_html_paragraphs_headings_and_bibliography():
    html = """<html><body><h1 class="ltx_title_document">Title</h1><section><h2>1 Introduction</h2><p>First paragraph. <math><annotation>x^2</annotation></math></p></section>
    <section><h2>2 Results</h2><p>We report 0.9 accuracy.</p><h3>2.1 Ablation</h3><p>Removing retrieval hurts.</p></section>
    <section class="ltx_bibliography"><h2>References</h2><p>Smith 2020</p></section></body></html>"""
    secs = fulltext.arxiv_sections(html)
    text = " ".join(t for _, t in secs)
    assert [h for h, _ in secs][:2] == ["1 Introduction", "2 Results"]
    assert "0.9 accuracy" in text and "2.1 Ablation." in text and "Smith 2020" not in text and "x^2" not in text and "References" not in text


class Resp:
    def __init__(self, js=None, content=b"", text=""):
        self._js, self.content, self.text = js, content, text

    def json(self):
        return self._js


def test_fetch_uses_europe_pmc_open_access_full_text(monkeypatch):
    calls = []

    def fake(method, url, limiter=None, **kw):
        calls.append(url)
        if url.endswith("/search"):
            return Resp({"resultList": {"result": [{"id": "1", "source": "MED", "pmcid": "PMC123", "isOpenAccess": "Y"}]}})
        return Resp(content=JATS)
    monkeypatch.setattr(fulltext, "request", fake)
    monkeypatch.setattr(fulltext, "MIN_WORDS", 100)
    got = fulltext.fetch({"pmid": "1", "doi": "10.1/x"})
    assert got["source"].startswith("Europe PMC") and got["url"].endswith("/PMC/123") and "Sensitivity was 0.91" in got["text"]
    assert calls[0].endswith("/search") and calls[1].endswith("/PMC123/fullTextXML")


def test_fetch_returns_none_for_closed_or_too_short_articles(monkeypatch):
    def closed(method, url, limiter=None, **kw):
        if url.endswith("/search"):
            return Resp({"resultList": {"result": [{"id": "1", "source": "MED", "pmcid": "", "isOpenAccess": "N"}]}})
        raise AssertionError("must not fetch full text for a closed article")
    monkeypatch.setattr(fulltext, "request", closed)
    assert fulltext.fetch({"pmid": "1"}) is None
    def short(method, url, limiter=None, **kw):
        if url.endswith("/search"):
            return Resp({"resultList": {"result": [{"id": "1", "source": "MED", "pmcid": "PMC9", "isOpenAccess": "Y"}]}})
        return Resp(content=JATS)
    monkeypatch.setattr(fulltext, "request", short)
    monkeypatch.setattr(fulltext, "MIN_WORDS", 5000)
    assert fulltext.fetch({"pmid": "1"}) is None                      # fewer than MIN_WORDS: an abstract-sized record


def test_fetch_falls_back_to_arxiv_html_and_survives_errors(monkeypatch):
    para = " ".join(["word"] * 1600)
    def fake(method, url, limiter=None, **kw):
        if "europepmc" in url:
            raise RuntimeError("down")
        return Resp(text=f"<html><body><h2>Results</h2><p>{para}</p></body></html>")
    monkeypatch.setattr(fulltext, "request", fake)
    got = fulltext.fetch({"arxiv_id": "2601.00001"})
    assert got["source"] == "arXiv (HTML)" and got["url"] == "https://arxiv.org/abs/2601.00001" and got["words"] >= 1500
    assert fulltext.fetch({}) is None


def test_pdf_sections_split_on_headings_and_drop_references():
    body = "word " * 50
    text = f"Title page\nAbstract\n{body}\n1. Introduction\n{body}\nMethods\n{body}\nResults\nSensitivity was 0.9.\n{body}\nReferences\n1. Smith J. Old paper.\n"
    secs = fulltext.pdf_sections(text)
    assert [h for h, _ in secs] == ["", "Abstract", "Introduction", "Methods", "Results"]
    assert "Sensitivity was 0.9." in dict(secs)["Results"] and "Old paper" not in " ".join(t for _, t in secs)
    assert [h for h, _ in fulltext.pdf_sections("no headings here at all\njust text")] == [""]


def test_fetch_falls_back_to_the_open_access_pdf_and_skips_blocked_ones(monkeypatch):
    para = " ".join(["word"] * 1500)

    def fake(method, url, limiter=None, **kw):
        if "europepmc" in url:
            return Resp({"resultList": {"result": []}})
        if "openalex" in url:
            return Resp({"best_oa_location": {"pdf_url": "https://pub.example/blocked.pdf", "is_oa": True},
                         "locations": [{"pdf_url": "https://repo.example/ok.pdf", "is_oa": True}]})
        if "blocked" in url:
            raise RuntimeError("403")
        return Resp(content=b"%PDF-1.5 fake")
    monkeypatch.setattr(fulltext, "request", fake)
    monkeypatch.setattr(fulltext, "_pdf_text", lambda url: (_ for _ in ()).throw(RuntimeError("blocked")) if "blocked" in url else f"Results\n{para}\n")
    got = fulltext.fetch({"doi": "10.1/x"})
    assert got["source"].startswith("open-access PDF") and got["url"] == "https://repo.example/ok.pdf"
    monkeypatch.setattr(fulltext, "_pdf_text", lambda url: (_ for _ in ()).throw(RuntimeError("blocked")))
    assert fulltext.fetch({"doi": "10.1/x"}) is None
