"""Full text of open-access articles, for the podcast's deep dive. Only legitimately open copies are used:

  * Europe PMC (PubMed Central open-access subset, and preprints that Europe PMC carries with full text), found from a PMID or DOI;
  * arXiv's own HTML version;
  * the open-access PDF that OpenAlex lists for the DOI (gold/green/hybrid copies, incl. medRxiv), read with `pdftotext`.
    A publisher or repository that refuses a scripted download is simply skipped: nothing here works around access controls.

Nothing is stored: the text is read once, condensed to fit the model call, and dropped. If no open full text exists, `fetch` returns None
and the caller picks another article.
"""
from __future__ import annotations

import logging
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

from ..core.http import RateLimiter, request

log = logging.getLogger(__name__)
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
MIN_WORDS = 1200                                      # shorter than this is an abstract-only record, a letter or a stub
_LIM = RateLimiter(0.3)
SKIP_TAGS = {"table-wrap", "fig", "ref-list", "back", "xref", "supplementary-material", "label", "object-id", "graphic", "media"}
SECTION_WEIGHT = {"abstract": 0.6, "intro": 0.5, "method": 1.0, "result": 1.6, "discussion": 1.2, "conclusion": 0.6, "limitation": 1.0}


def _local(tag) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _text(el) -> str:
    """Text of an element, minus citations, tables, figures and reference lists."""
    parts = [el.text or ""]
    for ch in el:
        if _local(ch.tag) not in SKIP_TAGS:
            parts.append(_text(ch))
        parts.append(ch.tail or "")
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def jats_sections(xml: bytes) -> tuple[str, list[tuple[str, str]]]:
    """(title, [(heading, text)]) from a JATS article: abstract first, then the body's top-level sections."""
    root = ET.fromstring(xml)
    title = ""
    for el in root.iter():
        if _local(el.tag) == "article-title":
            title = _text(el)
            break
    out: list[tuple[str, str]] = []
    body = None
    for el in root.iter():
        n = _local(el.tag)
        if n == "abstract" and not any(h == "Abstract" for h, _ in out):
            out.append(("Abstract", " ".join(_text(p) for p in el.iter() if _local(p.tag) == "p") or _text(el)))
        elif n == "body" and body is None:
            body = el
    if body is None:
        return title, out
    loose = [_text(p) for p in body if _local(p.tag) == "p"]
    if loose:
        out.append(("", " ".join(loose)))
    def walk(el, paras: list[str], top: bool) -> None:
        for ch in el:
            n = _local(ch.tag)
            if n in SKIP_TAGS:
                continue
            if n == "title" and not top:
                paras.append(_text(ch) + ".")                           # a subsection heading, kept inline
            elif n == "p":
                paras.append(_text(ch))
            elif n in ("sec", "boxed-text", "list", "list-item", "disp-quote"):
                walk(ch, paras, False)

    for sec in body:
        if _local(sec.tag) != "sec":
            continue
        head = next((_text(t) for t in sec if _local(t.tag) == "title"), "")
        paras: list[str] = []
        walk(sec, paras, True)
        text = " ".join(p for p in paras if p)
        if text:
            out.append((head, text))
    return title, [(h, t) for h, t in out if t.strip()]


class _ArxivHTML(HTMLParser):
    """Paragraph and heading text from arXiv's LaTeXML HTML, skipping the bibliography, footers and math alt-text."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.sections: list[tuple[str, list[str]]] = [("", [])]
        self._buf: list[str] = []
        self._cap: str | None = None
        self._skip = 0
        self._stack: list[bool] = []

    def handle_starttag(self, tag, attrs):
        cls = dict(attrs).get("class", "") or ""
        skip = any(k in cls for k in ("ltx_bibliography", "ltx_page_footer", "ltx_authors", "ltx_title_document", "ltx_abstract_title")) or tag in ("script", "style", "nav", "math", "annotation")
        if tag in ("p", "h1", "h2", "h3", "h4", "h5", "h6") and not self._skip and not skip:
            self._cap, self._buf = tag, []
        if tag not in ("br", "img", "meta", "link", "input"):
            self._stack.append(skip)
            self._skip += skip

    def handle_endtag(self, tag):
        if tag in ("br", "img", "meta", "link", "input") or not self._stack:
            return
        self._skip -= self._stack.pop()
        if self._cap == tag:
            text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
            self._cap = None
            if text:
                if tag != "p":
                    if tag in ("h1", "h2"):
                        self.sections.append((text, []))
                    else:
                        self.sections[-1][1].append(text + ".")
                else:
                    self.sections[-1][1].append(text)

    def handle_data(self, data):
        if self._cap and not self._skip:
            self._buf.append(data)


def arxiv_sections(html: str) -> list[tuple[str, str]]:
    p = _ArxivHTML()
    p.feed(html)
    return [(h, " ".join(ps)) for h, ps in p.sections if ps]


def condense(sections: list[tuple[str, str]], cap_chars: int) -> str:
    """Fit the article into `cap_chars`, trimming each section's tail in proportion (results weigh most, intro least)."""
    def weight(head: str) -> float:
        h = head.lower()
        return next((w for k, w in SECTION_WEIGHT.items() if k in h), 0.8)
    total = sum(len(t) for _, t in sections)
    if total <= cap_chars:
        keep = {i: len(t) for i, (_, t) in enumerate(sections)}
    else:
        want = [len(t) * weight(h) for h, t in sections]
        scale = cap_chars / max(1.0, sum(want))
        keep = {i: min(len(t), int(w * scale)) for i, ((h, t), w) in enumerate(zip(sections, want))}
    parts = []
    for i, (h, t) in enumerate(sections):
        piece = t if keep[i] >= len(t) else t[:keep[i]].rsplit(" ", 1)[0] + " [...]"
        parts.append((f"## {h}\n" if h else "") + piece)
    return "\n\n".join(parts)


HEADING = re.compile(r"^\s*(?:\d+\.?\s+)?(abstract|introduction|background|methods?|materials and methods|patients and methods|results|findings|discussion|"
                     r"conclusions?|limitations|strengths and limitations)\s*:?\s*$", re.I)
MAX_PDF_BYTES = 25_000_000


def pdf_sections(text: str) -> list[tuple[str, str]]:
    """Split pdftotext output at standard headings and drop the reference list; one section when no headings are found."""
    lines = text.replace("\x0c", "\n").splitlines()
    for i in range(len(lines) - 1, len(lines) // 2, -1):               # the last "References" line in the second half starts the bibliography
        if re.fullmatch(r"\s*(?:\d+\.?\s+)?(references|bibliography|literature cited)\s*", lines[i], re.I):
            lines = lines[:i]
            break
    secs: list[tuple[str, list[str]]] = [("", [])]
    for ln in lines:
        m = HEADING.match(ln)
        if m:
            secs.append((m.group(1).title(), []))
        elif ln.strip():
            secs[-1][1].append(ln.strip())
    out = [(h, re.sub(r"\s+", " ", " ".join(ls)).strip()) for h, ls in secs]
    return [(h, t) for h, t in out if t]


def _pdf_text(url: str) -> str:
    r = request("GET", url, _LIM, timeout=90, retries=1, headers={"Accept": "application/pdf"})
    data = r.content
    if not data.startswith(b"%PDF") or len(data) > MAX_PDF_BYTES:
        raise ValueError("not a PDF, or too large")
    with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
        f.write(data)
        f.flush()
        return subprocess.run(["pdftotext", "-nopgbrk", f.name, "-"], capture_output=True, text=True, timeout=90, check=True).stdout


def _openalex_pdf_urls(doi: str) -> list[str]:
    j = request("GET", f"https://api.openalex.org/works/doi:{doi}", _LIM, timeout=30, retries=1).json()
    locs = [j.get("best_oa_location") or {}] + (j.get("locations") or [])
    urls = []
    for loc in locs:
        u = loc.get("pdf_url")
        if u and loc.get("is_oa", True) and u not in urls:
            urls.append(u)
    return urls[:3]


def _words(sections: list[tuple[str, str]]) -> int:
    return sum(len(t.split()) for _, t in sections)


def _epmc_record(ss: dict) -> dict | None:
    queries = []
    if ss.get("pmid"):
        queries.append(f"EXT_ID:{ss['pmid']} AND SRC:MED")
    for d in (ss.get("doi"), ss.get("published_doi")):
        if d:
            queries.append(f'DOI:"{d}"')
    for q in queries:
        r = request("GET", f"{EPMC}/search", _LIM, params={"query": q, "format": "json", "resultType": "lite", "pageSize": 3}, timeout=30).json()
        for rec in (r.get("resultList") or {}).get("result", []):
            if rec.get("pmcid") or rec.get("source") == "PPR":
                return rec
    return None


def fetch(ss: dict, cap_chars: int = 28000) -> dict | None:
    """Open full text for an item's stream_specific (pmid / doi / published_doi / arxiv_id), or None."""
    tries: list[tuple[str, str]] = []
    try:
        rec = _epmc_record(ss)
    except Exception as e:  # noqa: BLE001
        log.info("Europe PMC lookup failed: %s", e)
        rec = None
    if rec:
        if rec.get("pmcid") and rec.get("isOpenAccess", "Y") != "N":
            tries.append(("pmc", rec["pmcid"]))
        elif rec.get("source") == "PPR" and rec.get("id"):
            tries.append(("preprint", rec["id"]))
    for kind, ident in tries:
        try:
            xml = request("GET", f"{EPMC}/{ident}/fullTextXML", _LIM, timeout=60, retries=1).content
            title, secs = jats_sections(xml)
        except Exception as e:  # noqa: BLE001 - 404 = not open access; parse errors = not usable
            log.info("full text %s unavailable: %s", ident, e)
            continue
        if _words(secs) >= MIN_WORDS:
            return {"text": condense(secs, cap_chars), "source": "Europe PMC" + (" (PMC open access)" if kind == "pmc" else " (preprint)"),
                    "url": f"https://europepmc.org/article/{'PMC' if kind == 'pmc' else 'PPR'}/{ident.removeprefix('PMC')}", "words": _words(secs)}
    aid = (ss.get("arxiv_id") or "").strip()
    if aid:
        try:
            html = request("GET", f"https://arxiv.org/html/{aid}", _LIM, timeout=60, retries=1).text
            secs = arxiv_sections(html)
        except Exception as e:  # noqa: BLE001
            log.info("arXiv HTML for %s unavailable: %s", aid, e)
            secs = []
        if _words(secs) >= MIN_WORDS:
            return {"text": condense(secs, cap_chars), "source": "arXiv (HTML)", "url": f"https://arxiv.org/abs/{aid}", "words": _words(secs)}
    doi = (ss.get("doi") or ss.get("published_doi") or "").strip()
    if doi:
        try:
            urls = _openalex_pdf_urls(doi)
        except Exception as e:  # noqa: BLE001
            log.info("OpenAlex OA lookup failed for %s: %s", doi, e)
            urls = []
        for u in urls:
            try:
                secs = pdf_sections(_pdf_text(u))
            except Exception as e:  # noqa: BLE001 - blocked, not a PDF, or unreadable: try the next copy
                log.info("open PDF %s unavailable: %s", u, e)
                continue
            if _words(secs) >= MIN_WORDS:
                return {"text": condense(secs, cap_chars), "source": "open-access PDF via OpenAlex", "url": u, "words": _words(secs)}
    return None
