"""PubMed via NCBI E-utilities (esearch + efetch)."""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import date

from .. import queries
from ..core.http import RateLimiter, request
from ..core.models import Paper

log = logging.getLogger(__name__)
BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
RETMAX = 2000
FETCH_CHUNK = 150


def _limiter(api_key: str) -> RateLimiter:
    # NCBI: 3 req/s without a key, 10 req/s with one. Stay a little under.
    return RateLimiter(0.11 if api_key else 0.40)


def esearch(term: str, start: date, end: date, api_key: str, limiter: RateLimiter) -> list[str]:
    data = {"db": "pubmed", "term": term, "retmode": "json", "retmax": RETMAX, "tool": "litsearch",
            **queries.date_params(start, end)}
    if api_key:
        data["api_key"] = api_key
    r = request("POST", f"{BASE}/esearch.fcgi", limiter, data=data)
    res = r.json()["esearchresult"]
    if "ERROR" in res:
        raise RuntimeError(f"esearch error: {res['ERROR']}")
    if res.get("errorlist"):
        log.warning("esearch errorlist: %s", res["errorlist"])
    if res.get("warninglist"):
        log.warning("esearch warnings: %s", res["warninglist"])
    ids = res.get("idlist", [])
    if int(res.get("count", 0)) > len(ids):
        log.warning("esearch returned %s of %s hits (retmax)", len(ids), res["count"])
    return ids


def efetch(pmids: list[str], api_key: str, limiter: RateLimiter) -> list[ET.Element]:
    out: list[ET.Element] = []
    for i in range(0, len(pmids), FETCH_CHUNK):
        chunk = pmids[i:i + FETCH_CHUNK]
        data = {"db": "pubmed", "id": ",".join(chunk), "retmode": "xml", "tool": "litsearch"}
        if api_key:
            data["api_key"] = api_key
        r = request("POST", f"{BASE}/efetch.fcgi", limiter, data=data, timeout=120)
        out.extend(ET.fromstring(r.content).findall("PubmedArticle"))
    return out


def _text(el: ET.Element | None) -> str:
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}


def _date(art: ET.Element) -> str:
    ad = art.find("MedlineCitation/Article/ArticleDate")
    if ad is not None and ad.findtext("Year"):
        return f"{ad.findtext('Year')}-{int(ad.findtext('Month', '1')):02d}-{int(ad.findtext('Day', '1')):02d}"
    pd = art.find("MedlineCitation/Article/Journal/JournalIssue/PubDate")
    if pd is None:
        return ""
    y = pd.findtext("Year") or (pd.findtext("MedlineDate") or "")[:4]
    m = pd.findtext("Month", "")
    mi = int(m) if m.isdigit() else _MONTHS.get(m[:3].lower())
    d = pd.findtext("Day", "")
    return "-".join(x for x in (y, f"{mi:02d}" if mi else "", f"{int(d):02d}" if d.isdigit() and mi else "") if x)


def parse_article(art: ET.Element) -> Paper:
    mc = art.find("MedlineCitation")
    article = mc.find("Article")
    parts = []
    for ab in article.findall("Abstract/AbstractText"):
        label, txt = ab.get("Label"), _text(ab)
        parts.append(f"{label}: {txt}" if label and txt else txt)
    authors = []
    for a in article.findall("AuthorList/Author"):
        if a.findtext("CollectiveName"):
            authors.append(a.findtext("CollectiveName"))
        elif a.findtext("LastName"):
            authors.append(f"{a.findtext('LastName')} {a.findtext('Initials', '')}".strip())
    doi = None
    for aid in art.findall("PubmedData/ArticleIdList/ArticleId"):
        if aid.get("IdType") == "doi":
            doi = aid.text
    if not doi:
        for e in article.findall("ELocationID"):
            if e.get("EIdType") == "doi":
                doi = e.text
    pub_types = [_text(t) for t in article.findall("PublicationTypeList/PublicationType")]
    refs = [(c.get("RefType", ""), c.findtext("PMID", ""))
            for c in mc.findall("CommentsCorrectionsList/CommentsCorrections") if c.findtext("PMID")]
    return Paper(
        source="pubmed",
        title=_text(article.find("ArticleTitle")).rstrip("."),
        abstract=" ".join(parts),
        authors=authors,
        journal=article.findtext("Journal/ISOAbbreviation") or article.findtext("Journal/Title") or "",
        date=_date(art),
        pmid=mc.findtext("PMID"),
        doi=doi,
        is_preprint="Preprint" in pub_types,
        pub_types=pub_types,
        mesh=[_text(d) for d in mc.findall("MeshHeadingList/MeshHeading/DescriptorName")],
        refs=refs,
    )


def fetch_pubmed(q: dict, start: date, end: date, api_key: str = "") -> tuple[list[Paper], str | None]:
    """`q` is queries.build_all() output. Returns (papers, error)."""
    limiter = _limiter(api_key)
    errors = []
    mine_ids: list[str] = []
    major_ids: list[str] = []
    for name, term, sink in (("mine", q["pubmed_mine"], mine_ids), ("major", q["pubmed_major"], major_ids)):
        try:
            sink.extend(esearch(term, start, end, api_key, limiter))
        except Exception as e:  # noqa: BLE001 - report and carry on with the other track
            log.exception("PubMed %s search failed", name)
            errors.append(f"PubMed {name} search: {e}")
    ids = list(dict.fromkeys(mine_ids + major_ids))
    log.info("PubMed: %d mine + %d major ids (%d unique)", len(mine_ids), len(major_ids), len(ids))
    papers: list[Paper] = []
    if ids:
        try:
            major_set = set(major_ids)
            for art in efetch(ids, api_key, limiter):
                try:
                    p = parse_article(art)
                except Exception:  # noqa: BLE001 - one bad record shouldn't sink the batch
                    log.exception("could not parse a PubMed record")
                    continue
                p.is_major = p.pmid in major_set
                papers.append(p)
        except Exception as e:  # noqa: BLE001
            log.exception("PubMed efetch failed")
            errors.append(f"PubMed efetch: {e}")
    return papers, "; ".join(errors) or None
