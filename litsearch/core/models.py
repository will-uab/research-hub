"""Item records. `Paper` is the literature stream's record; `Item` serves every other stream.

Core code is duck-typed over both: title, summary, kw_score, relevance, track, takeaway, fit_note, domains,
llm_scored, stream, source, source_id, url, date, stream_specific, store_keys(), lookup_keys().
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


def norm_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    d = doi.strip().lower()
    d = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", d)
    return d or None


def title_key(title: str) -> str:
    """Normalised title for preprint <-> published matching."""
    return re.sub(r"[^a-z0-9]+", "", (title or "").lower())


@dataclass
class Paper:
    source: str                      # pubmed | medrxiv | arxiv
    title: str
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    journal: str = ""
    date: str = ""                   # ISO-ish, as given by the source
    pmid: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    published_doi: str | None = None  # for preprints already published elsewhere
    is_preprint: bool = False
    is_major: bool = False           # came through the major-journal track
    pub_types: list[str] = field(default_factory=list)
    mesh: list[str] = field(default_factory=list)
    refs: list[tuple[str, str]] = field(default_factory=list)  # (RefType, PMID) corrections
    kw_score: float = 0.0
    matched: list[str] = field(default_factory=list)
    # triage results
    relevance: float | None = None
    track: str | None = None
    takeaway: str = ""
    fit_note: str = ""
    domains: list[str] = field(default_factory=list)
    llm_scored: bool = False
    is_notice: bool = False          # erratum/retraction for an already-sent paper

    def __post_init__(self):
        self.doi = norm_doi(self.doi)
        self.published_doi = norm_doi(self.published_doi)

    # ---- generic-item surface used by the core
    stream = "literature"

    @property
    def why(self) -> str:                 # template alias
        return self.fit_note

    @property
    def source_id(self) -> str:
        return self.key

    @property
    def summary(self) -> str:
        return self.abstract

    @property
    def stream_specific(self) -> dict:
        return {"authors": self.authors, "journal": self.journal, "pmid": self.pmid, "doi": self.doi,
                "arxiv_id": self.arxiv_id, "published_doi": self.published_doi, "is_preprint": self.is_preprint,
                "is_major": self.is_major, "pub_types": self.pub_types, "links": self.links,
                "source_label": self.source_label}

    def store_keys(self) -> list[tuple[str, str]]:
        """Identifiers recorded for dedup: (kind, value)."""
        ks = []
        if self.pmid:
            ks.append(("pmid", self.pmid))
        if self.doi:
            ks.append(("doi", self.doi))
        if self.arxiv_id:
            ks.append(("arxiv", self.arxiv_id))
        if self.published_doi:
            ks.append(("pdoi", self.published_doi))
        tk = title_key(self.title)
        if len(tk) >= 20:                 # avoid collisions on very short titles
            ks.append(("title", tk))
        return ks

    def lookup_keys(self) -> list[tuple[str, str]]:
        """What to look for among stored items, incl. preprint <-> published links."""
        ks = []
        if self.pmid:
            ks.append(("pmid", self.pmid))
        if self.arxiv_id:
            ks.append(("arxiv", self.arxiv_id))
        for d in (self.doi, self.published_doi):
            if d:
                ks += [("doi", d), ("pdoi", d)]
        tk = title_key(self.title)
        if len(tk) >= 20:
            ks.append(("title", tk))
        return ks

    @property
    def key(self) -> str:
        """Stable id used on the wire and in feedback codes."""
        if self.pmid:
            return self.pmid
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id}"
        return f"doi:{self.doi}" if self.doi else f"title:{title_key(self.title)[:40]}"

    @property
    def url(self) -> str:
        if self.doi:
            return f"https://doi.org/{self.doi}"
        if self.pmid:
            return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"
        if self.arxiv_id:
            return f"https://arxiv.org/abs/{self.arxiv_id}"
        return ""

    @property
    def links(self) -> list[tuple[str, str]]:
        out = []
        if self.pmid:
            out.append(("PubMed", f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"))
        if self.doi:
            out.append(("DOI", f"https://doi.org/{self.doi}"))
        if self.arxiv_id:
            out.append(("arXiv", f"https://arxiv.org/abs/{self.arxiv_id}"))
        return out

    @property
    def source_label(self) -> str:
        if self.source == "arxiv":
            return "arXiv"
        if self.source == "medrxiv":
            return "medRxiv"
        return self.journal or "PubMed"

    @property
    def author_str(self) -> str:
        a = self.authors
        if not a:
            return ""
        return ", ".join(a[:3]) + (" et al." if len(a) > 3 else "")

    @property
    def feedback_code(self) -> str:
        return self.key


@dataclass
class Item:
    """Generic record for the non-literature streams (grants, trials, awards, journal club)."""
    stream: str
    source: str
    source_id: str
    title: str
    url: str = ""
    date: str = ""
    summary: str = ""
    domains: list[str] = field(default_factory=list)
    relevance: float | None = None
    takeaway: str = ""
    fit_note: str = ""
    stream_specific: dict = field(default_factory=dict)
    kw_score: float = 0.0
    track: str | None = None
    llm_scored: bool = False
    is_notice: bool = False
    id: int | None = None

    @property
    def key(self) -> str:
        return self.source_id

    def store_keys(self) -> list[tuple[str, str]]:
        return [("id", f"{self.source}:{self.source_id}")]

    def lookup_keys(self) -> list[tuple[str, str]]:
        return self.store_keys()
