"""Cheap local keyword/MeSH scoring and exclusion rules (no LLM)."""
from __future__ import annotations

import re

from .core.matching import compile_terms
from .core.models import Paper


def _norm_mesh(items) -> set[str]:
    return {m.lower() for m in items}


def clause_hits(clause: dict, title: str, text: str, mesh: set[str]) -> tuple[bool, bool]:
    """(matched anywhere, matched in title)."""
    rx = compile_terms(tuple(clause.get("tiab", [])))
    in_title = bool(rx and rx.search(title))
    in_text = bool(rx and rx.search(text))
    in_mesh = any(m.lower() in mesh for m in clause.get("mesh", []))
    return (in_title or in_text or in_mesh), in_title


def score_paper(paper: Paper, groups: list[dict]) -> tuple[float, list[str]]:
    """Group weights where every clause matches; title hits boost; diminishing sum."""
    title = paper.title or ""
    text = f"{title}\n{paper.abstract or ''}"
    mesh = _norm_mesh(paper.mesh)
    scores, matched = [], []
    for g in groups:
        hits = [clause_hits(c, title, text, mesh) for c in g["all"]]
        if all(h[0] for h in hits):
            boost = 1.5 if any(h[1] for h in hits) else 1.0
            scores.append(g["weight"] * boost)
            matched.append(g["id"])
    if not scores:
        return 0.0, []
    scores.sort(reverse=True)
    return scores[0] + 0.3 * sum(scores[1:]), matched


def has_health_terms(paper: Paper, terms: list[str]) -> bool:
    rx = compile_terms(tuple(terms))
    return bool(rx and rx.search(f"{paper.title}\n{paper.abstract}"))


def is_excluded_type(paper: Paper, exclude_cfg: dict) -> bool:
    """Erratum / comment / retraction style records."""
    bad = {t.lower() for t in exclude_cfg.get("publication_types", [])}
    if any(t.lower() in bad for t in paper.pub_types):
        return True
    for pat in exclude_cfg.get("title_patterns", []):
        if re.search(pat, paper.title.strip(), re.IGNORECASE):
            return True
    return False


def score_all(papers: list[Paper], cfg: dict) -> None:
    """Fill kw_score/matched in place. Major-journal items get a floor score."""
    for p in papers:
        p.kw_score, p.matched = score_paper(p, cfg["groups"])
        if p.is_major and p.kw_score == 0:
            p.kw_score = 1.0  # passed the journal + topic-net filter in PubMed
