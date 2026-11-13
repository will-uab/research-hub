"""Build PubMed / arXiv / medRxiv query strings from config/interests.yaml."""
from __future__ import annotations

from datetime import date


def _pm_term(term: str) -> str:
    """One title/abstract term in PubMed syntax."""
    term = term.strip()
    if " " in term or "-" in term:
        return f'"{term}"[tiab]'
    return f"{term}[tiab]"


def clause_query(clause: dict) -> str:
    parts = [f'"{m}"[MeSH Terms]' for m in clause.get("mesh", [])]
    parts += [_pm_term(t) for t in clause.get("tiab", [])]
    if not parts:
        raise ValueError("clause has neither mesh nor tiab terms")
    return "(" + " OR ".join(parts) + ")"


def group_query(group: dict) -> str:
    """All clauses ANDed together."""
    return "(" + " AND ".join(clause_query(c) for c in group["all"]) + ")"


def interests_query(groups: list[dict]) -> str:
    """OR of every interest group: the 'mine' track PubMed query."""
    return " OR ".join(group_query(g) for g in groups)


def net_query(net: dict) -> str:
    return clause_query(net)


def journal_query(journals: list[dict], net: dict) -> str:
    """Major-journal track: unfiltered journals OR (net-filtered journals AND topic net)."""
    def ta(js):
        return "(" + " OR ".join(f'"{j["ta"]}"[ta]' for j in js) + ")"

    open_j = [j for j in journals if j.get("filter", "net") == "none"]
    net_j = [j for j in journals if j.get("filter", "net") != "none"]
    parts = []
    if open_j:
        parts.append(ta(open_j))
    if net_j:
        parts.append(f"({ta(net_j)} AND {net_query(net)})")
    if not parts:
        raise ValueError("no major journals configured")
    return " OR ".join(parts)


def date_params(start: date, end: date) -> dict:
    """Entrez-date window for esearch (inclusive)."""
    return {
        "datetype": "edat",
        "mindate": start.strftime("%Y/%m/%d"),
        "maxdate": end.strftime("%Y/%m/%d"),
    }


def arxiv_query(categories: list[str], start: date, end: date) -> str:
    cats = " OR ".join(f"cat:{c}" for c in categories)
    return (
        f"({cats}) AND submittedDate:[{start.strftime('%Y%m%d')}0000 "
        f"TO {end.strftime('%Y%m%d')}2359]"
    )


def build_all(cfg: dict, start: date, end: date) -> dict:
    """Every generated query, for display and for the retrievers."""
    return {
        "pubmed_mine": interests_query(cfg["groups"]),
        "pubmed_major": journal_query(cfg["major"]["journals"], cfg["major"]["net"]),
        "pubmed_dates": date_params(start, end),
        "arxiv": arxiv_query(cfg["arxiv"]["categories"], start, end),
        "medrxiv": {
            "url": f"https://api.biorxiv.org/details/medrxiv/{start:%Y-%m-%d}/{end:%Y-%m-%d}/{{cursor}}",
            "categories": cfg["medrxiv"]["categories"],
        },
    }
