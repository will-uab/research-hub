"""Select, order and render the digest."""
from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .core.models import Paper

TEMPLATES = Path(__file__).resolve().parent.parent / "templates"


@dataclass
class Stats:
    retrieved: int = 0
    by_source: str = ""
    screened: int = 0
    sent: int = 0
    cost_run: float = 0.0
    cost_today: float = 0.0
    budget: float = 0.0
    dry_run: bool = False


@dataclass
class Digest:
    subject: str
    html: str
    text: str
    mine: list[Paper] = field(default_factory=list)
    major: list[Paper] = field(default_factory=list)
    notices: list[Paper] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.mine or self.major or self.notices)

    @property
    def sent_papers(self) -> list[Paper]:
        return self.mine + self.major + self.notices


def select(scored: list[Paper], unscored: list[Paper], cfg: dict) -> tuple[list[Paper], list[Paper]]:
    """Apply thresholds and caps. Unscored (budget-guard) items are keyword-ranked, no threshold."""
    d = cfg["digest"]
    mine = [p for p in scored if p.track == "mine" and p.relevance >= d["min_relevance_mine"]]
    major = [p for p in scored if p.track == "major" and p.relevance >= d["min_relevance_major"]]
    mine.sort(key=lambda p: (-p.relevance, -p.kw_score))
    major.sort(key=lambda p: (-p.relevance, -p.kw_score))
    mine, major = mine[:d["max_mine"]], major[:d["max_major"]]

    cap = d["fallback_max_items"]
    ranked = sorted(unscored, key=lambda p: p.kw_score, reverse=True)
    mine += [p for p in ranked if not p.is_major][:cap]
    major += [p for p in ranked if p.is_major][:min(cap, d["max_major"])]
    return mine, major


def _wrap(text: str) -> str:
    return "\n".join(textwrap.fill(par, 88, initial_indent="   ", subsequent_indent="   ")
                     for par in text.split("\n") if par.strip())


def render(date_str: str, mine: list[Paper], major: list[Paper], notices: list[Paper], stats: Stats,
           errors: list[str], fallback_reason: str = "", n_unscored: int = 0,
           back_link: str | None = None) -> Digest:
    n, m = len(mine) + len(notices), len(major)
    empty = not (mine or major or notices)
    subject = (f"Lit digest {date_str}: nothing notable today" if empty
               else f"Lit digest {date_str}: {n} for you, {m} major")
    ctx = dict(subject=subject, date=date_str, mine=mine, major=major, notices=notices, stats=stats,
               errors=errors, empty=empty, fallback=bool(fallback_reason), fallback_reason=fallback_reason,
               n_unscored=n_unscored, back_link=back_link)
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html", "j2"],
                      default_for_string=False), trim_blocks=True, lstrip_blocks=True)
    env.filters["wrap"] = _wrap
    # .html.j2 should autoescape, .txt.j2 should not
    html = env.get_template("digest.html.j2").render(**ctx)
    txt_env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=False, trim_blocks=True,
                          lstrip_blocks=True)
    txt_env.filters["wrap"] = _wrap
    text = txt_env.get_template("digest.txt.j2").render(**ctx)
    return Digest(subject, html, text, mine, major, notices)
