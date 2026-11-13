"""The `learned:` block of interests.yaml: soft, reviewable preferences distilled from your ratings.

Everything here affects RANKING only (a bounded adjustment on the 0-10 relevance scale); nothing is ever filtered out.
"""
from __future__ import annotations

import re
from fnmatch import fnmatch

from .matching import compile_terms

KINDS = ("term", "journal", "author", "institute", "mechanism", "sponsor")
ADJ_MIN, ADJ_MAX = -3.0, 2.0
EMPTY = {"preferences": "", "prefer": [], "deprioritize": []}


def normalise(learned: dict | None) -> dict:
    l = dict(EMPTY)
    l.update(learned or {})
    l["prefer"] = [r for r in (l.get("prefer") or []) if isinstance(r, dict)]
    l["deprioritize"] = [r for r in (l.get("deprioritize") or []) if isinstance(r, dict)]
    l["preferences"] = str(l.get("preferences") or "").strip()
    return l


def features(title: str, summary: str, ss: dict) -> dict:
    """What rules match on, from an item's title and its stream_specific dict (works for DB rows and live items).

    Term rules match the TITLE only: tune-ups learn terms from titles, so a rule applies to exactly what it was learned from
    (matching abstracts too would touch far more items than the evidence behind it).
    """
    authors = ss.get("authors") or ss.get("pis") or []
    if isinstance(authors, str):
        authors = [a.strip() for a in authors.replace(" et al.", "").split(",") if a.strip()]
    return {"text": title or "", "journal": ss.get("source_label") or ss.get("journal") or "",
            "authors": [str(a) for a in authors], "institutes": ss.get("institutes") or ([ss["institute"]] if ss.get("institute") else []),
            "mechanism": ss.get("mechanism") or "", "sponsor": ss.get("sponsor") or ss.get("institution") or ""}


def features_of(item) -> dict:
    return features(item.title, item.summary, item.stream_specific)


def rule_matches(rule: dict, f: dict) -> bool:
    kind, v = rule.get("kind"), str(rule.get("value", "")).strip()
    if not v:
        return False
    lv = v.lower()
    if kind == "term":
        rx = compile_terms((v,))
        return bool(rx and rx.search(f["text"]))
    if kind == "journal":
        return f["journal"].lower() == lv
    if kind == "author":
        return any(a.lower() == lv for a in f["authors"])
    if kind == "institute":
        return any(i.lower() == lv for i in f["institutes"])
    if kind == "mechanism":
        return bool(f["mechanism"]) and fnmatch(f["mechanism"].lower(), lv)
    if kind == "sponsor":
        return lv in f["sponsor"].lower()
    return False


def adjust(f: dict, learned: dict) -> tuple[float, list[str]]:
    """(adjustment, reasons): summed rule weights clipped to [ADJ_MIN, ADJ_MAX]; reasons like '↓ veterinary'."""
    total, why = 0.0, []
    for rule in learned["prefer"] + learned["deprioritize"]:
        if not rule_matches(rule, f):
            continue
        w = float(rule.get("weight") or 0)
        if w:
            total += w
            label = rule["value"] if rule.get("kind") == "term" else f"{rule['kind']}: {rule['value']}"
            why.append(("↑ " if w > 0 else "↓ ") + label)
    return max(ADJ_MIN, min(ADJ_MAX, total)), why


def prompt_block(learned: dict) -> str:
    if not learned["preferences"]:
        return ""
    return ("\n\nLEARNED READER TENDENCIES (distilled from the reader's past likes and dislikes; soft guidance for ranking only, "
            f"the profile above takes precedence):\n{learned['preferences']}")


def insert_into_prompt(system_prompt: str, learned: dict) -> str:
    """Put the learned paragraph before the OUTPUT instructions (so the format rules stay last); unchanged when empty."""
    block = prompt_block(learned)
    if not block:
        return system_prompt
    m = re.search(r"\n\nOUTPUT\n", system_prompt)
    return system_prompt[:m.start()] + block + system_prompt[m.start():] if m else system_prompt + block
