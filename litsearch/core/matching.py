"""Keyword matching shared by domain tagging and every stream's prefilter."""
from __future__ import annotations

import re
from functools import lru_cache


def _term_regex(term: str) -> str:
    esc = re.escape(term.strip()).replace(r"\*", r"\w*").replace(r"\ ", r"\s+").replace(r"\-", r"[-\s]?")
    body = rf"(?<![A-Za-z0-9]){esc}(?![A-Za-z0-9])"
    # Short all-caps acronyms (AIDS, LLM, OUD) are case-sensitive: avoids "aids" the verb.
    if term.isupper() and len(term) <= 6:
        return f"(?-i:{body})"
    return body


@lru_cache(maxsize=None)
def compile_terms(terms: tuple[str, ...]) -> re.Pattern | None:
    if not terms:
        return None
    return re.compile("|".join(_term_regex(t) for t in terms), re.IGNORECASE)


def distinct_hits(terms: tuple[str, ...], text: str) -> set[str]:
    rx = compile_terms(terms)
    return {m.group(0).lower() for m in rx.finditer(text)} if rx else set()
