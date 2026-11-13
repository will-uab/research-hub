"""Domain tagging: keyword/MeSH first; LLM triage confirms or adds in the same call."""
from __future__ import annotations

from .matching import distinct_hits


class DomainTagger:
    """A domain applies if: a title keyword hit, or >=2 distinct keyword hits in the text, or a MeSH hit."""

    def __init__(self, domains_cfg: dict):
        self.ids = list(domains_cfg)
        self.labels = {d: c.get("label", d) for d, c in domains_cfg.items()}
        self._kw = {d: tuple(c.get("keywords", [])) for d, c in domains_cfg.items()}
        self._mesh = {d: {m.lower() for m in c.get("mesh", [])} for d, c in domains_cfg.items()}

    def tag(self, title: str, text: str = "", mesh=()) -> list[str]:
        mesh_l = {m.lower() for m in mesh}
        out = []
        for d in self.ids:
            if (distinct_hits(self._kw[d], title or "")
                    or len(distinct_hits(self._kw[d], f"{title}\n{text}")) >= 2
                    or (mesh_l & self._mesh[d])):
                out.append(d)
        return out

    def valid(self, ids) -> list[str]:
        """Filter an LLM-supplied list down to known domain ids, in config order."""
        want = {str(i).strip() for i in ids or []}
        return [d for d in self.ids if d in want]
