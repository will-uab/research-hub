"""Shared test helpers."""
from litsearch.core.triage import TriageSpec
from litsearch.streams.literature import TRACKS, format_paper, system_prompt

DOMAINS = {"llm_ai": "LLM / AI", "infectious_diseases": "Infectious Diseases", "hiv_addiction": "HIV + Addiction"}


def make_spec(feedback=(), words=250):
    return TriageSpec(system_prompt=system_prompt("profile", list(feedback), DOMAINS),
                      format_item=lambda p, pid: format_paper(p, pid, words),
                      header="Papers:", tracks=TRACKS, domain_ids=tuple(DOMAINS))


import json
import re
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


class ScriptedClient:
    """Fake Anthropic client: answers every [pN] block in the user message with a fixed verdict."""

    def __init__(self, relevance=8, domains=(), fit="fits well", takeaway="funds X", per_id=None, track=None):
        self.relevance, self.domains, self.fit, self.takeaway, self.track = relevance, list(domains), fit, takeaway, track
        self.last_system = ""
        self.per_id = per_id or {}
        self.calls = 0
        self.messages = self
        self.last_user = ""

    def create(self, **kw):
        from tests.test_budget import _B, _R
        self.calls += 1
        user = kw["messages"][0]["content"]
        self.last_user = user
        self.last_system = kw["system"][0]["text"]
        ids = re.findall(r"^\[(p\d+)\]", user, flags=re.M)
        out = [{"id": i, "relevance": self.per_id.get(i, self.relevance), "domains": self.domains,
                "one_sentence_takeaway": self.takeaway, "why_relevant": self.fit,
                **({"track": self.track} if self.track else {})} for i in ids]
        return _R(json.dumps(out), i=1500, o=120 * len(ids))


def settings_for(tmp_path, **env):
    from litsearch.core.config import Settings
    base = {"DAILY_BUDGET_USD": "0.15", "ANTHROPIC_API_KEY": "k", "PRICE_INPUT_PER_MTOK": "0.1",
            "PRICE_OUTPUT_PER_MTOK": "0.5"}
    return Settings(env={**base, **env}, root=tmp_path, has_env_file=True)


def patch_client(monkeypatch, client):
    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", lambda **k: client)
