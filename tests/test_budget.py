import pytest

from litsearch.core.budget import Budget, Prices, PricesMissing
from litsearch.core.models import Paper
from litsearch.core.triage import Triager
from tests.helpers import make_spec

ENV = {"PRICE_INPUT_PER_MTOK": "1.0", "PRICE_OUTPUT_PER_MTOK": "5.0",
       "PRICE_CACHE_WRITE_PER_MTOK": "1.25", "PRICE_CACHE_READ_PER_MTOK": "0.1"}


def test_prices_come_from_env_and_are_required():
    with pytest.raises(PricesMissing):
        Prices.from_env({})
    p = Prices.from_env(ENV)
    assert p.inp == 1.0 and p.out == 5.0


def test_cost_formula_all_token_classes():
    p = Prices.from_env(ENV)
    cost = p.cost({"input_tokens": 1_000_000, "output_tokens": 200_000,
                   "cache_write_tokens": 100_000, "cache_read_tokens": 1_000_000})
    assert cost == pytest.approx(1.0 + 1.0 + 0.125 + 0.1)


def test_cache_prices_default_to_multiples_of_input():
    p = Prices.from_env({"PRICE_INPUT_PER_MTOK": "2", "PRICE_OUTPUT_PER_MTOK": "10"})
    assert p.cache_write == 2.5 and p.cache_read == pytest.approx(0.2)


def test_guard_blocks_when_worst_case_exceeds_limit(db):
    b = Budget(db, "2026-11-03", 0.10, Prices.from_env(ENV))
    assert b.allow(0.05)
    b.record({"input_tokens": 50_000, "output_tokens": 10_000})        # $0.05 + $0.05 = $0.10
    assert b.spent_today == pytest.approx(0.10)
    assert not b.allow(0.0001) and b.tripped


def test_spend_is_per_day_and_persisted(db):
    b1 = Budget(db, "2026-11-03", 0.10, Prices.from_env(ENV))
    b1.record({"input_tokens": 10_000, "output_tokens": 1000})
    assert Budget(db, "2026-11-03", 0.10, Prices.from_env(ENV)).spent_today == pytest.approx(0.015)
    assert Budget(db, "2026-11-04", 0.10, Prices.from_env(ENV)).spent_today == 0


# ---- guard wired into the triager, with a fake client
class _U:
    def __init__(self, i, o): self.input_tokens, self.output_tokens = i, o; self.cache_creation_input_tokens = 0; self.cache_read_input_tokens = 0
class _B:
    type = "text"
    def __init__(self, t): self.text = t
class _R:
    stop_reason = "end_turn"
    def __init__(self, text, i=1000, o=500): self.content = [_B(text)]; self.usage = _U(i, o)
class FakeClient:
    def __init__(self, texts): self.texts = list(texts); self.calls = 0; self.messages = self
    def create(self, **kw):
        self.calls += 1
        return _R(self.texts.pop(0))


def _papers(n):
    return [Paper("pubmed", f"Paper number {i} about large language models", "abstract " * 20, pmid=str(i), kw_score=n - i)
            for i in range(n)]


def _good(ids):
    import json
    return json.dumps([{"id": i, "relevance": 8, "track": "mine", "one_sentence_takeaway": "t", "why_relevant": "w"} for i in ids])


def test_triager_stops_when_budget_would_be_exceeded(db):
    prices = Prices.from_env({"PRICE_INPUT_PER_MTOK": "1000", "PRICE_OUTPUT_PER_MTOK": "5000"})  # absurdly pricey
    budget = Budget(db, "d", 0.10, prices)
    client = FakeClient([_good(["p1", "p2"])])
    t = Triager(client, "m", "", budget, make_spec(), 2)
    res = t.run(_papers(4))
    assert client.calls == 0 and len(res.unscored) == 4 and budget.tripped and not res.scored


def test_triager_trips_midway_and_keeps_scored_batches(db):
    # price so that exactly one 2-paper call fits the budget
    prices = Prices.from_env({"PRICE_INPUT_PER_MTOK": "1", "PRICE_OUTPUT_PER_MTOK": "1"})
    spec = make_spec()
    system = spec.system_prompt
    budget = Budget(db, "d", 0.0, prices)
    t = Triager(FakeClient([_good(["p1", "p2"])] * 3), "m", "", budget, spec, 2)
    one_call = prices.estimate(len(system) + 800, 4096)
    budget.limit = one_call * 1.1   # room for one call; the 2nd's worst case no longer fits after real spend
    res = t.run(_papers(6))
    assert len(res.scored) == 2 and len(res.unscored) == 4 and budget.tripped
    assert db.spent("d") > 0                      # real usage was logged
