from datetime import date

from litsearch import queries


def test_clause_mixes_mesh_and_tiab():
    q = queries.clause_query({"mesh": ["HIV Infections"], "tiab": ["HIV", "pre-exposure prophylaxis", "phenotyp*"]})
    assert '"HIV Infections"[MeSH Terms]' in q
    assert "HIV[tiab]" in q                          # single token unquoted
    assert '"pre-exposure prophylaxis"[tiab]' in q   # phrase quoted
    assert "phenotyp*[tiab]" in q                    # wildcard preserved
    assert q.startswith("(") and q.endswith(")")


def test_group_ands_clauses_and_interests_or_groups(cfg):
    g = cfg["groups"][0]
    gq = queries.group_query(g)
    assert gq.count(") AND (") == len(g["all"]) - 1
    full = queries.interests_query(cfg["groups"])
    assert full.count(") OR ((") >= len(cfg["groups"]) - 1
    for grp in cfg["groups"]:
        assert queries.group_query(grp) in full


def test_journal_query_splits_unfiltered_and_net(cfg):
    q = queries.journal_query(cfg["major"]["journals"], cfg["major"]["net"])
    assert '"NEJM AI"[ta]' in q and '"N Engl J Med"[ta]' in q
    # unfiltered journals sit outside the topic-net AND; net journals inside it
    head, _, tail = q.partition(" OR ((")
    assert '"NEJM AI"[ta]' in head and '"N Engl J Med"[ta]' not in head
    assert '"N Engl J Med"[ta]' in tail and "[MeSH Terms]" in tail
    assert len(cfg["major"]["journals"]) == 17


def test_date_params_inclusive_entrez_dates():
    p = queries.date_params(date(2026, 11, 1), date(2026, 11, 3))
    assert p == {"datetype": "edat", "mindate": "2026/11/01", "maxdate": "2026/11/03"}


def test_arxiv_query(cfg):
    q = queries.arxiv_query(["cs.CL", "stat.AP"], date(2026, 11, 1), date(2026, 11, 3))
    assert q == "(cat:cs.CL OR cat:stat.AP) AND submittedDate:[202611010000 TO 202611032359]"


def test_empty_clause_rejected():
    import pytest
    with pytest.raises(ValueError):
        queries.clause_query({})
