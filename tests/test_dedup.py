from litsearch.core.models import Paper
from litsearch.streams.literature import cap_candidates, merge_duplicates, split_notices

LONG = "Large language models for extracting HIV outcomes from clinical notes"


def test_same_pmid_and_doi_found(db):
    db.record(Paper("pubmed", LONG, pmid="111", doi="10.1/abc"), "sent")
    assert db.find(Paper("pubmed", "other title entirely here", pmid="111"))
    assert db.find(Paper("medrxiv", "different words altogether", doi="10.1/ABC"))   # case-insens DOI
    assert db.find(Paper("pubmed", "unrelated paper about something else", pmid="222")) is None


def test_arxiv_id_and_title_match(db):
    db.record(Paper("arxiv", LONG, arxiv_id="2401.00001"), "sent")
    assert db.find(Paper("arxiv", "x", arxiv_id="2401.00001"))
    assert db.find(Paper("pubmed", LONG.upper() + "."))          # normalised title match


def test_published_version_of_sent_preprint_is_duplicate(db):
    db.record(Paper("medrxiv", LONG, doi="10.1101/2026.01.01.1234"), "sent")
    # via title
    assert db.find(Paper("pubmed", LONG, pmid="999", doi="10.1000/journal"))
    # via preprint's published_doi pointer recorded on the stored preprint
    db.record(Paper("medrxiv", "Another quite different preprint title here", doi="10.1101/zzz",
                    published_doi="10.1000/pub2"), "sent")
    assert db.find(Paper("pubmed", "Retitled completely for the journal version", pmid="5", doi="10.1000/pub2"))


def test_new_preprint_pointing_at_sent_publication(db):
    db.record(Paper("pubmed", LONG, pmid="1", doi="10.1000/pub"), "sent")
    assert db.find(Paper("medrxiv", "Wholly different preprint title for the same work", doi="10.1101/p",
                         published_doi="10.1000/pub"))


def test_short_titles_do_not_collide(db):
    db.record(Paper("pubmed", "Editorial", pmid="1"), "sent")
    assert db.find(Paper("pubmed", "Editorial", pmid="2")) is None


def test_merge_duplicates_prefers_pubmed_and_drops_published_preprint():
    pub = Paper("pubmed", LONG, pmid="1", doi="10.1000/pub", abstract="")
    pre = Paper("medrxiv", "Totally different preprint title text", doi="10.1101/x", published_doi="10.1000/pub",
                abstract="preprint abstract", is_preprint=True)
    other = Paper("arxiv", "Unrelated arXiv paper about LLM agents in science", arxiv_id="2411.1")
    out = merge_duplicates([pre, other, pub])
    assert [p.source for p in out] == ["pubmed", "arxiv"]
    assert out[0].abstract == "preprint abstract"        # filled from the duplicate
    assert out[0].pmid == "1"


def test_merge_same_doi_across_sources_keeps_major_flag():
    a = Paper("pubmed", LONG, pmid="1", doi="10.1/a")
    b = Paper("pubmed", LONG, pmid="1", doi="10.1/a", is_major=True)
    out = merge_duplicates([a, b])
    assert len(out) == 1 and out[0].is_major


def test_notice_kept_only_for_sent_paper(db, cfg):
    db.record(Paper("pubmed", LONG, pmid="100"), "sent")
    db.record(Paper("pubmed", "Some screened-out paper title that is long enough", pmid="101"), "screened")
    err_sent = Paper("pubmed", "Erratum: " + LONG, pmid="200", pub_types=["Published Erratum"],
                     refs=[("ErratumFor", "100")])
    err_unsent = Paper("pubmed", "Erratum: Other", pmid="201", pub_types=["Published Erratum"],
                       refs=[("ErratumFor", "101")])
    normal = Paper("pubmed", "A normal paper", pmid="300", pub_types=["Journal Article"])
    keep, notices = split_notices([err_sent, err_unsent, normal], db, cfg["exclude"])
    assert keep == [normal]
    assert notices == [err_sent] and err_sent.is_notice


def test_cap_reserves_major_quota():
    majors = [Paper("pubmed", f"m{i}", is_major=True, kw_score=1) for i in range(30)]
    others = [Paper("pubmed", f"o{i}", kw_score=10 + i) for i in range(100)]
    out = cap_candidates(majors + others, 60, 20)
    assert len(out) == 60 and sum(p.is_major for p in out) == 20


def test_cap_fills_with_majors_when_few_others():
    majors = [Paper("pubmed", f"m{i}", is_major=True, kw_score=1) for i in range(30)]
    out = cap_candidates(majors + [Paper("pubmed", "o", kw_score=5)], 25, 10)
    assert len(out) == 25


def test_feedback_roundtrip(db):
    pid = db.record(Paper("pubmed", LONG, pmid="123"), "sent")
    pid2 = db.record(Paper("arxiv", "An arXiv paper on agents for science", arxiv_id="2401.5"), "sent")
    assert db.resolve_code("123")["id"] == pid
    assert db.resolve_code("arxiv:2401.5")["id"] == pid2
    assert db.resolve_code("999") is None
    db.add_feedback(pid, 1)
    db.add_feedback(pid, -1)       # latest verdict wins
    db.add_feedback(pid2, 1)
    fb = db.recent_feedback(10)
    assert (1, "An arXiv paper on agents for science") in fb and (-1, LONG) in fb and len(fb) == 2
