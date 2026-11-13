from litsearch import scoring
from litsearch.core.models import Paper


def test_llm_ehr_paper_scores_high_and_matches_group(cfg):
    p = Paper("pubmed", "Large language models for chart review of clinical notes",
              "We extract phenotypes from electronic health record notes with GPT-4.")
    s, matched = scoring.score_paper(p, cfg["groups"])
    assert "llm_clinical_extraction" in matched and s >= 10


def test_irrelevant_paper_scores_zero(cfg):
    p = Paper("arxiv", "Diffusion models for image synthesis", "We improve FID on ImageNet.")
    assert scoring.score_paper(p, cfg["groups"])[0] == 0


def test_all_clauses_required(cfg):
    p = Paper("arxiv", "A large language model for poetry", "Generates sonnets.")
    assert scoring.score_paper(p, cfg["groups"])[0] == 0


def test_acronym_case_sensitive_aids_verb_not_hiv(cfg):
    verb = Paper("pubmed", "A tool that aids diagnosis", "It aids in outcomes tracking of cohorts.")
    assert scoring.score_paper(verb, cfg["groups"])[0] == 0
    hiv = Paper("pubmed", "PrEP uptake outcomes", "HIV prevention outcomes in a cohort.")
    assert "hiv_care_continuum" in scoring.score_paper(hiv, cfg["groups"])[1]


def test_mesh_can_satisfy_a_clause(cfg):
    p = Paper("pubmed", "Opioid agonist therapy retention", "A cohort study.",
              mesh=["Opioid-Related Disorders", "Treatment Outcome"])
    assert "oud_outcomes_infections" in scoring.score_paper(p, cfg["groups"])[1]


def test_exclusion_by_type_and_title(cfg):
    ex = cfg["exclude"]
    assert scoring.is_excluded_type(Paper("pubmed", "x", pub_types=["Published Erratum"]), ex)
    assert scoring.is_excluded_type(Paper("pubmed", "Erratum: Something"), ex)
    assert scoring.is_excluded_type(Paper("pubmed", "Retraction notice: a paper"), ex)
    assert not scoring.is_excluded_type(Paper("pubmed", "A normal trial", pub_types=["Journal Article"]), ex)


def test_health_term_gate(cfg):
    terms = cfg["arxiv"]["health_terms"]
    assert scoring.has_health_terms(Paper("arxiv", "LLMs for clinical notes"), terms)
    assert not scoring.has_health_terms(Paper("arxiv", "Faster attention kernels", "GPU throughput."), terms)
