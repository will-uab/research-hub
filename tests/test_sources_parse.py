from litsearch.sources.arxiv import parse_feed
from litsearch.sources.medrxiv import parse_record
from litsearch.sources.pubmed import parse_article
import xml.etree.ElementTree as ET

PUBMED = """<PubmedArticle><MedlineCitation><PMID>42970790</PMID><Article>
<Journal><ISOAbbreviation>JMIR Med Inform</ISOAbbreviation><Title>JMIR Medical Informatics</Title>
<JournalIssue><PubDate><Year>2026</Year><Month>Nov</Month><Day>2</Day></PubDate></JournalIssue></Journal>
<ArticleTitle>LLMs for <i>chart</i> review.</ArticleTitle>
<Abstract><AbstractText Label="BACKGROUND">First part.</AbstractText><AbstractText Label="RESULTS">Second <b>part</b>.</AbstractText></Abstract>
<AuthorList><Author><LastName>Song</LastName><Initials>J</Initials></Author><Author><CollectiveName>The Group</CollectiveName></Author></AuthorList>
<PublicationTypeList><PublicationType>Journal Article</PublicationType><PublicationType>Preprint</PublicationType></PublicationTypeList>
</Article><MeshHeadingList><MeshHeading><DescriptorName>Humans</DescriptorName></MeshHeading></MeshHeadingList>
<CommentsCorrectionsList><CommentsCorrections RefType="ErratumFor"><PMID>111</PMID></CommentsCorrections></CommentsCorrectionsList>
</MedlineCitation><PubmedData><ArticleIdList><ArticleId IdType="pubmed">42970790</ArticleId>
<ArticleId IdType="doi">10.2196/ABC</ArticleId></ArticleIdList></PubmedData></PubmedArticle>"""


def test_pubmed_parse():
    p = parse_article(ET.fromstring(PUBMED))
    assert p.pmid == "42970790" and p.doi == "10.2196/abc"
    assert p.title == "LLMs for chart review"
    assert p.abstract == "BACKGROUND: First part. RESULTS: Second part."
    assert p.authors == ["Song J", "The Group"] and p.journal == "JMIR Med Inform"
    assert p.date == "2026-11-02" and p.is_preprint and p.mesh == ["Humans"]
    assert p.refs == [("ErratumFor", "111")]


def test_medrxiv_parse_and_published_pointer():
    rec = {"doi": "10.1101/2026.10.30.1", "title": "A  title\n", "authors": "Smith, J. A.; Doe, B.",
           "date": "2026-10-31", "abstract": "Abs  text", "published": "10.1000/xyz", "category": "hiv/aids"}
    p = parse_record(rec)
    assert p.authors == ["Smith JA", "Doe B"] and p.title == "A title" and p.abstract == "Abs text"
    assert p.published_doi == "10.1000/xyz" and p.is_preprint
    assert parse_record({**rec, "published": "NA"}).published_doi is None


ARXIV = """<feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
 xmlns:arxiv="http://arxiv.org/schemas/atom"><opensearch:totalResults>1</opensearch:totalResults>
<entry><id>http://arxiv.org/abs/2411.01234v2</id><published>2026-11-01T10:00:00Z</published>
<title>LLM agents
 for EHR</title><summary> Abstract
 text. </summary><author><name>A B</name></author><author><name>C D</name></author>
<arxiv:doi>10.1000/pub</arxiv:doi></entry></feed>"""


def test_arxiv_parse_strips_version_and_whitespace():
    papers, total = parse_feed(ARXIV.encode())
    p = papers[0]
    assert total == 1 and p.arxiv_id == "2411.01234" and p.title == "LLM agents for EHR"
    assert p.abstract == "Abstract text." and p.authors == ["A B", "C D"] and p.published_doi == "10.1000/pub"
    assert p.key == "arxiv:2411.01234" and p.url == "https://arxiv.org/abs/2411.01234"
