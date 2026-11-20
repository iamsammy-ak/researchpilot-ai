import pytest

from researchpilot.corpus import Source, build_library
from researchpilot.demo import load_corpus, load_samples
from researchpilot.grounding import (
    Citation,
    check_text,
    markdown_sentences,
    negated_terms,
    numbers_in,
    parse_citations,
)
from researchpilot.retrieval import EvidenceIndex, chunk_library

SOURCES = [
    Source("S1", "Chunking study", "Under the tested configuration, cluster-based chunking did not outperform "
           "simpler fixed-size strategies on academic theses."),
    Source("S2", "Hallucination benchmark", "HaluBench contains 15k samples sourced from real-world domains "
           "such as finance and medicine."),
    Source("S3", "Battery features", "The pipeline engineers 30 features from charge voltage curves and "
           "reaches an error of 0.45% on fast-charged cells."),
]
INDEX = EvidenceIndex(chunk_library(SOURCES))


def _one(text: str):
    result = check_text(text, INDEX)
    assert len(result.checks) == 1
    return result.checks[0]


def test_parsing_citations_sentences_and_numbers():
    assert parse_citations("x [S3] y [S3 p.2] z [S5; S7 p. 1] w [S9, page 4]") == [
        Citation("S3"), Citation("S3", 2), Citation("S5"), Citation("S7", 1), Citation("S9", 4)
    ]
    assert parse_citations("no citations [here] or [1]") == []
    md = "## Heading\n\n- **Bold** claim one. [S1]\n- Claim two with a [link](http://x.org) [S2].\n| a | b |\n"
    assert markdown_sentences(md) == ["Bold claim one. [S1]", "Claim two with a link [S2]."]
    assert numbers_in("1,000 cells, 0,5 V, 12.50% and 3.0 h") == {"1000", "0.5", "12.5", "3"}


def test_faithful_claim_is_supported_but_an_invented_number_is_not():
    check = _one("The pipeline engineers 30 features from charge voltage curves [S3].")
    assert check.verdict == "supported" and check.score == pytest.approx(1.0)
    invented = _one("The pipeline engineers 45 features from charge voltage curves [S3].")
    assert invented.verdict == "weak" and invented.missing_numbers == ("45",)


def test_wrong_source_is_flagged_and_better_source_suggested():
    check = _one("HaluBench contains 15k samples sourced from real-world domains [S3].")
    assert check.verdict == "unsupported"
    assert check.missing_numbers == ("15",)
    assert check.suggestion == "S2"


def test_flipped_polarity_is_detected_both_ways():
    flipped = _one("Cluster-based chunking outperformed simpler fixed-size strategies on academic theses [S1].")
    assert flipped.verdict == "weak" and "did not outperform" in flipped.conflict
    same = _one("Cluster-based chunking did not outperform fixed-size strategies on academic theses [S1].")
    assert same.verdict == "supported" and same.conflict is None
    assert negated_terms("models without parameter incorporation") == set()
    assert negated_terms("methods fail to adequately account for drift") == {"account"}


def test_unknown_sources_and_uncited_claims():
    result = check_text("Sodium cells fade slower than lithium cells [S42]. "
                        "Field data from vehicles is still scarce compared with lab data. Short one.", INDEX)
    assert [c.verdict for c in result.checks] == ["unknown_source", "uncited"]
    assert result.checks[0].unknown == ("S42",)
    assert result.grounded_share == 0.0
    assert check_text("", INDEX).grounded_share is None


def test_page_citations_resolve_to_the_right_page():
    pdf = Source("", "Paper", "", pages=("Page one discusses solid electrolyte interphase growth.",
                                         "Page two reports lithium plating at low temperature."))
    index = EvidenceIndex(chunk_library(build_library([], [pdf])))
    right = check_text("Lithium plating appears at low temperature [S1 p.2].", index).checks[0]
    wrong = check_text("Lithium plating appears at low temperature [S1 p.1].", index).checks[0]
    missing = check_text("Lithium plating appears at low temperature [S1 p.9].", index).checks[0]
    assert right.verdict == "supported" and wrong.verdict == "unsupported"
    assert wrong.suggestion == "S1 p.2"
    assert missing.verdict == "unknown_source"


@pytest.mark.parametrize("sample", load_samples(), ids=lambda s: s["slug"])
def test_bundled_grounding_examples_exercise_every_verdict(sample):
    """The demo paragraphs must keep showing what the checker catches, against the bundled corpora."""
    library = build_library(load_corpus(sample["corpus"])["papers"])
    result = check_text(sample["grounding_example"], EvidenceIndex(chunk_library(library)))
    verdicts = [c.verdict for c in result.checks]
    assert {"supported", "weak", "unsupported", "unknown_source", "uncited"} <= set(verdicts)
    assert any(c.conflict for c in result.checks), "the flipped-claim example stopped being detected"
    assert any(c.suggestion for c in result.checks), "the wrong-source example lost its better match"
