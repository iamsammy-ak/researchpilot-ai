import json

from researchpilot.corpus import build_library
from researchpilot.demo import ANALYSES_DIR, load_corpus, load_samples, load_saved_analysis
from researchpilot.grounding import check_text
from researchpilot.litmap import build_map
from researchpilot.pipeline import evidence_fingerprint
from researchpilot.report import build_brief
from researchpilot.retrieval import (
    DEFAULT_CHUNK_WORDS,
    DEFAULT_OVERLAP_WORDS,
    DEFAULT_PER_SOURCE,
    DEFAULT_TOP_K,
    EvidenceIndex,
    chunk_library,
    extractive_answer,
)
from researchpilot.schemas import ResearchReport


def test_samples_and_corpora_are_complete():
    samples = load_samples()
    assert len(samples) == 2
    for sample in samples:
        corpus = load_corpus(sample["corpus"])
        papers = corpus["papers"]
        assert 40 <= len(papers) <= 60
        assert corpus["synthetic"] == any(p.get("kind") == "synthetic" for p in papers)
        assert "CC0" in corpus["license"] or corpus["synthetic"]
        for paper in papers:
            assert paper["title"] and paper["abstract"] and paper["published"][:4].isdigit()
        assert len({p["arxiv_id"] for p in papers}) == len(papers)
        assert sample["question"] and len(sample["qa_questions"]) == 3


def test_saved_analyses_match_current_evidence():
    """Saved analyses are optional; any that exist must still match the evidence the app retrieves."""
    expected = set()
    for sample in load_samples():
        index = EvidenceIndex(chunk_library(build_library(load_corpus(sample["corpus"])["papers"]),
                                            DEFAULT_CHUNK_WORDS, DEFAULT_OVERLAP_WORDS))
        for kind, question in [("report", sample["question"])] + [("qa", q) for q in sample["qa_questions"]]:
            expected.add(evidence_fingerprint(kind, question, index.search(question, DEFAULT_TOP_K,
                                                                            DEFAULT_PER_SOURCE)))
    for path in ANALYSES_DIR.glob("*.json"):
        assert path.stem in expected, f"{path.name} does not match any sample's evidence (stale fixture?)"
        analysis = load_saved_analysis(path.stem)
        if analysis["kind"] == "report":
            ResearchReport.model_validate(analysis["report"])
    assert load_saved_analysis("0000000000000000") is None


def test_brief_without_ai_has_every_computed_section():
    sample = load_samples()[0]
    library = build_library(load_corpus(sample["corpus"])["papers"])
    index = EvidenceIndex(chunk_library(library))
    hits = index.search(sample["question"], 8)
    brief = build_brief("RAG", sample["question"], library, build_map(library), hits,
                        extractive_answer(sample["question"], hits, index))
    for heading in ("Library at a glance", "Topic clusters", "Evidence retrieved", "Extractive answer", "References"):
        assert f"## {heading}" in brief
    assert "AI research report" not in brief and "no AI report" in brief
    assert brief.count("\n- [S") == len({h.chunk.sid for h in hits})  # one reference per cited source


def test_brief_with_report_includes_grounding_table():
    sample = load_samples()[0]
    library = build_library(load_corpus(sample["corpus"])["papers"])
    index = EvidenceIndex(chunk_library(library))
    report_md = "### Summary\n\nRGB evaluates noise robustness and negative rejection [S1]."
    brief = build_brief("RAG", "q", library, build_map(library), [], [], report_md, check_text(report_md, index),
                        "fake/model")
    assert "## Grounding check" in brief and "1 of 1 cited sentences" in brief and "fake/model" in brief
    json.dumps(brief)
