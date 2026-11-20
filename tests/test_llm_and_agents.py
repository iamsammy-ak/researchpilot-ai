"""LLM parsing and agent tests. No network: a fake client stands in for OpenRouter."""

import json

import pytest

from researchpilot.agents import answer_question, evidence_block, write_report
from researchpilot.config import Settings
from researchpilot.corpus import Source
from researchpilot.grounding import check_text
from researchpilot.llm import LLMClient, LLMError, clean_text, extract_json
from researchpilot.pipeline import evidence_fingerprint, report_markdown, run_qa, run_report
from researchpilot.retrieval import EvidenceIndex, chunk_library
from researchpilot.schemas import ResearchReport, normalise_labels

SOURCES = [
    Source("S1", "RGB benchmark", "RGB tests noise robustness, negative rejection, information integration and "
           "counterfactual robustness.", authors=("Jiawei Chen", "Hongyu Lin", "Xianpei Han"), published="2023-09-04"),
    Source("S2", "Chunk size study", "Larger chunks add noise and increase computational cost."),
]
INDEX = EvidenceIndex(chunk_library(SOURCES))
HITS = INDEX.search("noise robustness chunks", k=4)
BY_SID = {s.sid: s for s in SOURCES}

REPORT = {
    "question": "How is RAG evaluated?",
    "summary": "Benchmarks test robustness to noise [S1].",
    "key_findings": [
        {"statement": "RGB tests noise robustness and negative rejection [S1].", "citations": "[S1]",
         "strength": "Strong"},
        {"statement": "Larger chunks add noise and increase computational cost", "citations": ["S2 p. 1", "s2"]},
        "A plain string finding",
    ],
    "methods_overview": [{"method": "Benchmarking", "detail": "four abilities [S1]"}],
    "disagreements": "None in the evidence.",
    "gaps": None,
    "future_work": ["Test chunk sizes on more domains"],
}


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.last_model = "fake/model"
        self.prompts = []

    def chat(self, system, user, **kwargs):
        self.prompts.append(user)
        return self.replies.pop(0)

    def complete_json(self, system, user, schema, **kwargs):
        return schema.model_validate(extract_json(self.chat(system, user)))


def test_extract_json_and_clean_text_variants():
    assert extract_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert extract_json('<think>hmm {"no": 1}</think>Sure! {"a": 3} hope that helps') == {"a": 3}
    with pytest.raises(ValueError):
        extract_json("[1, 2]")
    assert clean_text("<think>plan</think>\n```markdown\nAnswer [S1].\n```") == "Answer [S1]."


def test_report_schema_coerces_free_model_quirks():
    report = ResearchReport.model_validate(REPORT)
    first, second, third = report.key_findings
    assert first.citations == ["S1"] and first.strength == "strong"
    assert second.citations == ["S2 p.1", "S2"] and second.strength == "moderate"
    assert third.statement == "A plain string finding" and third.citations == []
    assert report.methods_overview == ["Benchmarking — four abilities [S1]"]
    assert report.disagreements == ["None in the evidence."] and report.gaps == []
    assert normalise_labels("see [S3], S12 page 4 and S3") == ["S3", "S12 p.4"]


def test_report_markdown_adds_missing_citations_and_feeds_the_grounding_check():
    md = report_markdown(ResearchReport.model_validate(REPORT))
    assert "- Larger chunks add noise and increase computational cost [S2 p.1] [S2]" in md
    assert md.count("[S1]") >= 3  # inline citations are not duplicated
    result = check_text(md, INDEX)
    supported = [c.claim for c in result.checks if c.verdict == "supported"]
    assert any(c.startswith("RGB tests noise robustness") for c in supported)


def test_agents_send_only_labelled_evidence_and_parse_replies():
    fake = FakeLLM([json.dumps(REPORT), "<think>x</think>RGB tests noise robustness [S1]."])
    report = write_report(fake, "How is RAG evaluated?", HITS, BY_SID)
    assert report.question == "How is RAG evaluated?"
    assert "[S1] RGB benchmark (Jiawei Chen, Hongyu Lin et al., 2023)" in fake.prompts[0]
    assert answer_question(fake, "What does RGB test?", HITS, BY_SID) == "RGB tests noise robustness [S1]."
    assert evidence_block(HITS, BY_SID).count("[S") == len(HITS)


def test_pipeline_results_are_serialisable_and_keyed_by_evidence():
    fake = FakeLLM([json.dumps(REPORT), "Answer [S1]."])
    steps = []
    report = run_report(fake, "How is RAG evaluated?", HITS, BY_SID, on_step=steps.append)
    qa = run_qa(fake, "What does RGB test?", HITS, BY_SID)
    json.dumps([report, qa])
    assert report["fingerprint"] == evidence_fingerprint("report", "  how is RAG   evaluated? ", HITS)
    assert report["evidence"] == [h.label for h in HITS] and len(steps) == 1
    assert qa["fingerprint"] != evidence_fingerprint("qa", "What does RGB test?", HITS[:1])
    assert report["fingerprint"] != evidence_fingerprint("qa", "How is RAG evaluated?", HITS)


def test_client_needs_a_key_repairs_json_once_and_caches(tmp_path, monkeypatch):
    with pytest.raises(LLMError):
        LLMClient(Settings(api_key=None))
    client = LLMClient(Settings(api_key="test", models=("fake/model",)), cache_dir=tmp_path)
    replies = iter(["not json at all", json.dumps({"question": "q", "summary": "s [S1]."})])
    monkeypatch.setattr(client, "_call_with_fallback", lambda *args, **kw: (next(replies), "fake/model"))
    result = client.complete_json("sys", "user", ResearchReport)
    assert result.summary == "s [S1]."
    assert len(list(tmp_path.glob("*.json"))) == 1
    calls = []
    monkeypatch.setattr(client, "_call_with_fallback", lambda *a, **k: calls.append(1) or ("hello", "fake/model"))
    assert client.chat("s", "u") == client.chat("s", "u") == "hello"
    assert len(calls) == 1
