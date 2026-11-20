"""The two AI agents. Both see only retrieved evidence passages, each labelled with its citation:

- the report writer returns a structured ResearchReport (JSON, validated with Pydantic);
- the Q&A agent answers a question in a short cited paragraph.

Neither computes anything: retrieval, ranking and every number shown in the app come from Python.
"""

from __future__ import annotations

from typing import Protocol

from .corpus import Source
from .llm import clean_text
from .retrieval import Hit
from .schemas import ResearchReport


class SupportsLLM(Protocol):
    def complete_json(self, system: str, user: str, schema: type, **kwargs): ...
    def chat(self, system: str, user: str, **kwargs) -> str: ...


EVIDENCE_RULES = """Rules:
- Use only the evidence passages provided. Do not add outside knowledge.
- Cite every claim with the passage labels in square brackets exactly as written, e.g. [S3] or [S3 p.2].
  Cite only labels that appear in the evidence, and only for what that passage actually says.
- Never invent numbers, datasets, model names or results. Quote a figure only if it appears in the cited passage.
- If the evidence is thin, one-sided or silent on a point, say so instead of filling the gap.
- One claim per sentence, in plain, specific English."""

REPORT_SYSTEM = f"""You are a careful research analyst who writes structured literature reports.
{EVIDENCE_RULES}"""

QA_SYSTEM = f"""You answer questions about a research library.
{EVIDENCE_RULES}
Answer in 80-180 words of Markdown prose without headings."""


def evidence_block(hits: list[Hit], sources: dict[str, Source]) -> str:
    """The evidence pack sent to the model: label, source metadata, then the passage text."""
    blocks = []
    for hit in hits:
        source = sources.get(hit.chunk.sid)
        meta = ""
        if source is not None:
            year = f", {source.year}" if source.year else ""
            meta = f" {source.title} ({source.author_line(2)}{year})"
        blocks.append(f"[{hit.label}]{meta}\n{hit.chunk.text}")
    return "\n\n".join(blocks)


def write_report(llm: SupportsLLM, question: str, hits: list[Hit], sources: dict[str, Source]) -> ResearchReport:
    user = f"""Research question: {question}

Evidence passages:
{evidence_block(hits, sources)}

Write the report:
- question: the research question, restated precisely
- summary: 2-3 sentences that answer the question, with citations
- key_findings: 4-6 findings; each has a statement with inline citations, the list of cited labels,
  and a strength (strong = several passages agree, moderate = one clear passage, limited = indirect)
- methods_overview: 3-5 items describing the methods, datasets or metrics the evidence uses, with citations
- disagreements: 1-3 places where passages conflict or report different outcomes, with citations
  (if none, one item saying the evidence shows no direct disagreement)
- gaps: 2-4 things the evidence does not cover
- future_work: 2-4 concrete next studies or experiments that would close those gaps"""
    return llm.complete_json(REPORT_SYSTEM, user, ResearchReport, max_tokens=6000)


def answer_question(llm: SupportsLLM, question: str, hits: list[Hit], sources: dict[str, Source]) -> str:
    user = f"Question: {question}\n\nEvidence passages:\n{evidence_block(hits, sources)}"
    return clean_text(llm.chat(QA_SYSTEM, user, temperature=0.2, max_tokens=900))
