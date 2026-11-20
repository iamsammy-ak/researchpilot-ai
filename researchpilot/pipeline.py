"""Glue between retrieval and the AI agents, shared by the app and scripts/build_demo.py.

Saved analyses are keyed by a fingerprint of the question and the exact evidence passages, so a
saved answer is only shown when it was written from the same evidence the app retrieves now.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime, timezone

from .agents import SupportsLLM, answer_question, write_report
from .corpus import Source
from .retrieval import Hit
from .schemas import Finding, ResearchReport
from .text import normalise_space


def evidence_fingerprint(kind: str, question: str, hits: list[Hit]) -> str:
    payload = {
        "kind": kind,
        "question": normalise_space(question).lower(),
        "evidence": [[h.label, h.chunk.text] for h in hits],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_report(
    llm: SupportsLLM,
    question: str,
    hits: list[Hit],
    sources: dict[str, Source],
    on_step: Callable[[str], object] | None = None,
) -> dict:
    """One model call: the structured report. Returns a JSON-safe dict."""
    step = on_step or (lambda _msg: None)
    step(f"Report writer: reading {len(hits)} evidence passages and drafting the report…")
    report = write_report(llm, question, hits, sources)
    return {
        "kind": "report",
        "fingerprint": evidence_fingerprint("report", question, hits),
        "question": question,
        "evidence": [h.label for h in hits],
        "report": report.model_dump(),
        "model": getattr(llm, "last_model", None),
        "generated_at": _now(),
    }


def run_qa(llm: SupportsLLM, question: str, hits: list[Hit], sources: dict[str, Source]) -> dict:
    """One model call: a short cited answer."""
    return {
        "kind": "qa",
        "fingerprint": evidence_fingerprint("qa", question, hits),
        "question": question,
        "evidence": [h.label for h in hits],
        "answer": answer_question(llm, question, hits, sources),
        "model": getattr(llm, "last_model", None),
        "generated_at": _now(),
    }


def _with_citations(text: str, labels: list[str]) -> str:
    """Append citation labels to a statement that does not already carry them inline."""
    if not labels or "[S" in text:
        return text
    return f"{text.rstrip()} " + " ".join(f"[{label}]" for label in labels)


def finding_line(finding: Finding) -> str:
    return _with_citations(finding.statement, finding.citations)


def report_markdown(report: ResearchReport) -> str:
    """Render the structured report as Markdown; this exact text is what the grounding check reads."""
    lines = ["### Question", "", report.question, ""]
    if report.summary:
        lines += ["### Summary", "", report.summary, ""]
    if report.key_findings:
        lines += ["### Key findings", ""]
        lines += [f"- {finding_line(f)}" for f in report.key_findings]
        lines.append("")
    for title, items in (
        ("Methods overview", report.methods_overview),
        ("Disagreements", report.disagreements),
        ("Gaps", report.gaps),
        ("Future work", report.future_work),
    ):
        if items:
            lines += [f"### {title}", ""] + [f"- {item}" for item in items] + [""]
    return "\n".join(lines).strip() + "\n"
