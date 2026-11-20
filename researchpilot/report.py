"""Downloadable Markdown research brief: computed library map, evidence, extractive answer and, when
available, the AI report with its grounding check and a reference list."""

from __future__ import annotations

from datetime import date

from .corpus import Source
from .grounding import VERDICTS, GroundingResult, parse_citations
from .litmap import LiteratureMap, cluster_table
from .retrieval import ExtractiveSentence, Hit


def _cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _clip(text: str, limit: int = 280) -> str:
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


def reference(source: Source) -> str:
    year = f" ({source.year})" if source.year else ""
    url = f" {source.url}" if source.url else ""
    return f"[{source.sid}] {source.author_line(3)}{year}. {source.title}.{url}"


def build_brief(
    library_name: str,
    question: str,
    sources: list[Source],
    lit: LiteratureMap,
    hits: list[Hit],
    extract: list[ExtractiveSentence],
    report_md: str | None = None,
    grounding: GroundingResult | None = None,
    model: str | None = None,
) -> str:
    years = [s.year for s in sources if s.year]
    lines = [
        f"# ResearchPilot brief: {question}",
        "",
        f"_Generated {date.today().isoformat()} · Library: {library_name} ({len(sources)} sources)"
        + (f" · AI report by `{model}`" if model else " · computed sections only, no AI report")
        + "_",
        "",
        "## Library at a glance",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Sources | {len(sources)} |",
        f"| Publication years | {min(years)}–{max(years)} |" if years else "| Publication years | — |",
        f"| Topic clusters | {lit.k}"
        + (f" (silhouette {lit.silhouette:.2f})" if lit.silhouette is not None else "")
        + " |",
        "",
    ]
    if lit.k > 1:
        lines += ["## Topic clusters", "", "| # | Cluster | Papers | Years | Central paper |",
                  "|---:|---|---:|---|---|"]
        for row in cluster_table(sources, lit).itertuples():
            lines.append(f"| {row.cluster + 1} | {_cell(row.label)} | {row.papers} | {row.years} | "
                         f"{_cell(row.representative)} |")
        lines.append("")

    lines += ["## Evidence retrieved for the question", "", "| Rank | Citation | BM25 | Passage |",
              "|---:|---|---:|---|"]
    lines += [f"| {h.rank} | [{h.label}] | {h.score:.2f} | {_cell(_clip(h.chunk.text))} |" for h in hits]
    if not hits:
        lines.append("| — | — | — | No passage matched the question. |")
    lines += ["", "## Extractive answer", "", "_Sentences quoted verbatim from the sources, ranked by how many of the "
              "question's terms they cover._", ""]
    lines += [f"- {s.text} [{s.label}]" for s in extract] or ["- No sentence matched the question."]
    lines.append("")

    if report_md:
        lines += ["## AI research report", "", report_md.strip(), ""]
    if grounding is not None and grounding.checks:
        share = grounding.grounded_share
        lines += [
            "## Grounding check",
            "",
            f"{grounding.count('supported')} of {grounding.cited} cited sentences are supported by the passages "
            f"they cite" + (f" ({share:.0%})." if share is not None else "."),
            "",
            "| # | Sentence | Support | Verdict | Note |",
            "|---:|---|---:|---|---|",
        ]
        for i, check in enumerate(grounding.checks, start=1):
            score = f"{check.score:.0%}" if check.score is not None else "—"
            lines.append(f"| {i} | {_cell(check.sentence)} | {score} | {VERDICTS[check.verdict]} | "
                         f"{_cell(check.note) or ' '} |")
        lines.append("")

    cited = {h.chunk.sid for h in hits}
    for text in filter(None, [report_md]):
        cited |= {c.sid for c in parse_citations(text)}
    by_sid = {s.sid: s for s in sources}
    refs = [by_sid[sid] for sid in sorted(cited, key=lambda x: int(x[1:])) if sid in by_sid]
    if refs:
        lines += ["## References", ""] + [f"- {reference(s)}" for s in refs] + [""]
    lines += [
        "---",
        "_ResearchPilot is a research aid. Retrieval is lexical (BM25) and the grounding check measures word "
        "overlap, so read the cited passages before relying on any claim._",
    ]
    return "\n".join(lines)
