"""Generate saved AI analyses for the sample libraries, so demo mode can show them without a key.

For each sample it retrieves the evidence exactly as the app does with the default settings, then runs
the real agents: one report for the sample's research question and one answer per suggested question.
Results are stored under demo/analyses/<fingerprint>.json. Existing files are skipped unless --force
is given. Needs OPENROUTER_API_KEY.

    python scripts/build_demo.py [--force]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from researchpilot.corpus import build_library  # noqa: E402
from researchpilot.demo import load_corpus, load_samples, load_saved_analysis, save_analysis  # noqa: E402
from researchpilot.llm import LLMClient, LLMError  # noqa: E402
from researchpilot.pipeline import evidence_fingerprint, run_qa, run_report  # noqa: E402
from researchpilot.retrieval import (  # noqa: E402
    DEFAULT_CHUNK_WORDS,
    DEFAULT_OVERLAP_WORDS,
    DEFAULT_PER_SOURCE,
    DEFAULT_TOP_K,
    EvidenceIndex,
    chunk_library,
)


def main(force: bool = False) -> int:
    llm = LLMClient(app_name="ResearchPilot AI")
    failures = 0
    for sample in load_samples():
        library = build_library(load_corpus(sample["corpus"])["papers"])
        by_sid = {s.sid: s for s in library}
        index = EvidenceIndex(chunk_library(library, DEFAULT_CHUNK_WORDS, DEFAULT_OVERLAP_WORDS))
        jobs = [("report", sample["question"])] + [("qa", q) for q in sample["qa_questions"]]
        for kind, question in jobs:
            hits = index.search(question, DEFAULT_TOP_K, DEFAULT_PER_SOURCE)
            fp = evidence_fingerprint(kind, question, hits)
            if load_saved_analysis(fp) and not force:
                print(f"skip   {sample['name']} {kind}: {question[:60]} ({fp})")
                continue
            try:
                if kind == "report":
                    analysis = run_report(llm, question, hits, by_sid, on_step=lambda msg: print(f"       {msg}"))
                else:
                    analysis = run_qa(llm, question, hits, by_sid)
            except LLMError as exc:
                failures += 1
                print(f"FAILED {sample['name']} {kind}: {exc}")
                continue
            path = save_analysis(analysis)
            print(f"saved  {sample['name']} {kind} -> {path.name} ({analysis['model']})")
    print(f"model calls made: {llm.calls_made}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(force="--force" in sys.argv))
