from researchpilot.corpus import Source, build_library
from researchpilot.retrieval import EvidenceIndex, chunk_library, chunk_text, extractive_answer
from researchpilot.text import split_sentences, stem, tokenize


def _source(sid: str, title: str, text: str, pages: tuple[str, ...] = ()) -> Source:
    return Source(sid=sid, title=title, text=text, pages=pages)


LIBRARY = [
    _source("S1", "Battery ageing", "Lithium-ion cells lose capacity with every charge cycle. "
            "Heat speeds up capacity fade."),
    _source("S2", "Retrieval evaluation", "BM25 ranks passages by term frequency and inverse document frequency. "
            "Dense retrievers embed queries and passages."),
    _source("S3", "Retrieval benchmarks", "Benchmarks for retrieval augmented generation test noise robustness. "
            "They also test negative rejection."),
]


def test_stemming_tokenizing_and_sentence_splitting():
    assert {stem(w) for w in ("evaluation", "evaluated", "evaluating", "evaluates", "evaluate")} == {"evaluat"}
    assert stem("batteries") == stem("battery")
    assert stem("process") == stem("processes") == "process"
    assert tokenize("Retrieval-augmented generation for the LLMs") == ["retrieval", "augment", "generat", "llm"]
    text = 'Methods (e.g. BM25) work. Smith et al. report 12.5% gains [S3 p.2]. "Quoted." Next!'
    assert split_sentences(text) == [
        "Methods (e.g. BM25) work.",
        "Smith et al. report 12.5% gains [S3 p.2].",
        '"Quoted."',
        "Next!",
    ]
    assert split_sentences("   ") == []


def test_chunk_text_respects_size_overlap_and_order():
    sentences = [f"Sentence number {i} has exactly seven words." for i in range(20)]
    chunks = chunk_text(" ".join(sentences), chunk_words=30, overlap_words=10)
    assert all(len(c.split()) <= 30 for c in chunks)
    joined = " ".join(chunks)
    assert all(s in joined for s in sentences)  # nothing is lost
    first_tail = chunks[0].split(". ")[-1]
    assert first_tail.rstrip(".") in chunks[1]  # consecutive chunks overlap by a trailing sentence
    long_sentence = " ".join(["word"] * 250) + "."
    chunks = chunk_text(long_sentence, chunk_words=100, overlap_words=20)
    assert [len(c.split()) for c in chunks] == [100, 100, 50]
    assert chunk_text("", 100, 20) == []


def test_chunks_track_source_and_page_labels():
    pdf = _source("", "A PDF", "", pages=("First page text about cells.", "Second page text about heat."))
    library = build_library([], [pdf])
    chunks = chunk_library(library, 50, 10)
    assert [c.label for c in chunks] == ["S1 p.1", "S1 p.2"]
    assert chunk_library([_source("S7", "Abstract", "One sentence only.")])[0].label == "S7"


def test_bm25_ranks_relevant_passages_and_caps_per_source():
    index = EvidenceIndex(chunk_library(LIBRARY, 12, 0))
    hits = index.search("how do retrieval benchmarks test noise robustness", k=3)
    assert hits[0].chunk.sid == "S3"
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert all(a.score >= b.score for a, b in zip(hits, hits[1:], strict=False))
    capped = index.search("retrieval passages benchmarks", k=10, per_source=1)
    assert len({h.chunk.sid for h in capped}) == len(capped)
    assert EvidenceIndex([]).search("anything") == []
    assert index.search("the and of", k=5) == []  # stop words only
    assert index.search("quantum chromodynamics", k=5) == []  # no shared terms


def test_extractive_answer_quotes_sentences_verbatim():
    index = EvidenceIndex(chunk_library(LIBRARY))
    query = "what speeds up capacity fade"
    answer = extractive_answer(query, index.search(query, k=3), index)
    assert answer[0].text == "Heat speeds up capacity fade."
    assert answer[0].label == "S1"
    assert all(s.text in " ".join(src.text for src in LIBRARY) for s in answer)
