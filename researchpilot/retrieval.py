"""Chunking with source and page tracking, BM25 retrieval and an extractive answer.

Each chunk keeps the label of the source it came from (S3) and, for PDFs, its page (S3 p.2). Search
results carry that label, so every passage shown or handed to the model is citable as [S3 p.2].
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .corpus import Source
from .text import split_sentences, tokenize

DEFAULT_CHUNK_WORDS = 120
DEFAULT_OVERLAP_WORDS = 30
DEFAULT_TOP_K = 8
DEFAULT_PER_SOURCE = 2


@dataclass(frozen=True)
class Chunk:
    cid: int
    sid: str
    page: int | None
    text: str
    title: str = ""  # title of the source; indexed with the text so passages of on-topic papers rank higher

    @property
    def label(self) -> str:
        """Citation label: "S3" for unpaged sources, "S3 p.2" for PDF pages."""
        return f"{self.sid} p.{self.page}" if self.page else self.sid


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # raw BM25 score
    rank: int  # 1 = best

    @property
    def label(self) -> str:
        return self.chunk.label


def _pieces(text: str, chunk_words: int) -> list[str]:
    """Sentences, with any sentence longer than a chunk cut into chunk-sized pieces."""
    pieces = []
    for sentence in split_sentences(text):
        tokens = sentence.split()
        if len(tokens) <= chunk_words:
            pieces.append(sentence)
        else:
            pieces += [" ".join(tokens[i : i + chunk_words]) for i in range(0, len(tokens), chunk_words)]
    return pieces


def chunk_text(
    text: str, chunk_words: int = DEFAULT_CHUNK_WORDS, overlap_words: int = DEFAULT_OVERLAP_WORDS
) -> list[str]:
    """Pack whole sentences into chunks of about `chunk_words` words. Consecutive chunks share up to
    `overlap_words` words of trailing sentences, so a fact split across a boundary is still findable."""
    if chunk_words < 1:
        raise ValueError("chunk_words must be positive")
    overlap_words = max(0, min(overlap_words, chunk_words // 2))
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for piece in _pieces(text, chunk_words):
        n = len(piece.split())
        if current and count + n > chunk_words:
            chunks.append(" ".join(current))
            carry: list[str] = []
            carried = 0
            for sentence in reversed(current):
                k = len(sentence.split())
                if carried + k > overlap_words:
                    break
                carry.insert(0, sentence)
                carried += k
            if carried + n > chunk_words:
                carry, carried = [], 0
            current, count = carry, carried
        current.append(piece)
        count += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_library(
    sources: Iterable[Source],
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for source in sources:
        for page, text in source.page_texts():
            for piece in chunk_text(text, chunk_words, overlap_words):
                chunks.append(Chunk(len(chunks), source.sid, page, piece, source.title))
    return chunks


class LuceneBM25(BM25Okapi):
    """BM25 with Lucene's IDF, log(1 + (N - n + 0.5) / (n + 0.5)). It is always positive, so a term that
    appears in half the documents still counts. The textbook Okapi IDF drops to zero or below there,
    which makes a small library (one PDF with two pages) return no results at all."""

    def _calc_idf(self, nd: dict) -> None:
        for word, freq in nd.items():
            self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))


class EvidenceIndex:
    """BM25 over chunks, plus corpus IDF weights reused by the extractive answer and grounding check."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.tokens = [tokenize(f"{c.title} {c.text}") for c in chunks]
        self.term_sets = [set(t) for t in self.tokens]
        self._bm25 = LuceneBM25(self.tokens) if any(self.tokens) else None
        df: Counter[str] = Counter()
        for terms in self.term_sets:
            df.update(terms)
        n = len(chunks)
        self.idf = {term: math.log((n + 1) / (count + 1)) + 1.0 for term, count in df.items()}
        self._default_idf = math.log(n + 1) + 1.0  # a term the corpus never uses is maximally specific

    def __len__(self) -> int:
        return len(self.chunks)

    def weight(self, term: str) -> float:
        return self.idf.get(term, self._default_idf)

    def chunks_for(self, sid: str, page: int | None = None) -> list[Chunk]:
        return [c for c in self.chunks if c.sid == sid and (page is None or c.page == page)]

    def search(self, query: str, k: int = DEFAULT_TOP_K, per_source: int | None = DEFAULT_PER_SOURCE) -> list[Hit]:
        """Top-k chunks by BM25. `per_source` caps passages per source so one paper cannot crowd out the rest."""
        terms = tokenize(query)
        if self._bm25 is None or not terms or k <= 0:
            return []
        scores = self._bm25.get_scores(terms)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        hits: list[Hit] = []
        taken: Counter[str] = Counter()
        for i in order:
            if scores[i] <= 0:
                break
            chunk = self.chunks[i]
            if per_source and taken[chunk.sid] >= per_source:
                continue
            taken[chunk.sid] += 1
            hits.append(Hit(chunk, float(scores[i]), len(hits) + 1))
            if len(hits) >= k:
                break
        return hits


@dataclass(frozen=True)
class ExtractiveSentence:
    text: str
    label: str
    score: float  # IDF-weighted share of the query's terms that the sentence contains (0-1)


def extractive_answer(
    query: str, hits: list[Hit], index: EvidenceIndex, max_sentences: int = 4, per_source: int = 2
) -> list[ExtractiveSentence]:
    """Pick the retrieved sentences that cover the most (IDF-weighted) query terms. Sentences are quoted
    verbatim from the sources, so this answer needs no model and cannot invent anything."""
    q_terms = set(tokenize(query))
    total = sum(index.weight(t) for t in q_terms)
    if not q_terms or total == 0:
        return []
    candidates: dict[str, ExtractiveSentence] = {}
    for hit in hits:
        for sentence in split_sentences(hit.chunk.text):
            if len(sentence.split()) < 5 or sentence in candidates:
                continue
            covered = q_terms & set(tokenize(sentence))
            score = sum(index.weight(t) for t in covered) / total
            if score > 0:
                candidates[sentence] = ExtractiveSentence(sentence, hit.label, round(score, 4))
    ranked = sorted(candidates.values(), key=lambda s: -s.score)
    picked: list[ExtractiveSentence] = []
    per: Counter[str] = Counter()
    for sentence in ranked:
        sid = sentence.label.split()[0]
        if per[sid] >= per_source:
            continue
        per[sid] += 1
        picked.append(sentence)
        if len(picked) >= max_sentences:
            break
    return picked
