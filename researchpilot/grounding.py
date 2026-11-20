"""Citation grounding check for any Markdown text.

For every sentence that cites a source ([S3], [S3 p.2] or [S3; S5 p.1]) the checker measures how much
of the sentence the cited passages actually contain:

    support = sum of IDF weights of the sentence's content terms found in the cited passages
              ÷ sum of IDF weights of all the sentence's content terms

IDF weighting means rare, specific words ("Kalman", "hallucination") count for more than common ones
("battery", "model"). Two further checks catch what word overlap misses: numbers in the sentence must
appear in the cited passages, and a sentence must not flip the polarity of the source sentence it
resembles most ("X outperformed Y" against "X did not outperform Y"). Finally the checker searches the
whole library for a passage that would support the sentence better.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .retrieval import EvidenceIndex
from .text import normalise_space, split_sentences, tokenize, words

SUPPORTED_AT = 0.60  # support score at or above which a sentence counts as supported
WEAK_AT = 0.35  # below this a sentence counts as unsupported
BETTER_SOURCE_MARGIN = 0.15  # a supporting passage elsewhere must beat the cited one by this much
MIN_CLAIM_WORDS = 6  # shorter uncited sentences are treated as connective text, not claims
POLARITY_OVERLAP = 0.25  # source sentences at least this similar to the claim are compared for negation
NEGATORS = frozenset("not never cannot unable fail fails failed".split())  # words that negate a predicate
_NEG_TOKEN = re.compile(r"[a-z0-9]+(?:n't)?")

_REF = r"S\d+(?:\s*,?\s*(?:p|pp|page)\.?\s*\d+)?"
CITATION = re.compile(rf"\[\s*({_REF}(?:\s*[,;]\s*{_REF})*)\s*\]", re.I)
_REF_PARTS = re.compile(r"S(\d+)(?:\s*,?\s*(?:p|pp|page)\.?\s*(\d+))?", re.I)
_NUMBER = re.compile(r"(?<![A-Za-z\d.])\d+(?:[.,]\d+)*")
_THOUSANDS = re.compile(r"\d{1,3}(?:,\d{3})+")
_MD_NOISE = re.compile(r"(\*\*|__|`|\*(?=\S)|(?<=\S)\*)")
_LINK = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")

VERDICTS = {
    "supported": "Supported",
    "weak": "Weak support",
    "unsupported": "Not supported",
    "unknown_source": "Unknown source",
    "uncited": "No citation",
}


@dataclass(frozen=True)
class Citation:
    sid: str
    page: int | None = None

    @property
    def label(self) -> str:
        return f"{self.sid} p.{self.page}" if self.page else self.sid


@dataclass(frozen=True)
class SentenceCheck:
    sentence: str  # as written, citations included
    claim: str  # citations removed
    citations: tuple[Citation, ...]
    verdict: str  # a key of VERDICTS
    score: float | None = None  # support from the cited passages (0-1); None when not computable
    missing_numbers: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()  # cited labels that do not exist in the library
    suggestion: str | None = None  # a passage label that supports the sentence better
    suggestion_score: float | None = None
    conflict: str | None = None  # a similar source sentence whose negation differs from the claim

    @property
    def note(self) -> str:
        notes = []
        if self.unknown:
            notes.append("Not in the library: " + ", ".join(self.unknown) + ".")
        if self.missing_numbers:
            notes.append("Numbers not found in the cited passage: " + ", ".join(self.missing_numbers) + ".")
        if self.conflict:
            notes.append(f'Polarity differs from the source: "{self.conflict}"')
        if self.suggestion:
            notes.append(f"Better match: [{self.suggestion}] (support {self.suggestion_score:.0%}).")
        if self.verdict == "uncited":
            notes.append("A factual-looking sentence without a citation.")
        return " ".join(notes)


@dataclass(frozen=True)
class GroundingResult:
    checks: list[SentenceCheck] = field(default_factory=list)

    def count(self, verdict: str) -> int:
        return sum(1 for c in self.checks if c.verdict == verdict)

    @property
    def cited(self) -> int:
        return sum(1 for c in self.checks if c.citations)

    @property
    def grounded_share(self) -> float | None:
        """Share of cited sentences whose citations support them; None when nothing is cited."""
        return self.count("supported") / self.cited if self.cited else None


def parse_citations(text: str) -> list[Citation]:
    found: list[Citation] = []
    for group in CITATION.finditer(text or ""):
        for number, page in _REF_PARTS.findall(group.group(1)):
            citation = Citation(f"S{int(number)}", int(page) if page else None)
            if citation not in found:
                found.append(citation)
    return found


def strip_citations(text: str) -> str:
    return normalise_space(CITATION.sub(" ", text or "")).replace(" .", ".").replace(" ,", ",")


def _normalise_number(raw: str) -> str:
    """"1,000" -> "1000", "0,5" -> "0.5", "12.50" -> "12.5", "3.0" -> "3"."""
    number = raw.replace(",", "") if _THOUSANDS.fullmatch(raw) else raw.replace(",", ".")
    if "." in number:
        number = number.rstrip("0").rstrip(".")
    return number


def numbers_in(text: str) -> set[str]:
    return {_normalise_number(n) for n in _NUMBER.findall(text or "")}


def markdown_sentences(markdown: str) -> list[str]:
    """Sentences of the prose in a Markdown document. Headings, tables and code are skipped, and a
    citation that follows the full stop ("... gains. [S3]") is attached to the sentence before it."""
    sentences: list[str] = []
    in_code = False
    for raw in (markdown or "").splitlines():
        line = raw.strip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not line or line.startswith(("#", "|", ">", "---")):
            continue
        line = _BULLET.sub("", line)
        line = _LINK.sub(r"\1", line)
        line = _MD_NOISE.sub("", line)
        for sentence in split_sentences(line):
            if sentences and not strip_citations(sentence) and CITATION.search(sentence):
                sentences[-1] = f"{sentences[-1]} {sentence}"
            else:
                sentences.append(sentence)
    return sentences


def negated_terms(text: str) -> set[str]:
    """Stem of the predicate a negator applies to: the first content word after it, skipping adverbs.
    "did not outperform" -> {"outperform"}, "cannot be generalized" -> {"generaliz"},
    "fail to adequately account" -> {"account"}. Noun phrases such as "models without X" are left alone."""
    tokens = _NEG_TOKEN.findall((text or "").lower())
    found: set[str] = set()
    for i, token in enumerate(tokens):
        if token in NEGATORS or token.endswith("n't"):
            following = [t for t in tokens[i + 1 : i + 5] if tokenize(t) and not t.endswith("ly")]
            if following:
                found.update(tokenize(following[0]))
    return found


def polarity_conflict(claim: str, terms: set[str], passages: list[str], index: EvidenceIndex) -> str | None:
    """A similar source sentence that negates what the claim asserts, or asserts what the claim negates,
    unless another similar sentence states the same predicate with the claim's own polarity."""
    claim_negated = negated_terms(claim)
    similar = []
    for passage in passages:
        for sentence in split_sentences(passage):
            sentence_terms = set(tokenize(sentence))
            overlap = support_score(terms, sentence_terms, index) or 0.0
            if overlap >= POLARITY_OVERLAP:
                similar.append((overlap, sentence, sentence_terms, negated_terms(sentence)))
    agreeing = [(t, n) for _, _, t, n in similar if bool(n) == bool(claim_negated)]
    candidates = []
    for overlap, sentence, sentence_terms, source_negated in similar:
        if bool(source_negated) == bool(claim_negated):
            continue  # both negated or neither: same polarity
        disputed = (source_negated & terms) if source_negated else (claim_negated & sentence_terms)
        backed = {t for other_terms, other_negated in agreeing
                  for t in (other_negated if claim_negated else other_terms) & disputed}
        if disputed - backed:
            candidates.append((overlap, sentence))
    return max(candidates)[1] if candidates else None


def support_score(claim_terms: set[str], passage_terms: set[str], index: EvidenceIndex) -> float | None:
    total = sum(index.weight(t) for t in claim_terms)
    if not claim_terms or total == 0:
        return None
    return sum(index.weight(t) for t in claim_terms & passage_terms) / total


def check_sentence(sentence: str, index: EvidenceIndex, known_sids: set[str]) -> SentenceCheck | None:
    citations = tuple(parse_citations(sentence))
    claim = strip_citations(sentence)
    terms = set(tokenize(claim))
    if not citations:
        if len(words(claim)) >= MIN_CLAIM_WORDS and terms:
            return SentenceCheck(sentence, claim, (), "uncited")
        return None

    unknown = tuple(c.label for c in citations if c.sid not in known_sids or not index.chunks_for(c.sid, c.page))
    cited_chunks = [ch for c in citations for ch in index.chunks_for(c.sid, c.page)]
    if not cited_chunks:
        return SentenceCheck(sentence, claim, citations, "unknown_source", unknown=unknown)

    cited_terms = set().union(*(index.term_sets[ch.cid] for ch in cited_chunks))
    score = support_score(terms, cited_terms, index)
    cited_numbers = set().union(*(numbers_in(ch.text) for ch in cited_chunks))
    missing = tuple(sorted(numbers_in(claim) - cited_numbers))
    conflict = polarity_conflict(claim, terms, [ch.text for ch in cited_chunks], index)

    suggestion, suggestion_score = None, None
    cited_ids = {ch.cid for ch in cited_chunks}
    for hit in index.search(claim, k=5, per_source=None):
        if hit.chunk.cid in cited_ids:
            continue
        alt = support_score(terms, index.term_sets[hit.chunk.cid], index) or 0.0
        if alt >= SUPPORTED_AT and alt >= (score or 0.0) + BETTER_SOURCE_MARGIN and alt > (suggestion_score or 0.0):
            suggestion, suggestion_score = hit.label, alt

    if score is None:
        verdict = "weak"
    elif score >= SUPPORTED_AT and not missing and not conflict:
        verdict = "supported"
    elif score >= WEAK_AT:
        verdict = "weak"
    else:
        verdict = "unsupported"
    return SentenceCheck(
        sentence, claim, citations, verdict, score, missing, unknown, suggestion, suggestion_score, conflict
    )


def check_text(markdown: str, index: EvidenceIndex, known_sids: set[str] | None = None) -> GroundingResult:
    """Check every cited sentence (and flag claim-like uncited ones) in a Markdown text."""
    sids = known_sids if known_sids is not None else {c.sid for c in index.chunks}
    checks = [check_sentence(s, index, sids) for s in markdown_sentences(markdown)]
    return GroundingResult([c for c in checks if c is not None])
