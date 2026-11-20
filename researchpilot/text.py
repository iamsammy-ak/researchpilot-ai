"""Text utilities shared by retrieval, the literature map and the grounding check.

The tokenizer is deliberately simple and deterministic: lowercase word tokens, English stop words
removed and a light suffix stemmer, so "evaluation", "evaluated" and "evaluating" all become "evaluat".
"""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a about above after again against all also although am an and any are as at be because been before
    being below between both but by can could did do does doing done down during each either et etc few for
    from further had has have having he her here hers herself him himself his how however i if in into is it
    its itself just may me might more most much must my myself no nor not now of off on once only or other
    our ours ourselves out over own per same she should so some such than that the their theirs them
    themselves then there these they this those through thus to too under until up upon us use used uses
    using very via was we well were what when where whether which while who whom why will with within
    without would yet you your yours yourself yourselves al eg ie paper propose proposed show shows study
    work results based approach method methods new one two three""".split()
)

_WORD = re.compile(r"[a-z0-9]+")  # hyphenated terms split, so "retrieval-augmented" matches "retrieval augmented"
_ABBREVIATIONS = ("e.g.", "i.e.", "et al.", "etc.", "vs.", "fig.", "figs.", "eq.", "eqs.", "approx.", "cf.", "no.")
_SENTENCE_END = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"')\]]))\s+(?=[\"'(\[]?[A-Z0-9])")
_PROTECT = "⁣"  # invisible separator that shields abbreviation full stops while splitting

# (suffix, replacement, minimum word length for the rule to apply); the first matching rule wins
_SUFFIXES = (
    ("ational", "ate", 8),
    ("ization", "ize", 8),
    ("ations", "ate", 7),
    ("ation", "ate", 6),
    ("ments", "", 7),
    ("ment", "", 6),
    ("ness", "", 6),
    ("ings", "", 6),
    ("ing", "", 6),
    ("ies", "y", 5),
    ("ied", "y", 5),
    ("sses", "ss", 5),
    ("ed", "", 6),
    ("es", "e", 5),
    ("s", "", 4),
)


def stem(word: str) -> str:
    """Strip common English suffixes. Not linguistically perfect, but consistent for index and query."""
    if len(word) <= 3 or word.isdigit():
        return word
    for suffix, replacement, min_len in _SUFFIXES:
        if word.endswith(suffix) and len(word) >= min_len:
            if suffix == "s" and word.endswith(("ss", "us", "is")):
                break
            word = word[: -len(suffix)] + replacement
            break
    if len(word) > 4 and word.endswith("e"):
        word = word[:-1]
    return word


def words(text: str) -> list[str]:
    """Lowercase word tokens, punctuation removed."""
    return _WORD.findall((text or "").lower())


def tokenize(text: str) -> list[str]:
    """Content tokens: stop words and one-letter tokens removed, then stemmed."""
    return [stem(w) for w in words(text) if w not in STOPWORDS and len(w) > 1]


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences without breaking on common abbreviations such as "e.g." or "et al."."""
    text = normalise_space(text)
    if not text:
        return []
    protected = text
    for abbr in _ABBREVIATIONS:
        protected = re.sub(re.escape(abbr), lambda m: m.group(0).replace(".", _PROTECT), protected, flags=re.I)
    parts = _SENTENCE_END.split(protected)
    return [p.replace(_PROTECT, ".").strip() for p in parts if p.strip()]


def normalise_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()
