"""Literature map: keyphrases, document similarity, topic clusters and a 2-D projection.

    TF-IDF (1-3 word phrases that never span a stop word or punctuation, sublinear tf) on title + abstract
    -> keyphrases: highest-weighted phrases per paper, longer phrases slightly preferred
    -> cosine similarity between papers
    -> LSA (TruncatedSVD to a few dimensions, rows L2-normalised)
    -> k-means for k = 2..kmax with kmax = sqrt(n / 2), k chosen by the best silhouette score
    -> cluster labels: phrases over-represented in the cluster relative to the whole library
    -> 2-D map: PCA of the LSA vectors
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from .corpus import Source
from .text import STOPWORDS

# words that appear in almost every abstract and say nothing about the topic
GENERIC = {
    "paper", "papers", "propose", "proposed", "proposes", "approach", "approaches", "method", "methods",
    "result", "results", "show", "shows", "shown", "novel", "existing", "present", "presents", "study",
    "studies", "work", "works", "demonstrate", "demonstrates", "achieve", "achieves", "achieved", "significant",
    "significantly", "address", "addresses", "introduce", "introduces", "furthermore", "additionally", "key",
    "various", "different", "including", "provide", "provides", "use", "used", "using", "based",
    "new", "high", "improve", "improves", "improved", "experiments", "experimental", "end", "able", "vs",
    "trade", "offs", "impact", "fundamental", "combining", "instances", "conventional", "recent", "recently",
    "challenge", "challenges", "comprehensive", "effective", "effectively", "enables", "enable",
    "remains", "remain", "crucial", "critical", "important", "potential", "first", "finally", "overall",
}
STOP = frozenset(ENGLISH_STOP_WORDS | STOPWORDS | GENERIC)
MAX_K = 8
LSA_DIMS = 6  # few dimensions: clusters separate better than in the sparse TF-IDF space
MIN_CLUSTER = 2  # a k whose smallest cluster is smaller than this is skipped
PHRASE_BOOST = 0.25  # each extra word in a phrase adds 25% to its keyphrase ranking weight
_TOKEN = re.compile(r"[a-z][a-z0-9]+|\S")  # a word, or any other single character (which ends a phrase)


def phrase_analyzer(doc: str) -> list[str]:
    """Unigrams, bigrams and trigrams of consecutive content words. A stop word or punctuation mark
    ends a phrase, so "generation for large models" never yields the false bigram "generation large"."""
    terms: list[str] = []
    run: list[str] = []

    def flush() -> None:
        for i in range(len(run)):
            for n in (1, 2, 3):
                if i + n <= len(run):
                    terms.append(" ".join(run[i : i + n]))

    for token in _TOKEN.findall(doc.lower().replace("-", " ")):
        if len(token) < 2 or token in STOP:
            flush()
            run = []
        else:
            run.append(token)
    flush()
    return terms


def max_clusters(n: int, cap: int = MAX_K) -> int:
    """Rule of thumb k <= sqrt(n / 2): 50 papers -> 5 clusters at most."""
    return max(2, min(cap, int(math.sqrt(n / 2))))


@dataclass(frozen=True)
class LiteratureMap:
    sids: list[str]
    keyphrases: list[list[str]]
    similarity: np.ndarray  # n x n cosine similarity of TF-IDF vectors
    labels: np.ndarray  # cluster id per source
    k: int
    silhouettes: dict[int, float] = field(default_factory=dict)  # k -> mean silhouette
    cluster_names: dict[int, str] = field(default_factory=dict)
    representatives: dict[int, int] = field(default_factory=dict)  # cluster -> index of its central paper
    coords: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))

    @property
    def silhouette(self) -> float | None:
        return self.silhouettes.get(self.k)

    def similar_to(self, i: int, n: int = 3) -> list[tuple[int, float]]:
        row = self.similarity[i]
        order = [j for j in np.argsort(-row, kind="stable") if j != i]
        return [(int(j), float(row[j])) for j in order[:n]]

    def top_pairs(self, n: int = 8) -> list[tuple[int, int, float]]:
        size = len(self.sids)
        pairs = [(i, j, float(self.similarity[i, j])) for i in range(size) for j in range(i + 1, size)]
        return sorted(pairs, key=lambda p: -p[2])[:n]


def _document(source: Source) -> str:
    return f"{source.title}. {source.title}. {source.text}"  # title counted twice: it is the densest summary


def _pick_phrases(terms: list[str], n: int) -> list[str]:
    """Keep the best phrases, skipping any that shares a word with a phrase already chosen
    ("process regression" after "gaussian process", "models" after "language models")."""
    chosen: list[str] = []
    used: set[str] = set()
    for term in terms:
        parts = set(term.split())
        if not parts & used:
            used |= parts
            chosen.append(term)
        if len(chosen) >= n:
            break
    return chosen


def _vectorise(docs: list[str]) -> tuple[object, np.ndarray, np.ndarray] | None:
    n = len(docs)
    for min_df, max_df in ((2, 0.85), (1, 1.0)) if n >= 10 else ((1, 1.0),):
        vectorizer = TfidfVectorizer(analyzer=phrase_analyzer, min_df=min_df, max_df=max_df, sublinear_tf=True)
        try:
            matrix = vectorizer.fit_transform(docs)
        except ValueError:  # empty vocabulary
            continue
        return vectorizer, matrix, np.asarray(vectorizer.get_feature_names_out())
    return None


def build_map(sources: list[Source], max_k: int = MAX_K, phrases: int = 5, seed: int = 0) -> LiteratureMap:
    sids = [s.sid for s in sources]
    n = len(sources)
    empty = LiteratureMap(sids, [[] for _ in sources], np.eye(n), np.zeros(n, dtype=int), 1 if n else 0,
                          coords=np.zeros((n, 2)))
    if n == 0:
        return empty
    fitted = _vectorise([_document(s) for s in sources])
    if fitted is None:
        return empty
    _, matrix, vocab = fitted
    dense = matrix.toarray()

    boost = np.array([1.0 + PHRASE_BOOST * term.count(" ") for term in vocab])
    keyphrases = []
    for row in dense * boost:
        ranked = [vocab[j] for j in np.argsort(-row, kind="stable") if row[j] > 0]
        keyphrases.append(_pick_phrases(ranked, phrases))
    similarity = cosine_similarity(matrix)

    # LSA space for clustering and the 2-D map
    n_comp = min(LSA_DIMS, n - 1, dense.shape[1] - 1)
    if n_comp >= 2:
        lsa = normalize(TruncatedSVD(n_components=n_comp, random_state=seed).fit_transform(matrix))
    else:
        lsa = normalize(dense)

    labels = np.zeros(n, dtype=int)
    silhouettes: dict[int, float] = {}
    if n >= 4:
        best = None
        for k in range(2, min(max_clusters(n, max_k), n - 1) + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(lsa)
            sizes = np.bincount(km.labels_, minlength=k)
            if len(set(km.labels_)) < 2 or sizes.min() < min(MIN_CLUSTER, n // k):
                continue
            score = float(silhouette_score(lsa, km.labels_))
            silhouettes[k] = round(score, 4)
            if best is None or score > best[0] + 1e-9:
                best = (score, k, km.labels_)
        if best is not None:
            labels = _order_by_size(best[2])
    k = int(labels.max()) + 1

    overall = dense.mean(axis=0)
    names, reps = {}, {}
    for c in range(k):
        members = np.flatnonzero(labels == c)
        lift = dense[members].mean(axis=0) - overall
        lift = lift * boost
        top = [vocab[j] for j in np.argsort(-lift, kind="stable") if lift[j] > 0][:12]
        names[c] = " · ".join(_pick_phrases(top, 3)) or f"Cluster {c + 1}"
        centroid = lsa[members].mean(axis=0)
        reps[c] = int(members[np.argmax(lsa[members] @ centroid)])

    if lsa.shape[1] >= 2 and n >= 3:
        coords = PCA(n_components=2, random_state=seed).fit_transform(lsa)
    else:
        coords = np.zeros((n, 2))
    return LiteratureMap(sids, keyphrases, similarity, labels, k, silhouettes, names, reps, coords)


def _order_by_size(labels: np.ndarray) -> np.ndarray:
    """Renumber clusters so cluster 0 is the largest (ties broken by first appearance)."""
    counts = Counter(labels.tolist())
    order = sorted(counts, key=lambda c: (-counts[c], list(labels).index(c)))
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping[c] for c in labels.tolist()], dtype=int)


def cluster_table(sources: list[Source], lit: LiteratureMap) -> pd.DataFrame:
    rows = []
    for c in range(lit.k):
        members = [s for s, label in zip(sources, lit.labels, strict=True) if label == c]
        years = [s.year for s in members if s.year]
        rep = sources[lit.representatives[c]] if c in lit.representatives else members[0]
        rows.append(
            {
                "cluster": c,
                "label": lit.cluster_names.get(c, f"Cluster {c + 1}"),
                "papers": len(members),
                "years": _span(years),
                "median_year": int(np.median(years)) if years else None,
                "representative": f"{rep.sid} · {rep.title}",
            }
        )
    return pd.DataFrame(rows)


def _span(years: list[int]) -> str:
    if not years:
        return "—"
    low, high = min(years), max(years)
    return str(low) if low == high else f"{low}–{str(high)[2:]}" if str(low)[:2] == str(high)[:2] else f"{low}–{high}"


def timeline(sources: list[Source], labels: np.ndarray | None = None) -> pd.DataFrame:
    """Papers per year (and per cluster when labels are given)."""
    rows = [
        {"year": s.year, "cluster": int(labels[i]) if labels is not None else 0}
        for i, s in enumerate(sources)
        if s.year
    ]
    if not rows:
        return pd.DataFrame(columns=["year", "cluster", "papers"])
    df = pd.DataFrame(rows)
    return df.groupby(["year", "cluster"]).size().reset_index(name="papers").sort_values(["year", "cluster"])


def top_authors(sources: list[Source], n: int = 10) -> list[tuple[str, int]]:
    counts = Counter(a for s in sources for a in s.authors)
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]


def top_categories(sources: list[Source], n: int = 10) -> list[tuple[str, int]]:
    counts = Counter(c for s in sources for c in s.categories)
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
