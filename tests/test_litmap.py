import numpy as np

from researchpilot.corpus import Source
from researchpilot.litmap import build_map, cluster_table, max_clusters, phrase_analyzer, timeline, top_authors

TOPICS = {
    "battery": "lithium battery capacity fade electrode degradation charging cycles cell voltage",
    "retrieval": "retrieval augmented generation passages ranking query documents index search",
    "vision": "image segmentation convolutional pixels object detection camera frames",
}


def _planted_library(per_topic: int = 7) -> tuple[list[Source], list[str]]:
    rng = np.random.default_rng(3)
    sources, truth = [], []
    for topic, vocab in TOPICS.items():
        words = vocab.split()
        for i in range(per_topic):
            text = " ".join(rng.choice(words, size=40)) + "."
            sources.append(Source(f"S{len(sources) + 1}", f"{topic} study {i}", text,
                                  authors=(f"{topic.title()} Lab", f"Author {i}"), published=f"{2020 + i % 3}-01-01",
                                  categories=(f"cat.{topic}",)))
            truth.append(topic)
    return sources, truth


def test_phrase_analyzer_never_bridges_stop_words_or_punctuation():
    terms = phrase_analyzer("Retrieval-augmented generation for large language models (RAG) in 2024: a survey.")
    assert "retrieval augmented generation" in terms and "large language models" in terms
    assert "generation large" not in terms and "models rag" not in terms


def test_build_map_recovers_planted_topics():
    sources, truth = _planted_library()
    lit = build_map(sources)
    assert lit.k == 3
    for topic in TOPICS:
        labels = {int(lit.labels[i]) for i, t in enumerate(truth) if t == topic}
        assert len(labels) == 1, f"{topic} papers were split across clusters"
    assert lit.coords.shape == (len(sources), 2)
    assert np.allclose(np.diag(lit.similarity), 1.0)
    # the most similar paper to any paper is on the same topic
    assert all(truth[lit.similar_to(i, 1)[0][0]] == truth[i] for i in range(len(sources)))
    names = " ".join(lit.cluster_names.values())
    assert all(any(word in names for word in vocab.split()) for vocab in TOPICS.values())
    table = cluster_table(sources, lit)
    assert table["papers"].sum() == len(sources) and table["years"].iloc[0] == "2020–22"


def test_small_libraries_timeline_and_author_counts():
    empty = build_map([])
    assert empty.k == 0 and empty.coords.shape == (0, 2)
    tiny = build_map([Source("S1", "One", "Only one short document about batteries.")])
    assert tiny.k == 1 and tiny.keyphrases[0]
    three = build_map(_planted_library(1)[0])
    assert three.k == 1 and not three.silhouettes  # too few papers to cluster
    assert max_clusters(50) == 5 and max_clusters(4) == 2 and max_clusters(1000) == 8
    sources, _ = _planted_library(4)
    tl = timeline(sources)
    assert tl["papers"].sum() == len(sources)
    assert set(tl["year"]) == {2020, 2021, 2022}
    authors = dict(top_authors(sources, 3))
    assert authors == {"Battery Lab": 4, "Retrieval Lab": 4, "Vision Lab": 4}
