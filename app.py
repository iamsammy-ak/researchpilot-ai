"""ResearchPilot AI: read papers and datasets, retrieve evidence, map the literature and check citations."""

from __future__ import annotations

import hashlib
import html

import pandas as pd
import streamlit as st

from researchpilot import charts
from researchpilot import dataset as ds
from researchpilot.arxiv import ArxivClient, ArxivError
from researchpilot.config import load_settings
from researchpilot.corpus import Source, build_library, from_text, library_fingerprint, read_pdf
from researchpilot.demo import DATASET_NAME, load_bundled_dataset, load_corpus, load_samples, load_saved_analysis
from researchpilot.grounding import SUPPORTED_AT, VERDICTS, WEAK_AT, GroundingResult, check_text
from researchpilot.litmap import LiteratureMap, build_map, cluster_table, timeline, top_authors, top_categories
from researchpilot.llm import LLMClient, LLMError
from researchpilot.pipeline import evidence_fingerprint, finding_line, report_markdown, run_qa, run_report
from researchpilot.report import build_brief
from researchpilot.retrieval import (
    DEFAULT_CHUNK_WORDS,
    DEFAULT_OVERLAP_WORDS,
    DEFAULT_PER_SOURCE,
    DEFAULT_TOP_K,
    EvidenceIndex,
    chunk_library,
    extractive_answer,
)
from researchpilot.schemas import ResearchReport
from researchpilot.ui import (
    badge,
    card,
    cite,
    hero,
    highlight,
    inject_css,
    link,
    pill,
    score_bar,
    table,
    tags,
    with_citation_chips,
)

st.set_page_config(page_title="ResearchPilot AI", page_icon="🔬", layout="wide")
inject_css()

SAMPLES = load_samples()
SAMPLE_NAMES = [s["name"] for s in SAMPLES]
ARXIV = "Search arXiv"
OWN = "Only my documents"
UPLOAD = "Upload a CSV"
AI_SETUP = ("AI analysis runs once an OpenRouter key is configured: add `OPENROUTER_API_KEY` "
            "(see README → Configuration), then click Run.")
ss = st.session_state


# ----------------------------------------------------------------------------- state
def current_sample() -> dict | None:
    choice = ss.get("library_choice")
    return SAMPLES[SAMPLE_NAMES.index(choice)] if choice in SAMPLE_NAMES else None


def _load_choice() -> None:
    sample = current_sample()
    if sample is not None:
        ss["question"] = sample["question"]
        ss["query"] = sample["qa_questions"][0]
        ss["ground_text"] = sample["grounding_example"]
    ss["lib_filter"] = ""


def _use_suggestion(key: str) -> None:
    if ss.get(key):
        ss["query"] = ss[key]


if "library_choice" not in ss:
    ss.update(
        {
            "library_choice": SAMPLE_NAMES[0],
            "chunk_words": DEFAULT_CHUNK_WORDS,
            "overlap": DEFAULT_OVERLAP_WORDS,
            "top_k": DEFAULT_TOP_K,
            "per_source": DEFAULT_PER_SOURCE,
            "arxiv_query": "retrieval augmented generation evaluation",
            "arxiv_n": 30,
            "ai": {},
        }
    )
    _load_choice()


@st.cache_data(show_spinner=False)
def corpus_for(slug: str) -> dict:
    return load_corpus(slug)


@st.cache_data(show_spinner=False, max_entries=32)
def pdf_source(data: bytes, name: str) -> Source:
    return read_pdf(data, name)


@st.cache_resource(show_spinner=False, max_entries=6)
def get_index(key: str, chunk_words: int, overlap: int, _library: tuple[Source, ...]) -> EvidenceIndex:
    return EvidenceIndex(chunk_library(_library, chunk_words, overlap))


@st.cache_resource(show_spinner=False, max_entries=6)
def get_map(key: str, _library: tuple[Source, ...]) -> LiteratureMap:
    return build_map(list(_library))


@st.cache_data(show_spinner=False)
def bundled_dataset() -> pd.DataFrame:
    return load_bundled_dataset()


@st.cache_data(show_spinner=False, max_entries=8)
def uploaded_dataset(data: bytes) -> pd.DataFrame:
    return ds.read_csv(data)


@st.cache_data(show_spinner=False, max_entries=8)
def dataset_insights(data_id: str, _df: pd.DataFrame) -> dict:
    return {
        "profile": ds.profile(_df),
        "findings": ds.findings(_df),
        "pairs": ds.correlation_pairs(_df, top=8),
        "outliers": ds.iqr_outliers(_df),
    }


def own_documents() -> tuple[list[Source], list[str]]:
    docs, errors = [], []
    for upload in ss.get("uploads") or []:
        data = upload.getvalue()
        try:
            if upload.name.lower().endswith(".pdf"):
                docs.append(pdf_source(data, upload.name))
            else:
                docs.append(from_text(data.decode("utf-8", errors="replace"), upload.name.rsplit(".", 1)[0]))
        except Exception as exc:  # a broken upload should not take the app down
            errors.append(f"{upload.name}: {exc}")
    if (ss.get("paste_text") or "").strip():
        docs.append(from_text(ss["paste_text"], ss.get("paste_title") or ""))
    return docs, errors


def ai_setup(items: list[str]) -> None:
    """Demo mode: present the AI step as a setup step and say what it will produce."""
    st.info(AI_SETUP, icon="🔑")
    card("What the AI step will produce", items, raw_html=True)


def lookup(fp: str) -> tuple[dict | None, str | None]:
    """A fresh result from this session, else a saved demo analysis, else nothing."""
    if fp in ss["ai"]:
        return ss["ai"][fp], "fresh"
    saved = load_saved_analysis(fp)
    return (saved, "saved") if saved else (None, None)


# --------------------------------------------------------------------------- sidebar
settings = load_settings()
sample = current_sample()
with st.sidebar:
    st.markdown("## 🔬 ResearchPilot AI")
    if settings.live:
        fallbacks = len(settings.models) - 1
        st.success(f"**Live AI** via OpenRouter\n\n`{settings.models[0]}`\n\n+{fallbacks} fallback models")
    else:
        st.info(
            "**Demo mode**: retrieval, the literature map, dataset statistics and the grounding check "
            "all run live.\n\nThe AI report and Q&A run once `OPENROUTER_API_KEY` is set."
        )
    st.selectbox("Library", SAMPLE_NAMES + [ARXIV, OWN], key="library_choice", on_change=_load_choice)
    if ss["library_choice"] == ARXIV:
        st.text_input("arXiv query", key="arxiv_query", help='Plain words are AND-ed; use "quotes" for phrases. '
                      "Field syntax such as ti:, abs: or cat:cs.CL is passed through.")
        st.slider("Papers to fetch", 10, 100, step=10, key="arxiv_n")
        if st.button("🔎 Fetch from arXiv", width="stretch", disabled=not ss["arxiv_query"].strip()):
            with st.spinner("Querying arXiv (at most one request every 3 seconds)…"):
                try:
                    ss["arxiv_records"] = ArxivClient().search(ss["arxiv_query"], max_results=ss["arxiv_n"])
                    ss["arxiv_fetched"] = ss["arxiv_query"]
                except (ArxivError, ValueError) as exc:
                    st.error(str(exc))
        st.caption("Keyless public API, responses cached on disk. arXiv metadata is CC0.")
    with st.expander("➕ Add your own documents", expanded=ss["library_choice"] == OWN):
        st.file_uploader("PDF or text files", type=["pdf", "txt", "md"], accept_multiple_files=True, key="uploads")
        st.text_input("Title for pasted text", key="paste_title")
        st.text_area("Paste text", key="paste_text", height=110)
        st.caption("PDF pages are kept, so citations point at a page, e.g. [S51 p.3].")
    with st.expander("⚙️ Retrieval settings"):
        st.slider("Chunk size (words)", 60, 300, step=20, key="chunk_words")
        st.slider("Overlap (words)", 0, 80, step=10, key="overlap")
        st.slider("Passages per query (top-k)", 3, 15, key="top_k")
        st.slider("Max passages per source", 1, 4, key="per_source")
        st.caption("Saved demo analyses match the default settings.")

# --------------------------------------------------------------------------- library
own, upload_errors = own_documents()
corpus = None
if sample is not None:
    corpus = corpus_for(sample["corpus"])
    records, library_name = corpus["papers"], corpus["name"]
elif ss["library_choice"] == ARXIV:
    records = ss.get("arxiv_records", [])
    library_name = f"arXiv: {ss.get('arxiv_fetched', 'no search yet')}"
else:
    records, library_name = [], "My documents"
library = build_library(records, own)
by_sid = {s.sid: s for s in library}
lib_key = library_fingerprint(library)
index = get_index(lib_key, ss["chunk_words"], ss["overlap"], tuple(library))
lit = get_map(lib_key, tuple(library))

# ------------------------------------------------------------------------------ main
hero(
    "ResearchPilot AI",
    "A research assistant that reads papers and datasets, retrieves evidence with citations, maps the "
    "literature and writes structured reports whose citations are checked.",
    ["arXiv · PDF · CSV", "BM25 evidence retrieval", "Topic clusters", "Dataset statistics",
     "Citation grounding check"],
)

with st.expander("🎯 Research question", expanded=True):
    st.text_area("What do you want to find out?", key="question", height=68,
                 placeholder="e.g. Which methods estimate battery state of health, and how accurate are they?")
    parts = [f"**{library_name}**", f"{len(library)} sources"]
    if corpus is not None:
        parts.append("synthetic placeholders" if corpus.get("synthetic") else
                     f"arXiv metadata (CC0) fetched {corpus['fetched_at']}")
    if own:
        parts.append(f"{len(own)} of your documents")
    st.caption(" · ".join(parts))
for message in upload_errors:
    st.warning(message)

question = (ss.get("question") or "").strip()
report_hits = index.search(question, ss["top_k"], ss["per_source"]) if question else []
report_fp = evidence_fingerprint("report", question, report_hits)

run_col, note_col = st.columns([1, 2.6], vertical_alignment="center")
can_run = settings.live and bool(report_hits)
if run_col.button("✨ Run AI report", type="primary", disabled=not can_run, width="stretch"):
    with st.status("Writing the research report…", expanded=True) as status:
        try:
            llm = LLMClient(settings, app_name="ResearchPilot AI")
            result = run_report(llm, question, report_hits, by_sid, on_step=status.write)
            ss["ai"][result["fingerprint"]] = result
            status.update(label=f"Report complete · {result['model']}", state="complete", expanded=False)
        except LLMError as exc:
            status.update(label="The AI report failed", state="error")
            st.error(str(exc))

analysis, freshness = lookup(report_fp)
if freshness == "saved":
    note_col.caption(f"Showing the saved AI report for this question (`{analysis['model']}`, "
                     f"{analysis['generated_at'][:10]}). No model calls used.")
elif freshness == "fresh":
    note_col.caption(f"AI report by `{analysis['model']}`. Repeat runs are served from the cache.")
elif not report_hits:
    note_col.caption("Ask a question that matches the library to retrieve evidence for a report.")
elif settings.live:
    note_col.caption(f"One call to a free OpenRouter model with the {len(report_hits)} retrieved passages, "
                     "usually 20-60 seconds. Results are cached.")
else:
    note_col.caption("Demo mode: the AI report runs once an OpenRouter key is configured (see the 🧠 Report tab). "
                     "Everything else on this page is computed live.")

report = ResearchReport.model_validate(analysis["report"]) if analysis else None
report_md = report_markdown(report) if report else None
report_check: GroundingResult | None = check_text(report_md, index) if report_md else None

# ------------------------------------------------------------------------- headline
years = [s.year for s in library if s.year]
share = report_check.grounded_share if report_check else None
kpis = st.columns(6)
kpis[0].metric("Sources", len(library))
kpis[1].metric("Passages", len(index), help=f"Chunks of about {ss['chunk_words']} words with "
               f"{ss['overlap']} words of overlap, each tagged with its source and page.")
year_span = "—" if not years else str(min(years)) if min(years) == max(years) else f"{min(years)}–{str(max(years))[2:]}"
kpis[2].metric("Years", year_span,
               help="First-version years of the sources.")
kpis[3].metric("Topics", lit.k if len(library) >= 4 else "—",
               help="Topic clusters (k-means on LSA vectors, k chosen by silhouette score).")
kpis[4].metric("Authors", len({a for s in library for a in s.authors}))
kpis[5].metric("Grounded", f"{share:.0%}" if share is not None else "—",
               help="Share of the AI report's cited sentences that their cited passages support. "
                    "Needs an AI report.")
if report_check is not None and report_check.cited:
    pill(f"Grounding: {report_check.count('supported')} of {report_check.cited} cited sentences supported")
elif report_hits:
    pill(f"Evidence pack: {len(report_hits)} passages from "
         f"{len({h.chunk.sid for h in report_hits})} sources ready for the report")
st.write("")

tabs = st.tabs(["📚 Library", "🔎 Search & evidence", "🗺️ Literature map", "📊 Dataset insights", "🧠 Report",
                "✅ Grounding check"])
EMPTY = "The library is empty. Pick a sample, fetch papers from arXiv, or add your own documents in the sidebar."

# --------------------------------------------------------------------------- library
with tabs[0]:
    if not library:
        st.info(EMPTY)
    else:
        if corpus is not None and corpus.get("synthetic"):
            st.warning("These records are synthetic placeholders because arXiv was unreachable when the "
                       "snapshot was made. They are not real papers.")
        left, right = st.columns([1.4, 1])
        sids = [s.sid for s in library]
        if ss.get("inspect") not in sids:
            ss["inspect"] = sids[0]
        pick = left.selectbox("Inspect a source", sids, key="inspect",
                              format_func=lambda sid: f"{sid} · {by_sid[sid].title[:80]}")
        chosen = by_sid.get(pick, library[0])
        i = sids.index(chosen.sid)
        with left:
            meta = " · ".join(filter(None, [chosen.author_line(4), str(chosen.year or ""),
                                            ", ".join(chosen.categories[:3]), chosen.kind.upper()]))
            st.markdown(f"**{link(chosen.title, chosen.url)}**<br><span class='rp-small'>{html.escape(meta)}</span>"
                        f"<div class='rp-quote'>{html.escape(chosen.text[:1400])}"
                        f"{'…' if len(chosen.text) > 1400 else ''}</div>", unsafe_allow_html=True)
        with right:
            st.markdown("<div style='height: 1.9rem'></div>", unsafe_allow_html=True)
            card("🏷️ Keyphrases (TF-IDF)", [tags(lit.keyphrases[i]) or "—"], raw_html=True)
            similar = [f"{cite(library[j].sid)} {html.escape(library[j].title[:70])} "
                       f"<span class='rp-num rp-muted'>{sim:.0%}</span>" for j, sim in lit.similar_to(i, 4)]
            card("🔗 Most similar in this library (cosine)", similar, raw_html=True)
            if len(library) >= 4:
                c = int(lit.labels[i])
                card("🗺️ Topic cluster", [f"<span style='color:{charts.cluster_color(c)}'>●</span> "
                                          f"{c + 1} · {html.escape(lit.cluster_names.get(c, ''))}"], raw_html=True)

        st.text_input("Filter the library", key="lib_filter",
                      placeholder="Filter by title, author, category or keyphrase")
        needle = (ss.get("lib_filter") or "").strip().lower()
        rows = []
        for j, s in enumerate(library):
            hay = " ".join([s.title, *s.authors, *s.categories, *lit.keyphrases[j]]).lower()
            if needle and needle not in hay:
                continue
            rows.append([cite(s.sid), link(s.title, s.url), s.author_line(2), s.year or "—",
                         s.primary_category or s.kind.upper(), tags(lit.keyphrases[j][:3])])
        st.caption(f"Showing {len(rows)} of {len(library)} sources. Keyphrases are the highest-weighted TF-IDF "
                   "phrases of each title and abstract.")
        table(["Ref", "Title", "Authors", "Year", "Category", "Keyphrases"], rows, html_cols=(0, 1, 5))

# ---------------------------------------------------------------- search & evidence
with tabs[1]:
    if not library:
        st.info(EMPTY)
    else:
        if sample is not None:
            pill_key = f"suggested_{sample['slug']}"
            st.pills("Suggested questions", sample["qa_questions"], key=pill_key, on_change=_use_suggestion,
                     args=(pill_key,))
        st.text_input("Search the library", key="query", placeholder="Ask a question or type keywords")
        query = (ss.get("query") or "").strip()
        hits = index.search(query, ss["top_k"], ss["per_source"]) if query else []
        if not query:
            st.info("Type a question or keywords to retrieve evidence passages.")
        elif not hits:
            st.warning("No passage contains these terms. Try other wording or a broader question.")
        else:
            extract = extractive_answer(query, hits, index)
            left, right = st.columns([1.45, 1])
            with left:
                card("📌 Extractive answer (quoted from the sources, no AI)",
                     [f"{html.escape(e.text)} {cite(e.label)}" for e in extract] or ["No sentence matched."],
                     raw_html=True)
                st.caption("Sentences from the retrieved passages that cover the most query terms, weighted by "
                           "how rare each term is in the library (IDF).")
            with right:
                st.plotly_chart(charts.evidence_scores(hits), width="stretch")
                st.markdown("##### 🤖 AI answer")
                qa, qa_fresh = lookup(evidence_fingerprint("qa", query, hits))
                if qa:
                    with st.container(border=True):
                        st.markdown(with_citation_chips(qa["answer"]), unsafe_allow_html=True)
                    qa_check = check_text(qa["answer"], index)
                    label = "Saved answer" if qa_fresh == "saved" else "Answer"
                    st.caption(f"{label} by `{qa['model']}` from the passages below. Grounding check: "
                               f"{qa_check.count('supported')} of {qa_check.cited} cited sentences supported.")
                elif settings.live:
                    if st.button("✨ Run AI answer", key="run_qa"):
                        with st.spinner("Answering from the retrieved passages…"):
                            try:
                                llm = LLMClient(settings, app_name="ResearchPilot AI")
                                result = run_qa(llm, query, hits, by_sid)
                                ss["ai"][result["fingerprint"]] = result
                                st.rerun()
                            except LLMError as exc:
                                st.error(str(exc))
                    st.caption("One call to a free model; it sees only the passages listed below.")
                else:
                    ai_setup([
                        f"A short answer to “{html.escape(query)}” that cites the {len(hits)} passages below",
                        "A grounding score for that answer, computed as in ✅ Grounding check",
                    ])
            st.markdown("##### Evidence passages")
            top = hits[0].score
            rows = []
            for h in hits:
                src = by_sid[h.chunk.sid]
                passage = f"<b>{link(src.title, src.url)}</b><br>{highlight(h.chunk.text, query)}"
                rows.append([h.rank, cite(h.label), passage, score_bar(h.score / top, f"{h.score:.1f}")])
            table(["#", "Cite", "Passage (query terms highlighted)", "BM25"], rows, html_cols=(1, 2, 3))

# ----------------------------------------------------------------- literature map
with tabs[2]:
    if len(library) < 4:
        st.info("Add at least four sources to map the literature.")
    else:
        left, right = st.columns([1.5, 1])
        left.plotly_chart(charts.cluster_map(library, lit), width="stretch")
        with right:
            st.markdown("##### Topic clusters")
            ct = cluster_table(library, lit)
            table(["", "Cluster (distinctive phrases)", "Papers", "Years", "Central"],
                  [[f"<span style='color:{charts.cluster_color(r.cluster)};font-size:1.1rem'>●</span>",
                    html.escape(r.label), r.papers, f"<span style='white-space:nowrap'>{r.years}</span>",
                    cite(library[lit.representatives[r.cluster]].sid)]
                   for r in ct.itertuples()], html_cols=(0, 1, 3, 4))
            if lit.silhouette is not None:
                st.caption(f"k = {lit.k} has the best silhouette score ({lit.silhouette:.2f}) among k = 2 to "
                           f"{max(lit.silhouettes)}, the cap for {len(library)} papers (√(n/2)). Labels are the "
                           "phrases most over-represented in each cluster; *Central* is the paper closest to the "
                           "cluster centre and is labelled on the map.")
        left, right = st.columns(2)
        if lit.silhouettes:
            left.plotly_chart(charts.silhouette_chart(lit), width="stretch")
        right.plotly_chart(charts.timeline_chart(timeline(library, lit.labels), lit.cluster_names), width="stretch")
        left, right = st.columns(2)
        authors = top_authors(library, 8)
        if authors:
            left.plotly_chart(charts.count_bars(authors, "Most frequent authors"), width="stretch")
        categories = top_categories(library, 8)
        if categories:
            right.plotly_chart(charts.count_bars(categories, "arXiv categories", charts.ACCENT_DARK),
                               width="stretch")
        st.markdown("##### Most similar pairs")
        table(["Pair", "Similarity", "Shared keyphrases"],
              [[f"{cite(library[a].sid)} {html.escape(library[a].title[:60])}<br>{cite(library[b].sid)} "
                f"{html.escape(library[b].title[:60])}", score_bar(sim),
                tags(sorted(set(lit.keyphrases[a]) & set(lit.keyphrases[b]))) if
                set(lit.keyphrases[a]) & set(lit.keyphrases[b]) else "<span class='rp-muted'>—</span>"]
               for a, b, sim in lit.top_pairs(6)], html_cols=(0, 1, 2))

# ----------------------------------------------------------------- dataset insights
with tabs[3]:
    src_col, up_col = st.columns([1.2, 1])
    src_col.radio("Dataset", [DATASET_NAME, UPLOAD], key="data_choice", horizontal=True)
    df, data_id = None, None
    if ss["data_choice"] == UPLOAD:
        upload = up_col.file_uploader("CSV file", type=["csv", "txt"], key="data_upload")
        if upload is not None:
            raw = upload.getvalue()
            try:
                df = uploaded_dataset(raw)
                data_id = hashlib.sha256(raw).hexdigest()[:12]
            except Exception as exc:  # malformed CSVs come in many shapes
                st.error(f"Could not read {upload.name}: {exc}")
        else:
            st.info("Upload a CSV to profile it. Nothing leaves this app.")
    else:
        df, data_id = bundled_dataset(), "bundled"
        st.caption("⚠️ **Synthetic data**: 360 simulated lithium-ion cells generated by `scripts/make_dataset.py` "
                   "(seed 42) from a simple, documented ageing model. Useful for exploring the tools, not as "
                   "evidence about real batteries.")

    if df is not None and data_id is not None:
        info = dataset_insights(data_id, df)
        prof = info["profile"]
        numeric = ds.numeric_columns(df)
        groups = ds.group_columns(df)
        cats = [c for c in groups if ds.column_kind(df[c]) in ("categorical", "boolean")]
        m = st.columns(5)
        m[0].metric("Rows", f"{len(df):,}")
        m[1].metric("Columns", df.shape[1])
        m[2].metric("Numeric", len(numeric))
        m[3].metric("Categorical", int(prof["kind"].isin(["categorical", "boolean"]).sum()))
        total_cells = df.shape[0] * df.shape[1]
        m[4].metric("Missing", f"{df.isna().sum().sum() / total_cells:.1%}" if total_cells else "—",
                    help="Share of empty cells in the whole table.")
        card("🔍 What stands out (computed)", info["findings"])

        st.markdown("##### Schema")
        table(["Column", "Kind", "Missing", "Unique", "Summary"],
              [[r.column, r.kind, score_bar(r.missing_pct, f"{r.missing_pct:.1%}"), r.unique, r.summary]
               for r in prof.itertuples()], html_cols=(2,))

        st.markdown("##### Distributions")
        dist_options = [c for c in df.columns if ds.column_kind(df[c]) in ("numeric", "categorical", "boolean")]
        if dist_options:
            default = dist_options.index("capacity_retention_pct") if "capacity_retention_pct" in dist_options else 0
            left, right = st.columns([1.6, 1])
            col = left.selectbox("Column", dist_options, index=default, key=f"dist_{data_id}")
            is_num = col in numeric
            left.plotly_chart(charts.distribution(df, col, is_num), width="stretch")
            s = df[col].dropna()
            if is_num:
                stats_rows = [["Count", f"{len(s):,}"], ["Mean", f"{s.mean():.3g}"], ["Median", f"{s.median():.3g}"],
                              ["Std. deviation", f"{s.std():.3g}"], ["Skewness", f"{s.skew():.2f}"],
                              ["Min / max", f"{s.min():.3g} / {s.max():.3g}"]]
                skew = s.skew()
                shape = ("roughly symmetric" if abs(skew) < 0.5 else
                         "skewed to the right (a long tail of high values)" if skew > 0 else
                         "skewed to the left (a long tail of low values)")
                with right:
                    st.markdown("<div style='height: 4.6rem'></div>", unsafe_allow_html=True)
                    table(["Statistic", "Value"], stats_rows)
                    st.caption(f"The distribution is {shape}.")
            else:
                counts = s.astype(str).value_counts()
                with right:
                    st.markdown("<div style='height: 4.6rem'></div>", unsafe_allow_html=True)
                    table(["Category", "Rows", "Share"],
                          [[k, f"{v:,}", f"{v / len(s):.0%}"] for k, v in counts.head(10).items()])

        if len(numeric) >= 2:
            st.markdown("##### Correlations")
            method = st.radio("Coefficient", ["pearson", "spearman"], key="corr_method", horizontal=True,
                              format_func=lambda x: "Pearson (linear)" if x == "pearson" else "Spearman (rank)")
            left, right = st.columns([1.25, 1])
            left.plotly_chart(charts.corr_heatmap(ds.correlation_matrix(df, method), method), width="stretch")
            pairs = info["pairs"]
            with right:
                st.markdown("<div style='height: 2.6rem'></div>", unsafe_allow_html=True)
                table(["Strongest pairs", "r", "ρ", "Strength"],
                      [[f"{p.a} × {p.b}", f"{p.pearson:.2f}", f"{p.spearman:.2f}", p.strength]
                       for p in pairs.head(7).itertuples()])
            card("📝 Readouts", [p.readout for p in pairs.head(4).itertuples()])

        out = info["outliers"]
        if not out.empty:
            st.markdown("##### Outliers (Tukey fences, 1.5 × IQR)")
            table(["Column", "Q1", "Q3", "Fences", "Outliers", "Most extreme"],
                  [[r.column, f"{r.q1:.3g}", f"{r.q3:.3g}", f"{r.low_fence:.3g} to {r.high_fence:.3g}",
                    score_bar(r.share, f"{r.outliers} ({r.share:.1%})"), r.examples or "—"]
                   for r in out.itertuples()], html_cols=(4,))

        st.markdown("##### Significance tests")
        left, right = st.columns(2)
        with left:
            st.markdown("**Does a measure differ between groups?**")
            if numeric and groups:
                v_default = numeric.index("capacity_retention_pct") if "capacity_retention_pct" in numeric else 0
                g_default = groups.index("chemistry") if "chemistry" in groups else 0
                a1, a2 = st.columns(2)
                value_col = a1.selectbox("Measure", numeric, index=v_default, key=f"tv_{data_id}")
                group_col = a2.selectbox("Groups", groups, index=g_default, key=f"tg_{data_id}")
                try:
                    res = ds.compare_groups(df, value_col, group_col)
                    (st.success if res.significant else st.info)(res.readout)
                    for caveat in res.caveats:
                        st.caption(f"⚠️ {caveat}")
                    st.plotly_chart(charts.group_box(df, value_col, group_col), width="stretch")
                except ValueError as exc:
                    st.warning(str(exc))
            else:
                st.caption("Needs a numeric column and a column with 2-12 groups.")
        with right:
            st.markdown("**Are two categorical columns related?**")
            if len(cats) >= 2:
                c_a = cats.index("chemistry") if "chemistry" in cats else 0
                c_b = cats.index("form_factor") if "form_factor" in cats else 1
                b1, b2 = st.columns(2)
                cat_a = b1.selectbox("Column A", cats, index=c_a, key=f"ca_{data_id}")
                cat_b = b2.selectbox("Column B", cats, index=c_b, key=f"cb_{data_id}")
                try:
                    res = ds.chi_square(df, cat_a, cat_b)
                    (st.success if res.significant else st.info)(res.readout)
                    for caveat in res.caveats:
                        st.caption(f"⚠️ {caveat}")
                    crosstab = res.detail
                    table([f"{cat_a} \\ {cat_b}", *map(str, crosstab.columns)],
                          [[str(k), *[int(v) for v in row]] for k, row in crosstab.iterrows()])
                except ValueError as exc:
                    st.warning(str(exc))
            else:
                st.caption("Needs two columns with 2-12 categories.")

# ---------------------------------------------------------------------------- report
with tabs[4]:
    if report is not None:
        if freshness == "saved":
            st.caption(f"Saved AI report · `{analysis['model']}` · {analysis['generated_at'][:10]}")
        st.markdown(f"#### {html.escape(report.question)}")
        if report.summary:
            st.markdown(f"<div class='rp-quote'>{with_citation_chips(report.summary)}</div>", unsafe_allow_html=True)
        if report.key_findings:
            st.markdown("##### Key findings")
            table(["Finding", "Evidence"],
                  [[with_citation_chips(finding_line(f)), badge(f.strength, f.strength.title())]
                   for f in report.key_findings], html_cols=(0, 1))
        r1 = st.columns(2)
        with r1[0]:
            card("🧪 Methods overview", [with_citation_chips(x) for x in report.methods_overview], raw_html=True)
        with r1[1]:
            card("⚖️ Disagreements", [with_citation_chips(x) for x in report.disagreements], raw_html=True)
        r2 = st.columns(2)
        with r2[0]:
            card("🕳️ Gaps", [with_citation_chips(x) for x in report.gaps], raw_html=True)
        with r2[1]:
            card("🔭 Future work", [with_citation_chips(x) for x in report.future_work], raw_html=True)
        if report_check is not None and report_check.cited:
            st.caption(f"Grounding: {report_check.count('supported')} of {report_check.cited} cited sentences are "
                       "supported by the passages they cite. See ✅ Grounding check for each sentence.")
    elif not report_hits:
        st.info("Ask a research question that matches the library to retrieve an evidence pack.")
    elif settings.live:
        st.info("Click ✨ Run AI report above. The model sees only the evidence pack below and must cite it.")
    else:
        ai_setup([
            f"A structured report on “{html.escape(question)}”, written only from the {len(report_hits)} "
            "evidence passages below",
            "Key findings with inline citations such as " + cite("S3") + " and an evidence-strength rating",
            "A methods overview, disagreements between sources, gaps and future work",
            "A sentence-by-sentence citation check in ✅ Grounding check, and the Grounded score in the header",
        ])

    if report_hits:
        with st.expander(f"📦 Evidence pack for the report: {len(report_hits)} passages (computed)",
                         expanded=report is None):
            table(["Cite", "Source", "Passage"],
                  [[cite(h.label), html.escape(by_sid[h.chunk.sid].title), highlight(h.chunk.text, question)]
                   for h in report_hits], html_cols=(0, 2))
    if library:
        brief = build_brief(library_name, question or "(no question)", library, lit, report_hits,
                            extractive_answer(question, report_hits, index) if report_hits else [],
                            report_md, report_check, analysis["model"] if analysis else None)
        st.download_button("⬇️ Download research brief (Markdown)", brief, file_name="researchpilot-brief.md",
                           mime="text/markdown", type="primary")

# -------------------------------------------------------------------- grounding check
with tabs[5]:
    modes = ["Example or pasted text"] + (["AI report"] if report_md else [])
    if ss.get("ground_mode") not in modes:
        ss["ground_mode"] = modes[0]
    st.radio("Text to check", modes, key="ground_mode", horizontal=True,
             help="Once an AI report has been written, it can be checked here as well.")
    if ss["ground_mode"] == "AI report":
        text = report_md
        st.caption("Checking the AI report shown in the 🧠 Report tab.")
    else:
        st.text_area("Markdown with citations such as [S3] or [S3 p.2]", key="ground_text", height=230,
                     help="Paste any paragraph that cites sources from this library.")
        text = ss.get("ground_text") or ""
        if sample is not None:
            st.caption("The example mixes faithful sentences with planted errors: a wrong source, an invented "
                       "number, a flipped claim, an unknown reference and an uncited claim.")
    result = check_text(text, index, set(by_sid)) if library else GroundingResult()
    if not library:
        st.info(EMPTY)
    elif not result.checks:
        st.info("No sentences with citations (or factual-looking uncited sentences) found.")
    else:
        g = st.columns(5)
        g[0].metric("Checked", len(result.checks))
        g[1].metric("Supported", result.count("supported"))
        g[2].metric("Weak", result.count("weak"))
        g[3].metric("Not supported", result.count("unsupported") + result.count("unknown_source"))
        g[4].metric("Grounded", f"{result.grounded_share:.0%}" if result.grounded_share is not None else "—",
                    help="Supported sentences ÷ sentences with a citation.")
        left, right = st.columns([1.35, 1])
        left.plotly_chart(charts.support_chart(result.checks), width="stretch")
        with right:
            card("How the support score works", [
                "Support = the IDF-weighted share of the sentence's content words found in the cited passage(s): "
                "rare, specific words count for more than common ones.",
                f"Supported at {SUPPORTED_AT:.0%} or more; weak from {WEAK_AT:.0%}; below that, not supported.",
                "Numbers in the sentence must appear in the cited passage, and a sentence that flips the "
                "polarity of the source (“did not outperform” vs “outperformed”) is downgraded.",
                "When another passage would support the sentence clearly better, it is suggested.",
            ])
        rows = []
        for n, c in enumerate(result.checks, start=1):
            note = html.escape(c.note) if c.note else "<span class='rp-muted'>—</span>"
            if c.suggestion:
                note = note.replace(html.escape(f"[{c.suggestion}]"), cite(c.suggestion))
            rows.append([n, with_citation_chips(c.sentence), score_bar(c.score), badge(c.verdict, VERDICTS[c.verdict]),
                         note])
        table(["#", "Sentence", "Support", "Verdict", "Notes"], rows, html_cols=(1, 2, 3, 4))

st.divider()
st.caption(
    "ResearchPilot AI is a research aid. Retrieval is lexical (BM25) and the grounding check measures word overlap, "
    "so read the cited passages before relying on a claim. AI-written text can be wrong."
)
