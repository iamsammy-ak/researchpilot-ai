"""Plotly figures used by the app. Colours follow the ResearchPilot blue palette."""

from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .corpus import Source
from .grounding import SUPPORTED_AT, VERDICTS, WEAK_AT, SentenceCheck
from .litmap import LiteratureMap
from .retrieval import Hit

ACCENT = "#2563EB"
ACCENT_SOFT = "#93C5FD"
ACCENT_DARK = "#1E3A8A"
MUTED = "#94A3B8"
FONT = "Inter, 'Segoe UI', system-ui, sans-serif"
CLUSTER_COLORS = ["#2563EB", "#F59E0B", "#10B981", "#EF4444", "#8B5CF6", "#06B6D4", "#EC4899", "#84CC16"]
VERDICT_COLORS = {
    "supported": "#059669",
    "weak": "#D97706",
    "unsupported": "#DC2626",
    "unknown_source": "#7C3AED",
    "uncited": "#94A3B8",
}


def _style(fig: go.Figure, height: int = 340, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        title=dict(text=title, x=0, font=dict(size=15)) if title else None,
        margin=dict(l=8, r=8, t=48 if title else 16, b=8),
        font=dict(family=FONT, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font_family=FONT),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.25)", zeroline=False)
    return fig


def cluster_color(c: int) -> str:
    return CLUSTER_COLORS[c % len(CLUSTER_COLORS)]


def _wrap(text: str, width: int = 60) -> str:
    return "<br>".join(textwrap.wrap(text, width)) or text


def cluster_map(sources: list[Source], lit: LiteratureMap) -> go.Figure:
    fig = go.Figure()
    for c in range(lit.k):
        idx = np.flatnonzero(lit.labels == c)
        if len(idx) == 0:
            continue
        central = lit.representatives.get(c)
        fig.add_scatter(
            x=lit.coords[idx, 0],
            y=lit.coords[idx, 1],
            mode="markers+text",
            text=[sources[i].sid if i == central else "" for i in idx],
            hovertext=[sources[i].sid for i in idx],
            textposition="top center",
            textfont=dict(size=11, color=cluster_color(c)),
            marker=dict(size=13, color=cluster_color(c), line=dict(width=1.5, color="white"), opacity=0.9),
            name=f"{c + 1} · {lit.cluster_names.get(c, '')}",
            customdata=[[_wrap(sources[i].title), sources[i].year or "—"] for i in idx],
            hovertemplate="<b>%{hovertext}</b> · %{customdata[1]}<br>%{customdata[0]}<extra></extra>",
        )
    fig.update_xaxes(showticklabels=False, title="Topic axis 1 (PCA of LSA vectors)")
    fig.update_yaxes(showticklabels=False, title="Topic axis 2", showgrid=False)
    _style(fig, height=470, title="Papers placed by content similarity")
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0, font=dict(size=12)),
                      margin=dict(l=8, r=8, t=48, b=8))
    return fig


def silhouette_chart(lit: LiteratureMap) -> go.Figure:
    ks = sorted(lit.silhouettes)
    fig = go.Figure(
        go.Bar(
            x=[str(k) for k in ks],
            y=[lit.silhouettes[k] for k in ks],
            marker_color=[ACCENT if k == lit.k else ACCENT_SOFT for k in ks],
            text=[f"{lit.silhouettes[k]:.2f}" for k in ks],
            textposition="outside",
            hovertemplate="k = %{x}: silhouette %{y:.3f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Number of clusters (k)")
    fig.update_yaxes(title="Mean silhouette", rangemode="tozero")
    top = max(lit.silhouettes.values(), default=0)
    fig.update_yaxes(range=[0, top * 1.25 if top > 0 else 1])
    return _style(fig, height=300, title="Choosing k: higher silhouette = better separated")


def timeline_chart(tl: pd.DataFrame, names: dict[int, str]) -> go.Figure:
    fig = go.Figure()
    for c in sorted(tl["cluster"].unique()) if not tl.empty else []:
        part = tl[tl["cluster"] == c]
        fig.add_bar(x=part["year"], y=part["papers"], name=f"{int(c) + 1}", marker_color=cluster_color(int(c)),
                    hovertemplate=f"{names.get(int(c), '')}<br>%{{x}}: %{{y}} papers<extra></extra>")
    fig.update_layout(barmode="stack", showlegend=False)
    fig.update_xaxes(title="Year of first version", dtick=1)
    fig.update_yaxes(title="Papers")
    return _style(fig, height=300, title="Publication timeline by cluster")


def count_bars(pairs: list[tuple[str, int]], title: str, color: str = ACCENT) -> go.Figure:
    data = list(reversed(pairs))
    fig = go.Figure(
        go.Bar(
            y=[name for name, _ in data],
            x=[count for _, count in data],
            orientation="h",
            marker_color=color,
            text=[count for _, count in data],
            textposition="outside",
            hovertemplate="%{y}: %{x}<extra></extra>",
        )
    )
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showgrid=False, automargin=True)
    return _style(fig, height=max(240, 26 * len(pairs) + 70), title=title)


def evidence_scores(hits: list[Hit]) -> go.Figure:
    data = list(reversed(hits))
    fig = go.Figure(
        go.Bar(
            y=[f"#{h.rank} {h.label}" for h in data],  # rank prefix keeps repeated sources apart
            x=[h.score for h in data],
            orientation="h",
            marker_color=[ACCENT if h.rank <= 3 else ACCENT_SOFT for h in data],
            text=[f"{h.score:.1f}" for h in data],
            textposition="outside",
            hovertemplate="%{y}: BM25 %{x:.2f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="BM25 score", showgrid=True, gridcolor="rgba(148,163,184,0.25)")
    fig.update_yaxes(showgrid=False, automargin=True, type="category")
    top = max((h.score for h in hits), default=1)
    fig.update_xaxes(range=[0, top * 1.18])
    return _style(fig, height=max(260, 30 * len(hits) + 80), title="Retrieval scores")


def distribution(df: pd.DataFrame, column: str, numeric: bool) -> go.Figure:
    s = df[column].dropna()
    if numeric:
        counts, edges = np.histogram(s.astype(float), bins=min(30, max(5, s.nunique())))
        fig = go.Figure(go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=counts, width=np.diff(edges) * 0.96,
                               marker_color=ACCENT, opacity=0.85,
                               customdata=np.column_stack([edges[:-1], edges[1:]]),
                               hovertemplate="%{customdata[0]:.3g} to %{customdata[1]:.3g}: %{y} rows<extra></extra>"))
        fig.update_yaxes(range=[0, max(counts.max(), 1) * 1.3])  # headroom so the mean/median labels clear the bars
        mean, median = float(s.mean()), float(s.median())
        fig.add_vline(x=mean, line=dict(color=ACCENT_DARK, width=2), annotation_text="mean",
                      annotation_position="top right" if mean >= median else "top left")
        fig.add_vline(x=median, line=dict(color="#F59E0B", width=2, dash="dash"), annotation_text="median",
                      annotation_position="top left" if mean >= median else "top right")
        fig.update_yaxes(title="Rows")
    else:
        counts = s.astype(str).value_counts().head(20)
        fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=ACCENT, text=counts.values,
                               textposition="outside", hovertemplate="%{x}: %{y} rows<extra></extra>"))
        fig.update_yaxes(title="Rows", range=[0, counts.max() * 1.18 if len(counts) else 1])
    fig.update_xaxes(title=column)
    return _style(fig, height=320, title=f"Distribution of {column}")


def corr_heatmap(corr: pd.DataFrame, method: str) -> go.Figure:
    names = [c.replace("_", " ") for c in corr.columns]
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=names,
            y=[_wrap(n, 16) for n in names],
            zmin=-1,
            zmax=1,
            colorscale=[[0, "#DC2626"], [0.5, "#F8FAFC"], [1, ACCENT]],
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>",
            colorbar=dict(thickness=12, len=0.8),
        )
    )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, tickangle=-35, tickfont=dict(size=11))
    fig.update_yaxes(tickfont=dict(size=11))
    name = "Pearson r" if method == "pearson" else "Spearman ρ"
    return _style(fig, height=500, title=f"{name} between numeric columns")


def group_box(df: pd.DataFrame, value: str, group: str) -> go.Figure:
    fig = go.Figure()
    data = df[[value, group]].dropna()
    for i, (name, part) in enumerate(data.groupby(group, sort=True)):
        fig.add_box(y=part[value], name=str(name), marker_color=cluster_color(i), boxmean=True, boxpoints="outliers")
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title=value)
    fig.update_xaxes(title=group)
    return _style(fig, height=330, title=f"{value} by {group}")


def support_chart(checks: list[SentenceCheck]) -> go.Figure:
    data = list(reversed(list(enumerate(checks, start=1))))
    fig = go.Figure(
        go.Bar(
            y=[f"#{i}" for i, _ in data],
            x=[c.score if c.score is not None else 0 for _, c in data],
            orientation="h",
            marker_color=[VERDICT_COLORS[c.verdict] for _, c in data],
            text=[f"{c.score:.0%}" if c.score is not None else VERDICTS[c.verdict] for _, c in data],
            textposition="outside",
            customdata=[[_wrap(c.claim, 70), VERDICTS[c.verdict]] for _, c in data],
            hovertemplate="%{y} · %{customdata[1]}<br>%{customdata[0]}<extra></extra>",
        )
    )
    for x, label in ((WEAK_AT, "weak"), (SUPPORTED_AT, "supported")):
        fig.add_vline(x=x, line=dict(color=MUTED, width=1, dash="dot"), annotation_text=label,
                      annotation_position="top", annotation_font_size=11)
    fig.update_xaxes(range=[0, 1.28], tickformat=".0%", title="Support from the cited passages",
                     showgrid=True, gridcolor="rgba(148,163,184,0.25)")
    fig.update_yaxes(showgrid=False, type="category")
    return _style(fig, height=max(260, 28 * len(checks) + 90), title="Support score per sentence")
