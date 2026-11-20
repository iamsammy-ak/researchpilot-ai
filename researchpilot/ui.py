"""Small HTML/CSS helpers for a polished Streamlit look (works in light and dark themes)."""

from __future__ import annotations

import html
import re

import streamlit as st

from .grounding import CITATION, parse_citations
from .text import STOPWORDS, stem, tokenize

ACCENT = "#2563EB"
ACCENT_DARK = "#1E3A8A"

_CSS = f"""
<style>
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1240px; }}
.rp-hero {{
  border-radius: 22px; padding: 30px 34px; margin-bottom: 1.2rem; color: #fff;
  background:
    radial-gradient(circle at 88% 18%, rgba(255,255,255,0.20), transparent 38%),
    radial-gradient(circle at 10% 110%, rgba(255,255,255,0.10), transparent 40%),
    linear-gradient(135deg, {ACCENT} 0%, {ACCENT_DARK} 100%);
  box-shadow: 0 18px 40px -22px rgba(30,58,138,0.65);
}}
.rp-hero .rp-title {{ font-size: 2.15rem; font-weight: 750; letter-spacing: -0.02em; line-height: 1.15; }}
.rp-hero .rp-sub {{ margin-top: 8px; font-size: 1.03rem; opacity: 0.92; max-width: 800px; }}
.rp-chip {{
  display: inline-block; margin: 14px 8px 0 0; padding: 4px 12px; border-radius: 999px;
  background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.22); font-size: 0.8rem;
}}
div[data-testid="stMetric"] {{
  border: 1px solid rgba(37,99,235,0.20); border-radius: 16px; padding: 14px 16px;
  background: linear-gradient(180deg, rgba(37,99,235,0.06), rgba(37,99,235,0.015));
}}
div[data-testid="stMetricLabel"] p {{ font-size: 0.82rem; opacity: 0.8; }}
div[data-testid="stMetricValue"] {{ font-size: 1.6rem; font-weight: 650; }}
.rp-card {{
  border: 1px solid rgba(148,163,184,0.35); border-radius: 16px; padding: 14px 18px; height: 100%;
  margin-bottom: 0.8rem;
}}
.rp-card h4 {{ margin: 0 0 6px 0; font-size: 0.95rem; }}
.rp-card ul {{ margin: 0; padding-left: 1.1rem; }}
.rp-card li {{ margin-bottom: 4px; }}
.rp-pill {{
  display: inline-block; padding: 5px 14px; border-radius: 999px; font-weight: 600; font-size: 0.9rem;
  background: rgba(37,99,235,0.10); color: {ACCENT}; border: 1px solid rgba(37,99,235,0.25);
}}
.rp-table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; margin: 0.2rem 0 1rem 0; }}
.rp-table th {{
  text-align: left; font-weight: 600; opacity: 0.7; padding: 8px 10px;
  border-bottom: 1px solid rgba(148,163,184,0.45);
}}
.rp-table td {{ padding: 9px 10px; border-bottom: 1px solid rgba(148,163,184,0.2); vertical-align: top; }}
.rp-table tr:hover td {{ background: rgba(37,99,235,0.03); }}
.rp-table a {{ color: inherit; text-decoration: none; border-bottom: 1px dotted rgba(37,99,235,0.6); }}
.rp-bar {{
  display: inline-block; width: 64px; height: 8px; border-radius: 99px; overflow: hidden;
  background: rgba(37,99,235,0.13); vertical-align: middle; margin-right: 8px;
}}
.rp-bar > div {{ height: 100%; border-radius: 99px; background: {ACCENT}; }}
.rp-num {{ font-weight: 650; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.rp-muted {{ opacity: 0.55; }}
.rp-small {{ font-size: 0.8rem; opacity: 0.7; }}
.rp-cite {{
  display: inline-block; padding: 0 7px; margin: 1px 2px; border-radius: 6px; font-size: 0.78rem;
  font-weight: 650; white-space: nowrap; background: rgba(37,99,235,0.10); color: {ACCENT};
  border: 1px solid rgba(37,99,235,0.22); font-variant-numeric: tabular-nums;
}}
.rp-tag {{
  display: inline-block; padding: 1px 8px; margin: 2px 3px 2px 0; border-radius: 999px; font-size: 0.76rem;
  background: rgba(148,163,184,0.16); border: 1px solid rgba(148,163,184,0.3);
}}
.rp-badge {{ display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 650;
  white-space: nowrap; }}
.rp-supported {{ background: rgba(5,150,105,0.12); color: #047857; }}
.rp-weak {{ background: rgba(217,119,6,0.13); color: #B45309; }}
.rp-unsupported {{ background: rgba(220,38,38,0.12); color: #B91C1C; }}
.rp-unknown_source {{ background: rgba(124,58,237,0.12); color: #6D28D9; }}
.rp-uncited {{ background: rgba(100,116,139,0.14); color: #475569; }}
.rp-strong {{ background: rgba(5,150,105,0.12); color: #047857; }}
.rp-moderate {{ background: rgba(37,99,235,0.10); color: {ACCENT}; }}
.rp-limited {{ background: rgba(217,119,6,0.13); color: #B45309; }}
mark.rp-hit {{ background: rgba(250,204,21,0.38); color: inherit; padding: 0 2px; border-radius: 3px; }}
.rp-quote {{ border-left: 3px solid rgba(37,99,235,0.45); padding: 2px 0 2px 12px; margin: 6px 0 10px 0; }}
</style>
"""

_WORD = re.compile(r"[A-Za-z0-9]+")


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, chips: list[str]) -> None:
    chip_html = "".join(f'<span class="rp-chip">{html.escape(c)}</span>' for c in chips)
    st.markdown(
        f'<div class="rp-hero"><div class="rp-title">{html.escape(title)}</div>'
        f'<div class="rp-sub">{html.escape(subtitle)}</div>{chip_html}</div>',
        unsafe_allow_html=True,
    )


def pill(text: str) -> None:
    st.markdown(f'<span class="rp-pill">{html.escape(text)}</span>', unsafe_allow_html=True)


def table(headers: list[str], rows: list[list], html_cols: tuple[int, ...] = ()) -> None:
    """Render a wrapping HTML table. Cells are escaped unless their column is in `html_cols`."""
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{cell if i in html_cols else html.escape(str(cell))}</td>" for i, cell in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    st.markdown(f'<table class="rp-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>',
                unsafe_allow_html=True)


def score_bar(value: float | None, label: str | None = None) -> str:
    """A 0-1 value as a small bar plus its label (a percentage by default)."""
    if value is None:
        return '<span class="rp-muted">—</span>'
    width = max(0.0, min(100.0, value * 100))
    text = label if label is not None else f"{value:.0%}"
    return (f'<span style="white-space:nowrap"><span class="rp-bar"><div style="width:{width:.0f}%"></div></span>'
            f'<span class="rp-num">{html.escape(text)}</span></span>')


def badge(key: str, text: str) -> str:
    return f'<span class="rp-badge rp-{html.escape(key)}">{html.escape(text)}</span>'


def tags(items: list[str]) -> str:
    return "".join(f'<span class="rp-tag">{html.escape(i)}</span>' for i in items) or '<span class="rp-muted">—</span>'


def cite(label: str) -> str:
    return f'<span class="rp-cite">{html.escape(label)}</span>'


def with_citation_chips(text: str) -> str:
    """Escape text and turn [S3 p.2] markers into citation chips."""
    escaped = html.escape(text)
    return CITATION.sub(lambda m: "".join(cite(c.label) for c in parse_citations(m.group(0))), escaped)


def highlight(text: str, query: str) -> str:
    """Escape text and mark the words whose stem matches a query term."""
    q_terms = set(tokenize(query))
    out, last = [], 0
    for m in _WORD.finditer(text):
        out.append(html.escape(text[last : m.start()]))
        word = m.group(0)
        lowered = word.lower()
        if lowered not in STOPWORDS and stem(lowered) in q_terms:
            out.append(f'<mark class="rp-hit">{html.escape(word)}</mark>')
        else:
            out.append(html.escape(word))
        last = m.end()
    out.append(html.escape(text[last:]))
    return "".join(out)


def card(title: str, items: list[str], raw_html: bool = False) -> None:
    body = "".join(f"<li>{i if raw_html else html.escape(i)}</li>" for i in items) or "<li>—</li>"
    st.markdown(f'<div class="rp-card"><h4>{html.escape(title)}</h4><ul>{body}</ul></div>', unsafe_allow_html=True)


def link(title: str, url: str) -> str:
    if not url:
        return html.escape(title)
    return f'<a href="{html.escape(url, quote=True)}" target="_blank">{html.escape(title)}</a>'
