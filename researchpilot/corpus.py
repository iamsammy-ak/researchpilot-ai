"""The research library: arXiv records, uploaded PDFs and pasted text, each with a stable source label.

Sources are numbered S1, S2, ... in library order, and every later citation ([S3] or [S3 p.2]) refers to
these labels. PDFs keep their page texts so citations can point at a page.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Iterable
from dataclasses import dataclass, replace

from .text import normalise_space


@dataclass(frozen=True)
class Source:
    sid: str  # "S1", "S2", ... assigned by build_library
    title: str
    text: str  # abstract for arXiv records, full text for PDFs and pasted text
    authors: tuple[str, ...] = ()
    published: str = ""  # ISO date, YYYY-MM-DD
    categories: tuple[str, ...] = ()  # primary category first
    url: str = ""
    kind: str = "arxiv"  # arxiv | pdf | text | synthetic
    pages: tuple[str, ...] = ()  # page texts for PDFs; empty when the source has no pages

    @property
    def year(self) -> int | None:
        head = self.published[:4]
        return int(head) if head.isdigit() else None

    @property
    def primary_category(self) -> str:
        return self.categories[0] if self.categories else ""

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def author_line(self, limit: int = 3) -> str:
        if not self.authors:
            return "Unknown authors" if self.kind == "arxiv" else "Your document"
        shown = ", ".join(self.authors[:limit])
        return shown + (" et al." if len(self.authors) > limit else "")

    def page_texts(self) -> list[tuple[int | None, str]]:
        """(page number, text) pairs; page is None for sources without pages."""
        if self.pages:
            return [(i + 1, text) for i, text in enumerate(self.pages)]
        return [(None, self.text)]


def source_from_record(record: dict) -> Source:
    """Turn a stored arXiv record (see arxiv.parse_feed) into a Source without a label yet."""
    return Source(
        sid="",
        title=normalise_space(record.get("title", "")) or "Untitled",
        text=normalise_space(record.get("abstract", "")),
        authors=tuple(record.get("authors", [])),
        published=str(record.get("published", ""))[:10],
        categories=tuple(record.get("categories", [])),
        url=record.get("url", ""),
        kind=record.get("kind", "arxiv"),
    )


def build_library(records: Iterable[dict], extra: Iterable[Source] = ()) -> list[Source]:
    """Number every source S1..Sn: corpus records first, then the user's own documents."""
    sources = [source_from_record(r) for r in records] + list(extra)
    return [replace(s, sid=f"S{i}") for i, s in enumerate(sources, start=1)]


def from_text(text: str, title: str = "") -> Source:
    text = text.strip()
    if not text:
        raise ValueError("the pasted text is empty")
    first_line = text.splitlines()[0].strip()
    return Source(sid="", title=(title.strip() or first_line[:90] or "Pasted text"), text=normalise_space(text),
                  kind="text")


def read_pdf(data: bytes, name: str = "document.pdf") -> Source:
    """Extract page texts from a PDF. Scanned PDFs without a text layer raise ValueError."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = tuple(normalise_space(page.extract_text() or "") for page in reader.pages)
    if not any(pages):
        raise ValueError(f"{name}: no extractable text (is it a scanned PDF?)")
    meta_title = ""
    try:
        meta_title = normalise_space(str((reader.metadata or {}).get("/Title") or ""))
    except Exception:  # broken metadata should not block the text
        meta_title = ""
    fallback = name.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").strip()
    return Source(
        sid="",
        title=meta_title or fallback or "Uploaded PDF",
        text=" ".join(p for p in pages if p),
        kind="pdf",
        pages=pages,
    )


def library_fingerprint(sources: Iterable[Source]) -> str:
    payload = [[s.sid, s.title, s.text, list(s.pages)] for s in sources]
    return hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()[:16]
