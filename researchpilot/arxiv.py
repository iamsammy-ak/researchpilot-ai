"""Keyless arXiv search through the public Atom API, parsed with feedparser.

arXiv asks API clients to wait about three seconds between requests, so the client enforces a
minimum interval across the whole process and caches every response on disk: repeating a search
costs no request at all.
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from urllib.parse import urlencode

import feedparser
import requests

from .text import STOPWORDS, normalise_space, words

API_URL = "https://export.arxiv.org/api/query"
MIN_INTERVAL_S = 3.0
PAGE_SIZE = 100
DEFAULT_CACHE_DIR = Path(".cache") / "arxiv"
USER_AGENT = "ResearchPilot/0.1 (+https://github.com/iamsammy-ak/researchpilot-ai)"
_FIELD = re.compile(r"\b(?:all|ti|abs|au|cat|co|jr|rn|id):", re.I)
_VERSION = re.compile(r"v\d+$")
_last_request = [0.0]  # monotonic time of the last network request in this process


class ArxivError(RuntimeError):
    """The arXiv API could not be reached or rejected the query."""


def build_search_query(text: str) -> str:
    """Plain words become an AND of all-field terms; "quoted phrases" stay phrases.
    Text that already uses arXiv field syntax (ti:, abs:, au:, cat:, ...) is passed through."""
    text = normalise_space(text)
    if not text:
        raise ValueError("the search query is empty")
    if _FIELD.search(text):
        return text
    phrases = re.findall(r'"([^"]+)"', text)
    rest = re.sub(r'"[^"]*"', " ", text)
    terms = [f'all:"{normalise_space(p)}"' for p in phrases if p.strip()]
    terms += [f"all:{w}" for w in words(rest) if w not in STOPWORDS]
    if not terms:
        raise ValueError("the search query has no searchable words")
    return " AND ".join(terms)


def query_url(search_query: str, max_results: int, start: int = 0, sort_by: str = "relevance") -> str:
    params = {
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": "descending",
    }
    return f"{API_URL}?{urlencode(params)}"


def parse_feed(xml_text: str) -> list[dict]:
    """Turn an arXiv Atom response into plain records (title, authors, date, categories, abstract, links)."""
    feed = feedparser.parse(xml_text)
    records = []
    for entry in feed.entries:
        entry_id = entry.get("id", "")
        if "/api/errors" in entry_id or normalise_space(entry.get("title", "")).lower() == "error":
            raise ArxivError("arXiv rejected the query: " + normalise_space(entry.get("summary", "")))
        arxiv_id = _VERSION.sub("", entry_id.rsplit("/abs/", 1)[-1])
        categories = [t.get("term", "") for t in entry.get("tags", []) if t.get("term")]
        primary = (entry.get("arxiv_primary_category") or {}).get("term")
        if primary:
            categories = [primary] + [c for c in categories if c != primary]
        pdf_url = next((link.get("href", "") for link in entry.get("links", []) if link.get("title") == "pdf"), "")
        records.append(
            {
                "arxiv_id": arxiv_id,
                "title": normalise_space(entry.get("title", "")),
                "authors": [normalise_space(a.get("name", "")) for a in entry.get("authors", []) if a.get("name")],
                "published": entry.get("published", "")[:10],
                "updated": entry.get("updated", "")[:10],
                "categories": categories,
                "abstract": normalise_space(entry.get("summary", "")),
                "url": entry.get("link", f"https://arxiv.org/abs/{arxiv_id}"),
                "pdf_url": pdf_url,
            }
        )
    return records


class ArxivClient:
    def __init__(
        self,
        cache_dir: Path | None = DEFAULT_CACHE_DIR,
        min_interval: float = MIN_INTERVAL_S,
        timeout: float = 30.0,
    ) -> None:
        self.cache_dir = cache_dir
        self.min_interval = min_interval
        self.timeout = timeout
        self.requests_made = 0

    def _fetch(self, url: str) -> str:
        path = None
        if self.cache_dir is not None:
            path = self.cache_dir / (hashlib.sha256(url.encode("utf-8")).hexdigest()[:32] + ".xml")
            if path.exists():
                return path.read_text(encoding="utf-8")
        wait = self.min_interval - (time.monotonic() - _last_request[0])
        if wait > 0:
            time.sleep(wait)
        try:
            self.requests_made += 1
            response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ArxivError(f"arXiv request failed: {exc}") from exc
        finally:
            _last_request[0] = time.monotonic()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(response.text, encoding="utf-8")
        return response.text

    def search(self, query: str, max_results: int = 50, sort_by: str = "relevance") -> list[dict]:
        """Search arXiv; plain text is converted with build_search_query. Duplicate ids are dropped."""
        search_query = build_search_query(query)
        records: list[dict] = []
        seen: set[str] = set()
        start = 0
        while start < max_results:
            size = min(PAGE_SIZE, max_results - start)
            page = parse_feed(self._fetch(query_url(search_query, size, start, sort_by)))
            for record in page:
                if record["arxiv_id"] not in seen:
                    seen.add(record["arxiv_id"])
                    records.append(record)
            if len(page) < size:
                break
            start += size
        return records
