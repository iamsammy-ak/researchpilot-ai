"""arXiv client and document ingestion. No network: requests.get is replaced and the clock is faked."""

from pathlib import Path

import pytest
import requests

from researchpilot import arxiv
from researchpilot.arxiv import ArxivClient, ArxivError, build_search_query, parse_feed
from researchpilot.corpus import build_library, from_text, read_pdf

FIXTURE = (Path(__file__).parent / "fixtures" / "arxiv_sample.xml").read_text(encoding="utf-8")


def make_pdf(pages: list[str]) -> bytes:
    """A minimal, valid PDF with one line of Helvetica text per page."""
    n = len(pages)
    objects = {
        1: "<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{' '.join(f'{4 + 2 * i} 0 R' for i in range(n))}] /Count {n} >>",
        3: "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for i, text in enumerate(pages):
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
        objects[4 + 2 * i] = (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                              f"/Resources << /Font << /F1 3 0 R >> >> /Contents {5 + 2 * i} 0 R >>")
        objects[5 + 2 * i] = f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"
    out = b"%PDF-1.4\n"
    offsets = []
    for number in sorted(objects):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{objects[number]}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


def test_build_search_query():
    assert build_search_query('"state of health" battery estimation') == (
        'all:"state of health" AND all:battery AND all:estimation'
    )
    assert build_search_query("ti:transformer AND cat:cs.CL") == "ti:transformer AND cat:cs.CL"
    with pytest.raises(ValueError):
        build_search_query("the of and")


def test_parse_feed_reads_bundled_atom_snapshot():
    records = parse_feed(FIXTURE)
    assert len(records) == 3
    first = records[0]
    assert first["arxiv_id"] == "1903.07672"  # version suffix removed
    assert first["title"].startswith("State of health estimation for lithium-ion battery")
    assert first["authors"] and first["published"] == "2019-03-15"
    assert first["categories"][0] == first["categories"][0].strip() and "." in first["categories"][0]
    assert first["pdf_url"].startswith("https://arxiv.org/pdf/")
    assert "  " not in first["abstract"]


class _Response:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        pass


def test_client_caches_responses_and_spaces_requests(tmp_path, monkeypatch):
    calls, sleeps, clock = [], [], [1000.0]
    monkeypatch.setattr(arxiv, "_last_request", [0.0])
    monkeypatch.setattr(arxiv.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(arxiv.time, "sleep", lambda s: sleeps.append(s))

    def fake_get(url, headers, timeout):
        calls.append(url)
        assert "ResearchPilot" in headers["User-Agent"]
        return _Response(FIXTURE)

    monkeypatch.setattr(arxiv.requests, "get", fake_get)
    client = ArxivClient(cache_dir=tmp_path)
    assert len(client.search("battery health", max_results=3)) == 3
    clock[0] += 1.0  # one second later: a new query must wait about two more seconds
    client.search("battery capacity", max_results=3)
    assert sleeps and sleeps[0] == pytest.approx(2.0)
    client.search("battery health", max_results=3)  # repeated query: served from the disk cache
    assert len(calls) == 2 and client.requests_made == 2

    def offline(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(arxiv.requests, "get", offline)
    with pytest.raises(ArxivError):
        client.search("solid state electrolyte")  # not cached: the network error surfaces as ArxivError


def test_pdf_pages_and_library_labels():
    pdf = read_pdf(make_pdf(["Capacity fade on page one", "Lithium plating on page two"]), "my_notes.pdf")
    assert pdf.kind == "pdf" and pdf.title == "my notes"
    assert [page for page, _ in pdf.page_texts()] == [1, 2]
    assert "plating" in pdf.pages[1]
    with pytest.raises(ValueError):
        read_pdf(make_pdf([""]), "scan.pdf")
    library = build_library(parse_feed(FIXTURE), [pdf, from_text("Pasted notes\nabout cells")])
    assert [s.sid for s in library] == ["S1", "S2", "S3", "S4", "S5"]
    assert library[3].kind == "pdf" and library[4].title == "Pasted notes"
    with pytest.raises(ValueError):
        from_text("   ")
