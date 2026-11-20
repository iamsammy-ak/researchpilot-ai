"""End-to-end UI test: runs the real Streamlit script headlessly (no browser, no model or arXiv calls)."""

import pytest
from streamlit.testing.v1 import AppTest

from researchpilot.demo import load_samples


@pytest.fixture(autouse=True)
def _demo_mode(monkeypatch):
    """Force demo mode even on a machine with a key in its environment or global Streamlit secrets."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setattr("researchpilot.config._secret", lambda name: None)


def _app() -> AppTest:
    at = AppTest.from_file("../app.py", default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def _metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def test_app_renders_headline_metrics_and_tabs():
    at = _app()
    labels = [m.label for m in at.metric]
    for label in ("Sources", "Passages", "Years", "Topics", "Authors", "Grounded"):
        assert label in labels
    assert len(at.tabs) == 6
    assert _metric(at, "Sources") == "50"
    assert any("Demo mode" in i.value for i in at.sidebar.info)
    assert any("OPENROUTER_API_KEY" in i.value for i in at.info)  # AI sections explain the setup step


def test_every_library_choice_renders():
    at = _app()
    for name in [s["name"] for s in load_samples()] + ["Search arXiv", "Only my documents"]:
        at.selectbox(key="library_choice").set_value(name).run()
        assert not at.exception, (name, at.exception)
    assert _metric(at, "Sources") == "0"  # "Only my documents" with nothing added
    at.text_area(key="paste_text").set_value("Solid electrolyte interphase growth slows down after formation "
                                             "cycling. Lithium plating dominates at low temperature.").run()
    assert not at.exception and _metric(at, "Sources") == "1"


def test_question_search_and_settings_drive_the_computation():
    at = _app()
    at.text_input(key="query").set_value("hallucination detection models").run()
    assert not at.exception
    at.slider(key="top_k").set_value(4).run()
    at.slider(key="chunk_words").set_value(200).run()
    assert not at.exception and _metric(at, "Passages") != "88"
    at.text_area(key="question").set_value("zzz qqq").run()  # nothing matches: no evidence pack
    assert not at.exception


def test_grounding_check_and_dataset_controls():
    at = _app()
    sample = load_samples()[0]
    assert at.text_area(key="ground_text").value == sample["grounding_example"]
    assert _metric(at, "Checked") == "10"
    at.text_area(key="ground_text").set_value(
        "RGB divides the instances within the benchmark into 4 separate testbeds [S1].").run()
    assert _metric(at, "Supported") == "1"
    at.selectbox(key="tg_bundled").set_value("lab").run()
    assert any(i.value.startswith("No evidence that capacity_retention_pct differs") for i in at.info)
    at.radio(key="corr_method").set_value("spearman").run()
    at.selectbox(key="dist_bundled").set_value("chemistry").run()
    assert not at.exception
