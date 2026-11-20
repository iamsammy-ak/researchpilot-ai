"""Bundled samples (arXiv corpora, a synthetic dataset) and saved AI analyses, so the app works without a key."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"
CORPORA_DIR = DEMO_DIR / "corpora"
ANALYSES_DIR = DEMO_DIR / "analyses"
DATASET_FILE = DEMO_DIR / "datasets" / "battery_ageing_synthetic.csv"
DATASET_NAME = "Battery ageing tests (synthetic)"


def load_samples() -> list[dict]:
    return json.loads((DEMO_DIR / "samples.json").read_text(encoding="utf-8"))


def load_corpus(slug: str) -> dict:
    return json.loads((CORPORA_DIR / f"{slug}.json").read_text(encoding="utf-8"))


def load_bundled_dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET_FILE)


def load_saved_analysis(fingerprint: str) -> dict | None:
    path = ANALYSES_DIR / f"{fingerprint}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_analysis(analysis: dict) -> Path:
    ANALYSES_DIR.mkdir(parents=True, exist_ok=True)
    path = ANALYSES_DIR / f"{analysis['fingerprint']}.json"
    path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
