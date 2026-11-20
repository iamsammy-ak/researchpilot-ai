"""Pydantic model for the AI research report.

Free models do not always follow a schema exactly, so validators coerce common variations (a string
where a list was expected, objects instead of strings, "[S3], S5 p. 2" instead of ["S3", "S5 p.2"])
rather than rejecting the reply.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

_LABEL = re.compile(r"S\s*(\d+)(?:\s*,?\s*(?:p|pp|page)\.?\s*(\d+))?", re.I)


def _as_text(item: Any) -> str:
    if isinstance(item, dict):
        return " — ".join(str(v) for v in item.values() if v not in (None, "", []))
    return str(item).strip()


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    return [text for text in (_as_text(v) for v in value) if text]


def normalise_labels(value: Any) -> list[str]:
    """Pull citation labels out of whatever the model returned: "S3", "[S3 p. 2]", ["S3", "S5"], "S3, S5"."""
    if value is None:
        return []
    raw = " ".join(_as_text_list(value)) if isinstance(value, list) else str(value)
    labels: list[str] = []
    for number, page in _LABEL.findall(raw):
        label = f"S{int(number)}" + (f" p.{int(page)}" if page else "")
        if label not in labels:
            labels.append(label)
    return labels


class Finding(BaseModel):
    statement: str = Field(description="One finding in one or two sentences, with inline citations like [S3]")
    citations: list[str] = Field(default_factory=list, description='Labels of the supporting passages, e.g. ["S3"]')
    strength: str = Field("moderate", description="strong, moderate or limited: how well the evidence backs it")

    @field_validator("statement", mode="before")
    @classmethod
    def _statement(cls, value: Any) -> str:
        return _as_text(value)

    @field_validator("citations", mode="before")
    @classmethod
    def _citations(cls, value: Any) -> list[str]:
        return normalise_labels(value)

    @field_validator("strength", mode="before")
    @classmethod
    def _strength(cls, value: Any) -> str:
        text = str(value or "moderate").lower()
        for level in ("strong", "moderate", "limited"):
            if level in text:
                return level
        return "limited" if "weak" in text else "moderate"


class ResearchReport(BaseModel):
    question: str = Field(description="The research question, restated precisely")
    summary: str = Field("", description="Two or three sentences answering the question, with citations")
    key_findings: list[Finding] = Field(default_factory=list)
    methods_overview: list[str] = Field(default_factory=list, description="Methods used across the evidence")
    disagreements: list[str] = Field(default_factory=list, description="Where the sources disagree or conflict")
    gaps: list[str] = Field(default_factory=list, description="What the evidence does not cover")
    future_work: list[str] = Field(default_factory=list, description="Concrete next studies or experiments")

    @field_validator("question", "summary", mode="before")
    @classmethod
    def _text(cls, value: Any) -> str:
        return _as_text(value) if value is not None else ""

    @field_validator("key_findings", mode="before")
    @classmethod
    def _findings(cls, value: Any) -> list:
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]
        return [{"statement": v} if isinstance(v, str) else v for v in value]

    @field_validator("methods_overview", "disagreements", "gaps", "future_work", mode="before")
    @classmethod
    def _lists(cls, value: Any) -> list[str]:
        return _as_text_list(value)
