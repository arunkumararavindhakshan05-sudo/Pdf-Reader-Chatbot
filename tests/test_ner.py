from dataclasses import dataclass

import pytest

from src.ner import build_presidio_detector, load_presidio_detector
from src.pii_redactor import RedactionSession


@dataclass
class FakeResult:
    entity_type: str
    start: int
    end: int
    score: float


class FakeAnalyzer:
    def __init__(self, results: list[FakeResult]) -> None:
        self.results = results
        self.calls: list[dict] = []

    def analyze(self, **kwargs: object) -> list[FakeResult]:
        self.calls.append(kwargs)
        return self.results


def test_detector_filters_low_scores_short_spans_and_placeholders() -> None:
    text = "Priya met Al at <EMAIL_1> in Chennai"
    analyzer = FakeAnalyzer(
        [
            FakeResult("PERSON", 0, 5, 0.85),
            FakeResult("PERSON", 10, 12, 0.85),
            FakeResult("PERSON", 16, 25, 0.85),
            FakeResult("LOCATION", 29, 36, 0.4),
        ]
    )

    spans = build_presidio_detector(analyzer)(text)

    assert spans == [(0, 5, "PERSON")]
    assert analyzer.calls[0]["entities"] == ["PERSON", "LOCATION"]


def test_detector_skips_blank_text() -> None:
    analyzer = FakeAnalyzer([])

    assert build_presidio_detector(analyzer)("   ") == []
    assert analyzer.calls == []


def test_session_uses_detector_after_patterns() -> None:
    text = "Priya Sharma wrote to priya@example.com"
    session = RedactionSession(
        entity_detector=lambda _: [(0, 12, "PERSON"), (22, 39, "PERSON")]
    )

    assert session.redact(text) == "<PERSON_1> wrote to <EMAIL_1>"


def test_load_presidio_detector_returns_none_when_model_missing() -> None:
    load_presidio_detector.cache_clear()
    try:
        assert load_presidio_detector("model_that_does_not_exist") is None
    finally:
        load_presidio_detector.cache_clear()


def test_real_presidio_finds_names_when_installed() -> None:
    pytest.importorskip("presidio_analyzer")
    pytest.importorskip("en_core_web_sm")
    detector = load_presidio_detector()
    assert detector is not None

    session = RedactionSession(entity_detector=detector)
    redacted = session.redact("Priya Sharma approved the invoice.")

    assert "Priya Sharma" not in redacted
    assert "<PERSON_1>" in redacted
