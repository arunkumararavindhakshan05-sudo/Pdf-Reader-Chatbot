"""Optional name and location detection with Microsoft Presidio.

Presidio uses a spaCy language model to find people's names and places, which
regular expressions cannot do. It is optional: when ``presidio-analyzer`` or the
spaCy model is not installed, ``load_presidio_detector`` returns ``None`` and
the app keeps working with pattern-based masking only.
"""

import logging
import os
from collections.abc import Callable
from functools import lru_cache

LOGGER = logging.getLogger(__name__)

DEFAULT_SPACY_MODEL = os.getenv("PII_SPACY_MODEL", "en_core_web_sm")
DEFAULT_NER_ENTITIES = ("PERSON", "LOCATION")
DEFAULT_MIN_SCORE = 0.6

# A detector returns (start, end, entity) spans for a piece of text.
EntityDetector = Callable[[str], list[tuple[int, int, str]]]


def build_presidio_detector(
    analyzer: object,
    entities: tuple[str, ...] = DEFAULT_NER_ENTITIES,
    min_score: float = DEFAULT_MIN_SCORE,
) -> EntityDetector:
    """Wrap a Presidio ``AnalyzerEngine`` as a simple span detector."""

    def detect(text: str) -> list[tuple[int, int, str]]:
        if not text.strip():
            return []

        spans: list[tuple[int, int, str]] = []

        for result in analyzer.analyze(
            text=text,
            language="en",
            entities=list(entities),
        ):
            value = text[result.start : result.end]

            if result.score < min_score or len(value.strip()) < 3:
                continue

            # Never re-mask placeholders produced by the pattern recognizers.
            if value.startswith("<") and value.endswith(">"):
                continue

            spans.append((result.start, result.end, result.entity_type))

        return spans

    return detect


@lru_cache(maxsize=1)
def load_presidio_detector(
    model_name: str = DEFAULT_SPACY_MODEL,
) -> EntityDetector | None:
    """Load Presidio once; return ``None`` when it is not available."""
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": model_name}],
            }
        )
        analyzer = AnalyzerEngine(
            nlp_engine=provider.create_engine(),
            supported_languages=["en"],
        )
    except (Exception, SystemExit):
        LOGGER.warning(
            "Presidio name detection is unavailable; using pattern masking only.",
            exc_info=True,
        )
        return None

    return build_presidio_detector(analyzer)
