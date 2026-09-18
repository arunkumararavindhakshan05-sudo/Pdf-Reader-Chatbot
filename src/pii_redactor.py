"""Mask personal data before any text is sent to the language model.

Detection runs locally with regular expressions and checksum validation, so
no personal data leaves the container to be detected. Each value is replaced
by a stable placeholder such as ``<EMAIL_1>``; the same value always maps to
the same placeholder inside one ``RedactionSession``, which keeps answers
coherent ("<EMAIL_1> approved the invoice"). After the model answers,
``restore`` puts the original values back for the person who owns the file.

People's names and places cannot be found reliably with patterns. Pass an
``entity_detector`` (see ``src.ner.load_presidio_detector``) to add Microsoft
Presidio on top of the pattern recognizers.
"""

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

EntityDetector = Callable[[str], list[tuple[int, int, str]]]

PLACEHOLDER_PATTERN = re.compile(r"<([A-Z_]+)_(\d+)>")


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value)


def luhn_valid(value: str) -> bool:
    """Return True when a card number passes the Luhn checksum."""
    digits = _digits(value)

    if not 13 <= len(digits) <= 19:
        return False

    total = 0
    for position, character in enumerate(reversed(digits)):
        number = int(character)
        if position % 2 == 1:
            number *= 2
            if number > 9:
                number -= 9
        total += number

    return total % 10 == 0


_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_valid(value: str) -> bool:
    """Return True when a number passes the Verhoeff checksum (used by Aadhaar)."""
    digits = _digits(value)

    if not digits:
        return False

    check = 0
    for position, character in enumerate(reversed(digits)):
        check = _VERHOEFF_D[check][_VERHOEFF_P[position % 8][int(character)]]

    return check == 0


def aadhaar_valid(value: str) -> bool:
    digits = _digits(value)
    return len(digits) == 12 and digits[0] not in "01" and verhoeff_valid(digits)


@dataclass(frozen=True)
class Recognizer:
    """A named pattern with an optional validator that removes false positives."""

    entity: str
    pattern: re.Pattern[str]
    validator: Callable[[str], bool] | None = None


_STREET_WORDS = (
    r"road|rd|street|st|lane|ln|avenue|ave|main|cross|nagar|colony|layout|"
    r"puram|salai|marg|sector|block|phase|apartments?|towers?|residency|"
    r"enclave|garden|gardens|park|extension|extn|village|taluk|district|post"
)

# Order matters: longer, more specific values are matched first.
DEFAULT_RECOGNIZERS: tuple[Recognizer, ...] = (
    Recognizer(
        # A door number, a capitalised place name followed by a street word, a
        # comma, and an Indian PIN code, for example
        # "12 Gandhi Road, RS Puram, Coimbatore, Tamil Nadu 641002".
        "ADDRESS",
        re.compile(
            r"(?<![\w/])(?:(?i:no\.?|door\s*no\.?|flat\s*no\.?)\s*|#\s*)?"
            r"\d{1,5}[A-Za-z]?(?:[/-]\d{1,4}[A-Za-z]?)?,?\s+"
            r"(?:[A-Z][\w.'-]*,?\s+){1,4}(?i:" + _STREET_WORDS + r")\b"
            r"[^\n]{0,100}?,[^\n]{0,60}?\b[1-9]\d{2}\s?\d{3}\b"
        ),
    ),
    Recognizer(
        "EMAIL",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ),
    Recognizer(
        "CARD_NUMBER",
        re.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
        luhn_valid,
    ),
    Recognizer(
        "AADHAAR",
        re.compile(r"\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b"),
        aadhaar_valid,
    ),
    Recognizer(
        "PAN",
        re.compile(r"\b[A-Z]{3}[ABCFGHLJPT][A-Z]\d{4}[A-Z]\b"),
    ),
    Recognizer(
        "IFSC",
        re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
    ),
    Recognizer(
        "PHONE",
        re.compile(
            r"(?<![\w+])(?:\+91[ -]?|0091[ -]?|0)?[6-9]\d{4}[ -]?\d{5}\b"
            r"|(?<![\w])\+\d{1,3}[ -]?\(?\d{2,4}\)?[ -]?\d{3,4}[ -]?\d{3,4}\b"
        ),
    ),
    Recognizer(
        "PIN_CODE",
        re.compile(
            r"\b(?:pin\s*(?:code)?|pincode|postal\s*code)\s*[:#-]?\s*[1-9]\d{2}\s?\d{3}\b",
            re.IGNORECASE,
        ),
    ),
    Recognizer(
        "IP_ADDRESS",
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
        ),
    ),
)


@dataclass
class RedactionSession:
    """Mask text consistently and restore it after the model answers."""

    recognizers: tuple[Recognizer, ...] = DEFAULT_RECOGNIZERS
    enabled_entities: frozenset[str] | None = None
    entity_detector: EntityDetector | None = None
    _value_to_placeholder: dict[str, str] = field(default_factory=dict)
    _placeholder_to_value: dict[str, str] = field(default_factory=dict)
    _entity_counts: Counter[str] = field(default_factory=Counter)
    _registered: dict[str, str] = field(default_factory=dict)

    def register(self, entity: str, values: list[object]) -> None:
        """Mark known sensitive values, such as every cell of a "Customer" column.

        Registered values are masked wherever they appear, but only count as
        masked once they are actually found in text.
        """
        for value in values:
            text = str(value).strip() if value is not None else ""

            if len(text) >= 3 and not text.replace(".", "").isdigit():
                self._registered.setdefault(text, entity)

    def _placeholder_for(self, entity: str, value: str) -> str:
        key = f"{entity}:{value}"
        existing = self._value_to_placeholder.get(key)

        if existing:
            return existing

        self._entity_counts[entity] += 1
        placeholder = f"<{entity}_{self._entity_counts[entity]}>"
        self._value_to_placeholder[key] = placeholder
        self._placeholder_to_value[placeholder] = value
        return placeholder

    def redact(self, text: str) -> str:
        """Replace every recognised personal value in ``text`` with a placeholder."""
        spans: list[tuple[int, int, str]] = []

        for recognizer in self.recognizers:
            if (
                self.enabled_entities is not None
                and recognizer.entity not in self.enabled_entities
            ):
                continue

            for match in recognizer.pattern.finditer(text):
                start, end = match.span()

                if any(
                    start < taken_end and end > taken_start
                    for taken_start, taken_end, _ in spans
                ):
                    continue

                value = match.group(0)

                if recognizer.validator is not None and not recognizer.validator(value):
                    continue

                spans.append((start, end, recognizer.entity))

        spans.extend(self._known_value_spans(text, spans))

        if self.entity_detector is not None:
            for start, end, entity in self.entity_detector(text):
                if (
                    self.enabled_entities is not None
                    and entity not in self.enabled_entities
                ):
                    continue

                if any(
                    start < taken_end and end > taken_start
                    for taken_start, taken_end, _ in spans
                ):
                    continue

                spans.append((start, end, entity))

                self._registered.setdefault(text[start:end], entity)

            # Mask other copies of names the detector found only once in this text.
            spans.extend(self._known_value_spans(text, spans))

        if not spans:
            return text

        pieces: list[str] = []
        cursor = 0

        for start, end, entity in sorted(spans):
            pieces.append(text[cursor:start])
            pieces.append(self._placeholder_for(entity, text[start:end]))
            cursor = end

        pieces.append(text[cursor:])
        return "".join(pieces)

    def _known_value_spans(
        self,
        text: str,
        taken: list[tuple[int, int, str]],
    ) -> list[tuple[int, int, str]]:
        """Find values this session already masked, so they are masked everywhere.

        A language model can recognise "Priya Sharma" in one row of a table and
        miss it in the next. Re-using every value already found keeps masking
        consistent across the question, the evidence and the results.
        """
        spans: list[tuple[int, int, str]] = []
        occupied = list(taken)

        known = sorted(
            [
                *(key.split(":", 1) for key in self._value_to_placeholder),
                *((entity, value) for value, entity in self._registered.items()),
            ],
            key=lambda pair: len(pair[1]),
            reverse=True,
        )

        if not known:
            return spans

        for entity, value in known:
            if len(value) < 3 or (
                self.enabled_entities is not None
                and entity not in self.enabled_entities
            ):
                continue

            pattern = re.compile(r"(?<!\w)" + re.escape(value) + r"(?!\w)")

            for match in pattern.finditer(text):
                start, end = match.span()

                if any(
                    start < taken_end and end > taken_start
                    for taken_start, taken_end, _ in occupied
                ):
                    continue

                spans.append((start, end, entity))
                occupied.append((start, end, entity))

        return spans

    def restore(self, text: str) -> str:
        """Put original values back in place of known placeholders."""
        return PLACEHOLDER_PATTERN.sub(
            lambda match: self._placeholder_to_value.get(
                match.group(0), match.group(0)
            ),
            text,
        )

    @property
    def counts(self) -> dict[str, int]:
        """Number of distinct values masked, per entity type."""
        return dict(self._entity_counts)

    @property
    def total_masked(self) -> int:
        return sum(self._entity_counts.values())
