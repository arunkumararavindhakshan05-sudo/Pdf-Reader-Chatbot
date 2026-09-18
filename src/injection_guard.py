"""Detect prompt-injection attempts hidden inside uploaded documents.

Documents are untrusted input. A file can contain text such as "ignore all
previous instructions" or invisible characters meant to smuggle instructions
to the language model. This module does two things:

1. ``strip_invisible_characters`` removes zero-width and bidirectional
   control characters before text is indexed.
2. ``scan_text`` / ``scan_chunks`` flag suspicious phrases so the app can
   warn the user. Flagged text is still answered from, because the grounded
   prompt already tells the model to treat document text as data, but the
   user can see that the file tried to steer the AI.
"""

import re
from dataclasses import dataclass
from typing import Literal

Severity = Literal["high", "medium"]

INVISIBLE_CHARACTERS = re.compile(
    r"[\u200b\u200c\u200d\u2060\ufeff\u00ad\u202a-\u202e\u2066-\u2069]"
)

_RULES: list[tuple[str, Severity, re.Pattern[str]]] = [
    (
        "override_instructions",
        "high",
        re.compile(
            r"\b(ignore|disregard|forget|override|bypass)\b[\w\s,]{0,30}?"
            r"\b(previous|prior|above|earlier|all|any|system|your)\b[\w\s]{0,20}?"
            r"\b(instructions?|prompts?|rules?|directions?|guidelines?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "role_hijack",
        "high",
        re.compile(
            r"\b(you are now|act as|pretend to be|from now on,? you)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "system_prompt_probe",
        "high",
        re.compile(
            r"\b(reveal|print|show|repeat|output|leak)\b[\w\s]{0,20}?"
            r"\b(system prompt|hidden instructions|initial prompt|api key|secret)s?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "fake_chat_markup",
        "high",
        re.compile(
            r"(<\|?(system|im_start|im_end|assistant)\|?>|\[/?INST\]|###\s*(system|instruction))",
            re.IGNORECASE,
        ),
    ),
    (
        "tag_escape",
        "medium",
        re.compile(
            r"</\s*(pdf_evidence|document_evidence|question)\s*>", re.IGNORECASE
        ),
    ),
    (
        "exfiltration_link",
        "medium",
        re.compile(
            r"!\[[^\]]*\]\(https?://[^)]+[?&][^)]*=|"
            r"\b(send|post|upload|forward)\b[\w\s]{0,20}?\b(to|at)\b\s+https?://",
            re.IGNORECASE,
        ),
    ),
]


@dataclass(frozen=True)
class InjectionFinding:
    """One suspicious phrase found in a document."""

    rule: str
    severity: Severity
    excerpt: str
    location: str


def strip_invisible_characters(text: str) -> str:
    """Remove zero-width and bidirectional control characters."""
    return INVISIBLE_CHARACTERS.sub("", text)


def scan_text(text: str, location: str = "") -> list[InjectionFinding]:
    """Return every suspicious phrase in ``text``."""
    findings: list[InjectionFinding] = []

    if INVISIBLE_CHARACTERS.search(text):
        findings.append(
            InjectionFinding(
                rule="invisible_characters",
                severity="medium",
                excerpt="(zero-width or direction-control characters)",
                location=location,
            )
        )

    visible = strip_invisible_characters(text)

    for rule, severity, pattern in _RULES:
        for match in pattern.finditer(visible):
            start = max(match.start() - 30, 0)
            end = min(match.end() + 30, len(visible))
            excerpt = " ".join(visible[start:end].split())
            findings.append(
                InjectionFinding(
                    rule=rule,
                    severity=severity,
                    excerpt=excerpt,
                    location=location,
                )
            )

    return findings


def scan_chunks(chunks: list[dict]) -> list[InjectionFinding]:
    """Scan document chunks, labelling each finding with its source location."""
    findings: list[InjectionFinding] = []

    for chunk in chunks:
        location = chunk.get("location") or f"Page {chunk.get('page', '?')}"
        findings.extend(scan_text(chunk["text"], location))

    return findings


def sanitize_chunks(chunks: list[dict]) -> list[dict]:
    """Return chunks with invisible characters removed and empty chunks dropped."""
    sanitized = []

    for chunk in chunks:
        text = strip_invisible_characters(chunk["text"])

        if text.strip():
            sanitized.append({**chunk, "text": text})

    return sanitized
