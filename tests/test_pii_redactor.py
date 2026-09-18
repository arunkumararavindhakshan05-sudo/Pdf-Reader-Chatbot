import pytest

from src.pii_redactor import (
    RedactionSession,
    aadhaar_valid,
    luhn_valid,
    verhoeff_valid,
)


def make_aadhaar(prefix: str = "23456789012") -> str:
    """Append a valid Verhoeff check digit to an 11-digit prefix."""
    for check_digit in "0123456789":
        candidate = prefix + check_digit
        if verhoeff_valid(candidate):
            return candidate
    raise AssertionError("no check digit found")


VALID_AADHAAR = make_aadhaar()


def test_email_phone_pan_ifsc_are_masked() -> None:
    session = RedactionSession()
    text = (
        "Contact arun@example.com or +91 98765 43210. PAN ABCPE1234F, IFSC HDFC0001234."
    )

    redacted = session.redact(text)

    assert "arun@example.com" not in redacted
    assert "98765" not in redacted
    assert "ABCPE1234F" not in redacted
    assert "HDFC0001234" not in redacted
    assert "<EMAIL_1>" in redacted
    assert "<PHONE_1>" in redacted
    assert "<PAN_1>" in redacted
    assert "<IFSC_1>" in redacted
    assert session.total_masked == 4


def test_valid_aadhaar_masked_invalid_left_alone() -> None:
    session = RedactionSession()
    spaced = f"{VALID_AADHAAR[:4]} {VALID_AADHAAR[4:8]} {VALID_AADHAAR[8:]}"
    invalid = VALID_AADHAAR[:-1] + str((int(VALID_AADHAAR[-1]) + 1) % 10)

    redacted = session.redact(f"Aadhaar {spaced}; invoice {invalid}")

    assert "<AADHAAR_1>" in redacted
    assert invalid in redacted


def test_card_number_requires_luhn() -> None:
    session = RedactionSession()

    redacted = session.redact("Card 4111 1111 1111 1111, order 4111 1111 1111 1112")

    assert "<CARD_NUMBER_1>" in redacted
    assert "4111 1111 1111 1112" in redacted


def test_same_value_reuses_placeholder_and_restores() -> None:
    session = RedactionSession()

    first = session.redact("Mail a@b.io then c@d.io")
    second = session.redact("Reply to a@b.io")

    assert first == "Mail <EMAIL_1> then <EMAIL_2>"
    assert second == "Reply to <EMAIL_1>"
    assert session.restore("Sent to <EMAIL_2> and <EMAIL_1>; <EMAIL_9> unknown") == (
        "Sent to c@d.io and a@b.io; <EMAIL_9> unknown"
    )
    assert session.counts == {"EMAIL": 2}


def test_enabled_entities_limits_masking() -> None:
    session = RedactionSession(enabled_entities=frozenset({"EMAIL"}))

    redacted = session.redact("a@b.io 9876543210")

    assert redacted == "<EMAIL_1> 9876543210"


def test_plain_numbers_and_years_are_not_masked() -> None:
    session = RedactionSession()
    text = "Revenue 12000 in 2026, invoice 12345, version 1.2.3"

    assert session.redact(text) == text
    assert session.total_masked == 0


def test_ip_address_masked() -> None:
    assert RedactionSession().redact("server 192.168.1.20") == "server <IP_ADDRESS_1>"


@pytest.mark.parametrize(
    ("value", "expected"),
    [("4111111111111111", True), ("4111111111111112", False), ("12", False)],
)
def test_luhn(value: str, expected: bool) -> None:
    assert luhn_valid(value) is expected


def test_aadhaar_rejects_wrong_length_and_leading_digit() -> None:
    assert aadhaar_valid(VALID_AADHAAR)
    assert not aadhaar_valid(VALID_AADHAAR[:-1])
    assert not aadhaar_valid("1" + VALID_AADHAAR[1:])


@pytest.mark.parametrize(
    "address",
    [
        "12 Gandhi Road, RS Puram, Coimbatore, Tamil Nadu 641002",
        "Flat No. 4B/12, Lakshmi Apartments, Anna Nagar, Chennai - 600 040",
        "No. 45, Avinashi Road, Peelamedu, Coimbatore 641004",
    ],
)
def test_indian_addresses_with_pin_code_are_masked(address: str) -> None:
    session = RedactionSession()

    redacted = session.redact(f"Ship to {address}. Thanks")

    assert redacted == "Ship to <ADDRESS_1>. Thanks"


@pytest.mark.parametrize(
    "text",
    [
        "The report has 12 pages and covers road safety in 2026 across 641002 users.",
        "Chapter 3 main results: revenue 400001 rupees",
    ],
)
def test_address_pattern_ignores_ordinary_sentences(text: str) -> None:
    assert RedactionSession().redact(text) == text


def test_labelled_pin_code_is_masked() -> None:
    assert RedactionSession().redact("PIN: 641002") == "<PIN_CODE_1>"


def test_registered_values_are_masked_everywhere_and_counted_when_found() -> None:
    session = RedactionSession()
    session.register("PERSON", ["Arun Kumar", "Priya Sharma", "Al", "1200", None])

    redacted = session.redact("Arun Kumar sold to Arun Kumar Traders; Al paid 1200")

    assert redacted == "<PERSON_1> sold to <PERSON_1> Traders; Al paid 1200"
    assert session.counts == {"PERSON": 1}


def test_value_found_once_is_masked_in_later_text() -> None:
    calls = iter([[(0, 5, "PERSON")], []])
    session = RedactionSession(entity_detector=lambda _: next(calls))

    assert session.redact("Priya approved") == "<PERSON_1> approved"
    assert session.redact("Ask Priya again") == "Ask <PERSON_1> again"


def test_enabled_entities_also_limits_registered_values() -> None:
    session = RedactionSession(enabled_entities=frozenset({"EMAIL"}))
    session.register("PERSON", ["Arun Kumar"])

    assert session.redact("Arun Kumar") == "Arun Kumar"


def test_detector_value_is_masked_at_every_position_in_same_text() -> None:
    text = "South | a\nSouth | b"
    second = text.index("South", 1)
    session = RedactionSession(
        entity_detector=lambda _: [(second, second + 5, "LOCATION")]
    )

    assert session.redact(text) == "<LOCATION_1> | a\n<LOCATION_1> | b"
