import pytest

from src.language import DEFAULT_LANGUAGE, detect_language, normalize_language


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Whose resume is this?", "en"),
        ("இது யாருடைய resume?", "ta"),
        ("यह किसका बायोडाटा है?", "hi"),
        ("ഇത് ആരുടെ രേഖയാണ്?", "ml"),
        ("ఇది ఎవరి పత్రం?", "te"),
        ("ಇದು ಯಾರ ದಾಖಲೆ?", "kn"),
        ("هذا مستند من؟", "ar"),
    ],
)
def test_script_identifies_the_language(text: str, expected: str) -> None:
    assert detect_language(text) == expected


def test_mixed_text_follows_the_dominant_script() -> None:
    assert detect_language("resume வேலை அனுபவம் விவரம் காட்டு") == "ta"


@pytest.mark.parametrize("text", ["", "   ", "12345", "!!! ???"])
def test_text_without_letters_defaults_to_english(text: str) -> None:
    assert detect_language(text) == DEFAULT_LANGUAGE


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("ta-IN", "ta"),
        ("en_US", "en"),
        ("Tamil", "ta"),
        ("english", "en"),
        ("  hi  ", "hi"),
        (None, "en"),
        ("", "en"),
    ],
)
def test_normalize_language(value: str | None, expected: str) -> None:
    assert normalize_language(value) == expected
