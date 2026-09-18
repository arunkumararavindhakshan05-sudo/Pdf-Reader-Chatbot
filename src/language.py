"""Work out what language a question is in, without asking the user.

Two sources are used. A spoken question carries the language Whisper detected
while transcribing it, which is reliable because Whisper hears the sound. A
typed question is identified from its script: Tamil, Devanagari, Malayalam and
the other Indic blocks each have their own Unicode range, so a single pass over
the characters is enough, with no extra dependency and no language picker in
the interface.

Latin script is ambiguous between many languages, so it is reported as English,
which is the right default for the voices used to read answers aloud.
"""

import unicodedata

# Unicode block prefixes, in the order unicodedata names them, mapped to the
# ISO language code used for speech.
SCRIPT_LANGUAGES: tuple[tuple[str, str], ...] = (
    ("TAMIL", "ta"),
    ("DEVANAGARI", "hi"),
    ("MALAYALAM", "ml"),
    ("TELUGU", "te"),
    ("KANNADA", "kn"),
    ("BENGALI", "bn"),
    ("GUJARATI", "gu"),
    ("GURMUKHI", "pa"),
    ("ORIYA", "or"),
    ("SINHALA", "si"),
    ("ARABIC", "ar"),
    ("HEBREW", "he"),
    ("THAI", "th"),
    ("HIRAGANA", "ja"),
    ("KATAKANA", "ja"),
    ("HANGUL", "ko"),
    ("CJK", "zh"),
    ("CYRILLIC", "ru"),
    ("GREEK", "el"),
)

DEFAULT_LANGUAGE = "en"


def detect_language(text: str) -> str:
    """Return an ISO language code for a piece of text, defaulting to English."""
    counts: dict[str, int] = {}

    for character in text:
        if not character.isalpha():
            continue

        try:
            name = unicodedata.name(character)
        except ValueError:
            continue

        for prefix, language in SCRIPT_LANGUAGES:
            if name.startswith(prefix):
                counts[language] = counts.get(language, 0) + 1
                break
        else:
            counts[DEFAULT_LANGUAGE] = counts.get(DEFAULT_LANGUAGE, 0) + 1

    if not counts:
        return DEFAULT_LANGUAGE

    return max(counts, key=lambda language: counts[language])


def normalize_language(code: str | None) -> str:
    """Reduce a code such as "ta-IN" or "English" to a short ISO code."""
    if not code:
        return DEFAULT_LANGUAGE

    cleaned = code.strip().lower().replace("_", "-")

    # Whisper sometimes reports a language name rather than a code.
    names = {
        "english": "en",
        "tamil": "ta",
        "hindi": "hi",
        "malayalam": "ml",
        "telugu": "te",
        "kannada": "kn",
        "marathi": "mr",
        "bengali": "bn",
        "gujarati": "gu",
        "punjabi": "pa",
        "urdu": "ur",
        "arabic": "ar",
        "french": "fr",
        "german": "de",
        "spanish": "es",
    }

    if cleaned in names:
        return names[cleaned]

    return cleaned.split("-")[0] or DEFAULT_LANGUAGE
