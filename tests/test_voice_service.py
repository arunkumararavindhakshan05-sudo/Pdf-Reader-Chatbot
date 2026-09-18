from dataclasses import dataclass

import pytest

from src.voice_service import (
    DEFAULT_TRANSCRIPTION_MODEL,
    MAX_TRANSCRIPTION_PROMPT_CHARACTERS,
    TRANSCRIPTION_LANGUAGES,
    VoiceService,
    create_groq_audio_client,
)


@dataclass
class FakeTranscriptionResponse:
    """A predictable response from the fake transcription API."""

    text: str
    language: str | None = None


class FakeTranscriptionsAPI:
    """Record transcription requests and return configured text."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[
            tuple[
                tuple[str, bytes],
                str,
                str,
                str | None,
                float,
            ]
        ] = []
        self.prompts: list[str] = []
        self.language: str | None = None

    def create(
        self,
        *,
        file: tuple[str, bytes],
        model: str,
        response_format: str,
        language: str | None,
        temperature: float,
        prompt: str = "",
    ) -> FakeTranscriptionResponse:
        self.calls.append(
            (
                file,
                model,
                response_format,
                language,
                temperature,
            )
        )
        self.prompts.append(prompt)

        return FakeTranscriptionResponse(
            text=self.response_text,
            language=self.language,
        )


class FakeAudioAPI:
    """Expose the fake transcription endpoint."""

    def __init__(self, response_text: str) -> None:
        self.transcriptions = FakeTranscriptionsAPI(response_text)


class FakeGroqClient:
    """Provide the minimum client structure required by VoiceService."""

    def __init__(self, response_text: str) -> None:
        self.audio = FakeAudioAPI(response_text)


def test_transcribe_returns_cleaned_text() -> None:
    """Valid audio should produce a stripped transcript."""
    client = FakeGroqClient("  What is supervised learning?  ")
    service = VoiceService(client=client)

    transcript = service.transcribe(
        audio_bytes=b"fake-wav-audio",
        filename="question.wav",
        language=" en ",
    )

    assert transcript == "What is supervised learning?"
    assert client.audio.transcriptions.calls == [
        (
            ("question.wav", b"fake-wav-audio"),
            DEFAULT_TRANSCRIPTION_MODEL,
            "verbose_json",
            "en",
            0.0,
        )
    ]


def test_transcribe_accepts_uppercase_extension() -> None:
    """Audio extension validation should be case-insensitive."""
    service = VoiceService(
        client=FakeGroqClient("Question"),
    )

    result = service.transcribe(
        audio_bytes=b"audio",
        filename="QUESTION.WAV",
    )

    assert result == "Question"


def test_transcribe_rejects_empty_audio() -> None:
    """An empty microphone recording should fail before an API call."""
    service = VoiceService(
        client=FakeGroqClient("Question"),
    )

    with pytest.raises(ValueError, match="cannot be empty"):
        service.transcribe(
            audio_bytes=b"",
            filename="question.wav",
        )


def test_transcribe_rejects_oversized_audio() -> None:
    """Recordings larger than the configured limit should be rejected."""
    service = VoiceService(
        client=FakeGroqClient("Question"),
        maximum_audio_size=3,
    )

    with pytest.raises(ValueError, match="exceeds"):
        service.transcribe(
            audio_bytes=b"four",
            filename="question.wav",
        )


def test_transcribe_rejects_unsupported_format() -> None:
    """Only formats accepted by Groq should be submitted."""
    service = VoiceService(
        client=FakeGroqClient("Question"),
    )

    with pytest.raises(ValueError, match="Unsupported"):
        service.transcribe(
            audio_bytes=b"audio",
            filename="question.txt",
        )


@pytest.mark.parametrize(
    "model_name",
    [
        "",
        "   ",
    ],
)
def test_service_rejects_empty_model_name(
    model_name: str,
) -> None:
    """The transcription model must be configured."""
    with pytest.raises(ValueError, match="model name"):
        VoiceService(
            client=FakeGroqClient("Question"),
            model_name=model_name,
        )


@pytest.mark.parametrize(
    "maximum_audio_size",
    [
        0,
        -1,
    ],
)
def test_service_rejects_invalid_size_limit(
    maximum_audio_size: int,
) -> None:
    """The audio-size limit must be positive."""
    with pytest.raises(ValueError, match="greater than zero"):
        VoiceService(
            client=FakeGroqClient("Question"),
            maximum_audio_size=maximum_audio_size,
        )


@pytest.mark.parametrize(
    "response_text",
    [
        "",
        "   ",
    ],
)
def test_transcribe_rejects_empty_response(
    response_text: str,
) -> None:
    """Empty transcription responses should be reported clearly."""
    service = VoiceService(
        client=FakeGroqClient(response_text),
    )

    with pytest.raises(ValueError, match="empty text"):
        service.transcribe(
            audio_bytes=b"audio",
            filename="question.wav",
        )


@pytest.mark.parametrize(
    "api_key",
    [
        "",
        "   ",
    ],
)
def test_create_client_rejects_empty_api_key(
    api_key: str,
) -> None:
    """A Groq client cannot be created without an API key."""
    with pytest.raises(ValueError, match="cannot be empty"):
        create_groq_audio_client(api_key)


def test_transcribe_sets_language_and_document_context() -> None:
    """Whisper is given the document's own vocabulary, and a language if named."""
    client = FakeGroqClient("What is Arunkumar's notice period?")
    service = VoiceService(client=client)

    service.transcribe(
        audio_bytes=b"audio",
        language="en",
        context="Resume  of Arunkumar   Aravindhakshan, QA automation engineer.",
    )

    api = client.audio.transcriptions
    assert api.calls[0][3] == "en"
    assert (
        api.prompts[0] == "Resume of Arunkumar Aravindhakshan, QA automation engineer."
    )


def test_transcription_context_is_truncated() -> None:
    client = FakeGroqClient("ok")
    service = VoiceService(client=client)

    service.transcribe(audio_bytes=b"audio", context="word " * 400)

    assert (
        len(client.audio.transcriptions.prompts[0])
        == MAX_TRANSCRIPTION_PROMPT_CHARACTERS
    )


def test_transcription_model_favours_accuracy() -> None:
    assert DEFAULT_TRANSCRIPTION_MODEL == "whisper-large-v3"


def test_language_can_be_set_or_auto_detected() -> None:
    """A named language is passed through; None lets Whisper detect it."""
    client = FakeGroqClient("என்ன")
    service = VoiceService(client=client)

    service.transcribe(audio_bytes=b"audio", language="ta")
    service.transcribe(audio_bytes=b"audio", language=None)

    assert [call[3] for call in client.audio.transcriptions.calls] == ["ta", None]


def test_offered_languages_include_indian_languages() -> None:
    assert TRANSCRIPTION_LANGUAGES["Auto-detect"] is None
    assert TRANSCRIPTION_LANGUAGES["Tamil"] == "ta"
    assert {"English", "Hindi", "Malayalam", "Telugu", "Kannada"} <= set(
        TRANSCRIPTION_LANGUAGES
    )


def test_detected_language_is_recorded_from_the_response() -> None:
    """Whisper reports the language it heard, so the app needs no setting."""
    client = FakeGroqClient("இது யாருடைய resume?")
    client.audio.transcriptions.language = "tamil"
    service = VoiceService(client=client)

    service.transcribe(audio_bytes=b"audio")

    assert service.last_detected_language == "ta"
    assert client.audio.transcriptions.calls[0][2] == "verbose_json"


def test_language_defaults_to_auto_detection() -> None:
    client = FakeGroqClient("hello")
    service = VoiceService(client=client)

    service.transcribe(audio_bytes=b"audio")

    assert client.audio.transcriptions.calls[0][3] is None
    assert service.last_detected_language == "en"
