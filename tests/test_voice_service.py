from dataclasses import dataclass

import pytest

from src.voice_service import (
    DEFAULT_TRANSCRIPTION_MODEL,
    VoiceService,
    create_groq_audio_client,
)


@dataclass
class FakeTranscriptionResponse:
    """A predictable response from the fake transcription API."""

    text: str


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

    def create(
        self,
        *,
        file: tuple[str, bytes],
        model: str,
        response_format: str,
        language: str | None,
        temperature: float,
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

        return FakeTranscriptionResponse(text=self.response_text)


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
            "json",
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
