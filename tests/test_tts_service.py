import wave
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.voice_service import (
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE,
    VoiceService,
    combine_wav_segments,
    split_text_for_speech,
)


def make_wav(
    *,
    frame_count: int = 20,
    frame_rate: int = 16_000,
    sample_value: int = 0,
) -> bytes:
    """Create a small valid WAV recording for isolated unit tests."""
    output = BytesIO()
    sample = sample_value.to_bytes(2, byteorder="little", signed=True)

    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(frame_rate)
        writer.writeframes(sample * frame_count)

    return output.getvalue()


class FakeSpeechResponse:
    """Fake Groq response that writes prepared WAV bytes."""

    def __init__(self, audio_bytes: bytes) -> None:
        self.audio_bytes = audio_bytes

    def write_to_file(self, file_path: str | Path) -> None:
        Path(file_path).write_bytes(self.audio_bytes)


class FakeSpeechAPI:
    """Record TTS requests without contacting Groq."""

    def __init__(self, audio_responses: list[bytes]) -> None:
        self.audio_responses = audio_responses
        self.requests: list[dict[str, str]] = []

    def create(
        self,
        *,
        model: str,
        voice: str,
        input: str,
        response_format: str,
    ) -> FakeSpeechResponse:
        self.requests.append(
            {
                "model": model,
                "voice": voice,
                "input": input,
                "response_format": response_format,
            }
        )
        response_position = len(self.requests) - 1

        return FakeSpeechResponse(self.audio_responses[response_position])


def make_client(
    audio_responses: list[bytes],
) -> tuple[SimpleNamespace, FakeSpeechAPI]:
    """Build the smallest fake Groq client needed by VoiceService."""
    speech_api = FakeSpeechAPI(audio_responses)
    audio_api = SimpleNamespace(speech=speech_api, transcriptions=None)
    client = SimpleNamespace(audio=audio_api)

    return client, speech_api


def test_split_text_for_speech_cleans_markdown_and_citations() -> None:
    chunks = split_text_for_speech(
        "**Machine learning** finds patterns. [Page 13]",
        maximum_characters=100,
    )

    assert chunks == ["Machine learning finds patterns. (Page 13)"]


def test_split_text_for_speech_respects_character_limit() -> None:
    chunks = split_text_for_speech(
        "Machine learning uses examples to discover useful patterns.",
        maximum_characters=20,
    )

    assert " ".join(chunks) == (
        "Machine learning uses examples to discover useful patterns."
    )
    assert all(1 <= len(chunk) <= 20 for chunk in chunks)


def test_split_text_for_speech_splits_an_oversized_word() -> None:
    chunks = split_text_for_speech(
        "abcdefghij",
        maximum_characters=4,
    )

    assert chunks == ["abcd", "efgh", "ij"]


@pytest.mark.parametrize(
    ("text", "maximum_characters"),
    [
        ("", 200),
        ("   ", 200),
        ("valid text", 0),
        ("valid text", -1),
    ],
)
def test_split_text_for_speech_rejects_invalid_input(
    text: str,
    maximum_characters: int,
) -> None:
    with pytest.raises(ValueError):
        split_text_for_speech(text, maximum_characters)


def test_combine_wav_segments_preserves_all_frames() -> None:
    combined_audio = combine_wav_segments(
        [
            make_wav(frame_count=10, sample_value=1),
            make_wav(frame_count=15, sample_value=2),
        ]
    )

    with wave.open(BytesIO(combined_audio), "rb") as reader:
        assert reader.getnchannels() == 1
        assert reader.getsampwidth() == 2
        assert reader.getframerate() == 16_000
        assert reader.getnframes() == 25


@pytest.mark.parametrize("segments", [[], [b""]])
def test_combine_wav_segments_rejects_missing_audio(
    segments: list[bytes],
) -> None:
    with pytest.raises(ValueError):
        combine_wav_segments(segments)


def test_combine_wav_segments_rejects_incompatible_formats() -> None:
    with pytest.raises(ValueError, match="incompatible formats"):
        combine_wav_segments(
            [
                make_wav(frame_rate=16_000),
                make_wav(frame_rate=8_000),
            ]
        )


def test_synthesize_uses_default_groq_tts_settings() -> None:
    client, speech_api = make_client([make_wav()])
    service = VoiceService(client=client)

    generated_audio = service.synthesize("A grounded PDF answer.")

    assert generated_audio
    assert speech_api.requests == [
        {
            "model": DEFAULT_TTS_MODEL,
            "voice": DEFAULT_TTS_VOICE,
            "input": "A grounded PDF answer.",
            "response_format": "wav",
        }
    ]


def test_synthesize_generates_and_combines_multiple_segments() -> None:
    client, speech_api = make_client(
        [
            make_wav(frame_count=5),
            make_wav(frame_count=7),
        ]
    )
    service = VoiceService(
        client=client,
        maximum_speech_chunk_characters=20,
    )

    generated_audio = service.synthesize("One short sentence for testing.")

    assert len(speech_api.requests) == 2
    assert all(len(request["input"]) <= 20 for request in speech_api.requests)

    with wave.open(BytesIO(generated_audio), "rb") as reader:
        assert reader.getnframes() == 12


def test_synthesize_uses_custom_model_and_voice() -> None:
    client, speech_api = make_client([make_wav()])
    service = VoiceService(
        client=client,
        tts_model_name="custom-tts-model",
        tts_voice="autumn",
    )

    service.synthesize("Read this answer.")

    assert speech_api.requests[0]["model"] == "custom-tts-model"
    assert speech_api.requests[0]["voice"] == "autumn"


def test_synthesize_rejects_empty_generated_audio() -> None:
    client, _ = make_client([b""])
    service = VoiceService(client=client)

    with pytest.raises(ValueError, match="empty audio"):
        service.synthesize("Read this answer.")


@pytest.mark.parametrize(
    "keyword_arguments",
    [
        {"tts_model_name": ""},
        {"tts_voice": ""},
        {"maximum_speech_chunk_characters": 0},
    ],
)
def test_voice_service_rejects_invalid_tts_configuration(
    keyword_arguments: dict[str, str | int],
) -> None:
    client, _ = make_client([make_wav()])

    with pytest.raises(ValueError):
        VoiceService(client=client, **keyword_arguments)
