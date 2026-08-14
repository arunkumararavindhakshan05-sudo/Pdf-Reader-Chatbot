from pathlib import Path
from typing import Protocol

DEFAULT_TRANSCRIPTION_MODEL = "whisper-large-v3-turbo"
MAX_AUDIO_SIZE_BYTES = 25 * 1024 * 1024

SUPPORTED_AUDIO_EXTENSIONS = frozenset(
    {
        ".flac",
        ".m4a",
        ".mp3",
        ".mp4",
        ".mpeg",
        ".mpga",
        ".ogg",
        ".wav",
        ".webm",
    }
)


class TranscriptionResponse(Protocol):
    """Minimum response returned by the transcription API."""

    text: str


class TranscriptionsAPI(Protocol):
    """Interface required from Groq's transcription endpoint."""

    def create(
        self,
        *,
        file: tuple[str, bytes],
        model: str,
        response_format: str,
        language: str | None,
        temperature: float,
    ) -> TranscriptionResponse:
        """Transcribe an uploaded audio recording."""
        ...


class AudioAPI(Protocol):
    """Audio operations exposed by a Groq client."""

    transcriptions: TranscriptionsAPI


class GroqAudioClient(Protocol):
    """Minimum Groq client interface required by this service."""

    audio: AudioAPI


def create_groq_audio_client(api_key: str) -> GroqAudioClient:
    """Create a Groq client using a securely supplied API key."""
    cleaned_api_key = api_key.strip()

    if not cleaned_api_key:
        raise ValueError("The Groq API key cannot be empty.")

    from groq import Groq

    return Groq(api_key=cleaned_api_key)


class VoiceService:
    """Validate and transcribe microphone audio using Groq Whisper."""

    def __init__(
        self,
        client: GroqAudioClient,
        model_name: str = DEFAULT_TRANSCRIPTION_MODEL,
        maximum_audio_size: int = MAX_AUDIO_SIZE_BYTES,
    ) -> None:
        if not model_name.strip():
            raise ValueError("The transcription model name cannot be empty.")

        if maximum_audio_size <= 0:
            raise ValueError("The maximum audio size must be greater than zero.")

        self._client = client
        self._model_name = model_name.strip()
        self._maximum_audio_size = maximum_audio_size

    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "question.wav",
        language: str | None = None,
    ) -> str:
        """Convert an audio recording into a cleaned text question."""
        if not audio_bytes:
            raise ValueError("The audio recording cannot be empty.")

        if len(audio_bytes) > self._maximum_audio_size:
            maximum_megabytes = self._maximum_audio_size / (1024 * 1024)
            raise ValueError(
                f"The audio recording exceeds the {maximum_megabytes:.0f} MB limit."
            )

        extension = Path(filename).suffix.lower()

        if extension not in SUPPORTED_AUDIO_EXTENSIONS:
            supported_formats = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
            raise ValueError(
                f"Unsupported audio format. Use one of: {supported_formats}."
            )

        cleaned_language = language.strip() if language else None

        response = self._client.audio.transcriptions.create(
            file=(filename, audio_bytes),
            model=self._model_name,
            response_format="json",
            language=cleaned_language,
            temperature=0.0,
        )

        transcript = response.text

        if not isinstance(transcript, str) or not transcript.strip():
            raise ValueError("The transcription service returned empty text.")

        return transcript.strip()
