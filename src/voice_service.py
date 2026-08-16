import wave
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

DEFAULT_TRANSCRIPTION_MODEL = "whisper-large-v3-turbo"
DEFAULT_TTS_MODEL = "canopylabs/orpheus-v1-english"
DEFAULT_TTS_VOICE = "hannah"
MAX_AUDIO_SIZE_BYTES = 25 * 1024 * 1024
MAX_TTS_CHUNK_CHARACTERS = 200

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


class SpeechResponse(Protocol):
    """Binary speech response returned by Groq."""

    def write_to_file(self, file_path: str | Path) -> None:
        """Write generated speech to a WAV file."""
        ...


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


class SpeechAPI(Protocol):
    """Interface required from Groq's speech endpoint."""

    def create(
        self,
        *,
        model: str,
        voice: str,
        input: str,
        response_format: str,
    ) -> SpeechResponse:
        """Convert one text segment into WAV audio."""
        ...


class AudioAPI(Protocol):
    """Audio operations exposed by a Groq client."""

    transcriptions: TranscriptionsAPI
    speech: SpeechAPI


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


def split_text_for_speech(
    text: str,
    maximum_characters: int = MAX_TTS_CHUNK_CHARACTERS,
) -> list[str]:
    """Split text into word-safe segments accepted by Groq TTS."""
    if maximum_characters <= 0:
        raise ValueError("The speech chunk length must be greater than zero.")

    cleaned_text = " ".join(
        text.replace("[", "(").replace("]", ")").replace("*", "").split()
    )

    if not cleaned_text:
        raise ValueError("The speech text cannot be empty.")

    chunks: list[str] = []
    current_chunk = ""

    for word in cleaned_text.split():
        if len(word) > maximum_characters:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            chunks.extend(
                word[start : start + maximum_characters]
                for start in range(0, len(word), maximum_characters)
            )
            continue

        candidate = f"{current_chunk} {word}".strip()

        if len(candidate) <= maximum_characters:
            current_chunk = candidate
        else:
            chunks.append(current_chunk)
            current_chunk = word

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def combine_wav_segments(segments: list[bytes]) -> bytes:
    """Combine compatible WAV segments into one WAV recording."""
    if not segments:
        raise ValueError("At least one WAV segment is required.")

    first_format: tuple[int, int, int, str, str] | None = None
    frames: list[bytes] = []

    for segment in segments:
        if not segment:
            raise ValueError("Generated WAV segments cannot be empty.")

        with wave.open(BytesIO(segment), "rb") as reader:
            segment_format = (
                reader.getnchannels(),
                reader.getsampwidth(),
                reader.getframerate(),
                reader.getcomptype(),
                reader.getcompname(),
            )

            if first_format is None:
                first_format = segment_format
            elif segment_format != first_format:
                raise ValueError("Generated WAV segments use incompatible formats.")

            frames.append(reader.readframes(reader.getnframes()))

    if first_format is None:
        raise ValueError("The WAV format could not be determined.")

    (
        channels,
        sample_width,
        frame_rate,
        compression_type,
        compression_name,
    ) = first_format

    output = BytesIO()

    with wave.open(output, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(sample_width)
        writer.setframerate(frame_rate)
        writer.setcomptype(compression_type, compression_name)

        for frame_data in frames:
            writer.writeframes(frame_data)

    return output.getvalue()


class VoiceService:
    """Transcribe voice questions and synthesize spoken answers."""

    def __init__(
        self,
        client: GroqAudioClient,
        model_name: str = DEFAULT_TRANSCRIPTION_MODEL,
        maximum_audio_size: int = MAX_AUDIO_SIZE_BYTES,
        tts_model_name: str = DEFAULT_TTS_MODEL,
        tts_voice: str = DEFAULT_TTS_VOICE,
        maximum_speech_chunk_characters: int = MAX_TTS_CHUNK_CHARACTERS,
    ) -> None:
        if not model_name.strip():
            raise ValueError("The transcription model name cannot be empty.")

        if maximum_audio_size <= 0:
            raise ValueError("The maximum audio size must be greater than zero.")

        if not tts_model_name.strip():
            raise ValueError("The TTS model name cannot be empty.")

        if not tts_voice.strip():
            raise ValueError("The TTS voice cannot be empty.")

        if maximum_speech_chunk_characters <= 0:
            raise ValueError("The speech chunk length must be greater than zero.")

        self._client = client
        self._model_name = model_name.strip()
        self._maximum_audio_size = maximum_audio_size
        self._tts_model_name = tts_model_name.strip()
        self._tts_voice = tts_voice.strip()
        self._maximum_speech_chunk_characters = maximum_speech_chunk_characters

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

    def synthesize(self, text: str) -> bytes:
        """Convert an answer into one playable WAV recording."""
        speech_chunks = split_text_for_speech(
            text,
            maximum_characters=self._maximum_speech_chunk_characters,
        )
        audio_segments: list[bytes] = []

        with TemporaryDirectory(prefix="pdf-reader-tts-") as directory:
            temporary_directory = Path(directory)

            for position, speech_chunk in enumerate(
                speech_chunks,
                start=1,
            ):
                response = self._client.audio.speech.create(
                    model=self._tts_model_name,
                    voice=self._tts_voice,
                    input=speech_chunk,
                    response_format="wav",
                )

                output_path = temporary_directory / f"segment-{position}.wav"
                response.write_to_file(output_path)
                audio_bytes = output_path.read_bytes()

                if not audio_bytes:
                    raise ValueError("The speech service returned empty audio.")

                audio_segments.append(audio_bytes)

        return combine_wav_segments(audio_segments)
