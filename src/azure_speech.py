"""Multilingual spoken answers with Azure AI Speech.

Groq's voices are English only, so a Tamil or Hindi answer is mispronounced.
Azure AI Speech has neural voices for 140+ locales, including Indian
languages, and its free tier covers 500,000 characters a month.

This module speaks to the REST endpoint directly instead of the Azure Speech
SDK, which is a large dependency for one HTTP call. Voices are discovered at
runtime from Azure's own voice list, so no voice name is hard-coded and the
app keeps working when Microsoft renames or adds voices.

Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION (for example "centralindia") to
enable it.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Protocol
from xml.sax import saxutils

LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT_FORMAT = "audio-24khz-48kbitrate-mono-mp3"
AUDIO_MIME_TYPE = "audio/mp3"
REQUEST_TIMEOUT_SECONDS = 30.0
MAX_SPEECH_CHARACTERS = 5000

# Voices whose locale is an Indian one are preferred for Indian languages, so
# Tamil is read by a ta-IN voice rather than a Sri Lankan Tamil voice.
PREFERRED_REGIONS = ("IN",)


class HTTPResponse(Protocol):
    status_code: int
    content: bytes
    text: str

    def json(self) -> object: ...


class HTTPClient(Protocol):
    """The small part of an HTTP client this module needs."""

    def get(
        self, url: str, *, headers: dict[str, str], timeout: float
    ) -> HTTPResponse: ...

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        content: bytes,
        timeout: float,
    ) -> HTTPResponse: ...


class AzureSpeechError(RuntimeError):
    """Raised when Azure Speech cannot produce audio."""


def azure_speech_settings() -> tuple[str, str]:
    """Read the key and region from the environment; empty when not configured."""
    return (
        os.getenv("AZURE_SPEECH_KEY", "").strip(),
        os.getenv("AZURE_SPEECH_REGION", "").strip(),
    )


def is_configured() -> bool:
    key, region = azure_speech_settings()
    return bool(key and region)


def build_ssml(text: str, voice: str, locale: str) -> str:
    """Wrap text in SSML, escaping it so document text cannot inject markup."""
    safe_text = saxutils.escape(text)
    return (
        f"<speak version='1.0' xml:lang='{locale}'>"
        f"<voice xml:lang='{locale}' name='{voice}'>{safe_text}</voice>"
        f"</speak>"
    )


@dataclass
class AzureSpeechService:
    """Turn an answer into speech using a voice that matches its language."""

    api_key: str
    region: str
    client: HTTPClient
    output_format: str = DEFAULT_OUTPUT_FORMAT
    _voices: list[dict] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError("The Azure Speech key cannot be empty.")

        if not self.region.strip():
            raise ValueError("The Azure Speech region cannot be empty.")

    @property
    def _base_url(self) -> str:
        return (
            f"https://{self.region.strip()}.tts.speech.microsoft.com/cognitiveservices"
        )

    def list_voices(self) -> list[dict]:
        """Fetch Azure's voice list once and remember it."""
        if self._voices is not None:
            return self._voices

        response = self.client.get(
            f"{self._base_url}/voices/list",
            headers={"Ocp-Apim-Subscription-Key": self.api_key.strip()},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        if response.status_code != 200:
            raise AzureSpeechError(
                f"Azure Speech returned {response.status_code} for the voice list."
            )

        voices = response.json()

        if not isinstance(voices, list) or not voices:
            raise AzureSpeechError("Azure Speech returned no voices.")

        self._voices = voices
        return voices

    def pick_voice(self, language: str | None) -> tuple[str, str]:
        """Return (voice name, locale) for a language code such as "ta"."""
        voices = self.list_voices()
        wanted = (language or "en").split("-")[0].lower()

        matches = [
            voice
            for voice in voices
            if str(voice.get("Locale", "")).lower().startswith(f"{wanted}-")
        ]

        if not matches:
            matches = [
                voice
                for voice in voices
                if str(voice.get("Locale", "")).lower().startswith("en-")
            ]

        if not matches:
            raise AzureSpeechError(f"No Azure voice is available for '{wanted}'.")

        for region in PREFERRED_REGIONS:
            regional = [
                voice
                for voice in matches
                if str(voice.get("Locale", "")).upper().endswith(f"-{region}")
            ]
            if regional:
                matches = regional
                break

        voice = matches[0]
        return str(voice["ShortName"]), str(voice["Locale"])

    def synthesize(self, text: str, language: str | None = "en") -> bytes:
        """Convert an answer into audio bytes in a matching voice."""
        cleaned_text = " ".join(text.split())

        if not cleaned_text:
            raise ValueError("The speech text cannot be empty.")

        if len(cleaned_text) > MAX_SPEECH_CHARACTERS:
            cleaned_text = cleaned_text[:MAX_SPEECH_CHARACTERS]

        voice, locale = self.pick_voice(language)

        response = self.client.post(
            f"{self._base_url}/v1",
            headers={
                "Ocp-Apim-Subscription-Key": self.api_key.strip(),
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": self.output_format,
                "User-Agent": "document-reader-chatbot",
            },
            content=build_ssml(cleaned_text, voice, locale).encode("utf-8"),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        if response.status_code != 200:
            raise AzureSpeechError(
                f"Azure Speech returned {response.status_code} while generating audio."
            )

        if not response.content:
            raise AzureSpeechError("Azure Speech returned empty audio.")

        return response.content


def create_azure_speech_service(
    api_key: str = "",
    region: str = "",
    client: HTTPClient | None = None,
) -> AzureSpeechService:
    """Build the service from arguments or environment settings."""
    if not api_key or not region:
        environment_key, environment_region = azure_speech_settings()
        api_key = api_key or environment_key
        region = region or environment_region

    if client is None:
        import httpx

        client = httpx.Client()

    return AzureSpeechService(api_key=api_key, region=region, client=client)
