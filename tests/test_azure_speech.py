import pytest

from src.azure_speech import (
    AUDIO_MIME_TYPE,
    MAX_SPEECH_CHARACTERS,
    AzureSpeechError,
    AzureSpeechService,
    build_ssml,
    create_azure_speech_service,
    is_configured,
)

VOICES = [
    {"ShortName": "en-US-JennyNeural", "Locale": "en-US"},
    {"ShortName": "ta-LK-SaranyaNeural", "Locale": "ta-LK"},
    {"ShortName": "ta-IN-PallaviNeural", "Locale": "ta-IN"},
    {"ShortName": "hi-IN-SwaraNeural", "Locale": "hi-IN"},
]


class FakeResponse:
    def __init__(
        self, status_code: int = 200, content: bytes = b"", payload: object = None
    ) -> None:
        self.status_code = status_code
        self.content = content
        self.text = content.decode("utf-8", errors="replace")
        self._payload = payload

    def json(self) -> object:
        return self._payload


class FakeHTTPClient:
    def __init__(
        self,
        voices: object = VOICES,
        voices_status: int = 200,
        speech_status: int = 200,
        audio: bytes = b"AUDIO",
    ) -> None:
        self.voices = voices
        self.voices_status = voices_status
        self.speech_status = speech_status
        self.audio = audio
        self.get_calls: list[tuple[str, dict]] = []
        self.post_calls: list[tuple[str, dict, bytes]] = []

    def get(self, url: str, *, headers: dict, timeout: float) -> FakeResponse:
        self.get_calls.append((url, headers))
        return FakeResponse(status_code=self.voices_status, payload=self.voices)

    def post(
        self, url: str, *, headers: dict, content: bytes, timeout: float
    ) -> FakeResponse:
        self.post_calls.append((url, headers, content))
        return FakeResponse(status_code=self.speech_status, content=self.audio)


def make_service(client: FakeHTTPClient | None = None) -> AzureSpeechService:
    return AzureSpeechService(
        api_key="key", region="centralindia", client=client or FakeHTTPClient()
    )


def test_indian_voice_is_preferred_for_tamil() -> None:
    service = make_service()

    assert service.pick_voice("ta") == ("ta-IN-PallaviNeural", "ta-IN")


def test_unknown_language_falls_back_to_english() -> None:
    assert make_service().pick_voice("xx")[1].startswith("en-")


def test_voice_list_is_fetched_once() -> None:
    client = FakeHTTPClient()
    service = make_service(client)

    service.pick_voice("ta")
    service.pick_voice("hi")

    assert len(client.get_calls) == 1
    assert client.get_calls[0][1]["Ocp-Apim-Subscription-Key"] == "key"


def test_synthesize_posts_ssml_and_returns_audio() -> None:
    client = FakeHTTPClient()
    service = make_service(client)

    audio = service.synthesize("வணக்கம்", language="ta")

    url, headers, content = client.post_calls[0]
    assert audio == b"AUDIO"
    assert url == "https://centralindia.tts.speech.microsoft.com/cognitiveservices/v1"
    assert headers["Content-Type"] == "application/ssml+xml"
    assert b"ta-IN-PallaviNeural" in content
    assert "வணக்கம்" in content.decode("utf-8")


def test_ssml_escapes_document_text() -> None:
    ssml = build_ssml("Ampersand & <voice name='evil'>", "en-US-JennyNeural", "en-US")

    assert "&amp;" in ssml
    assert "<voice name='evil'>" not in ssml
    assert ssml.count("<voice") == 1


def test_long_answers_are_truncated() -> None:
    client = FakeHTTPClient()

    make_service(client).synthesize("word " * 4000, language="en")

    spoken = client.post_calls[0][2].decode("utf-8")
    assert spoken.count("word") <= MAX_SPEECH_CHARACTERS


@pytest.mark.parametrize(
    ("client", "message"),
    [
        (FakeHTTPClient(voices_status=401), "401"),
        (FakeHTTPClient(voices=[]), "no voices"),
        (FakeHTTPClient(speech_status=429), "429"),
        (FakeHTTPClient(audio=b""), "empty audio"),
    ],
)
def test_azure_failures_raise_clear_errors(
    client: FakeHTTPClient, message: str
) -> None:
    with pytest.raises(AzureSpeechError, match=message):
        make_service(client).synthesize("hello", language="en")


def test_empty_text_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        make_service().synthesize("   ")


@pytest.mark.parametrize(("key", "region"), [("", "centralindia"), ("key", "  ")])
def test_missing_settings_are_rejected(key: str, region: str) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        AzureSpeechService(api_key=key, region=region, client=FakeHTTPClient())


def test_is_configured_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AZURE_SPEECH_KEY", raising=False)
    monkeypatch.delenv("AZURE_SPEECH_REGION", raising=False)
    assert not is_configured()

    monkeypatch.setenv("AZURE_SPEECH_KEY", "key")
    monkeypatch.setenv("AZURE_SPEECH_REGION", "centralindia")
    assert is_configured()

    service = create_azure_speech_service(client=FakeHTTPClient())
    assert service.region == "centralindia"


def test_audio_mime_type_matches_the_requested_format() -> None:
    client = FakeHTTPClient()
    make_service(client).synthesize("hello", language="en")

    assert "mp3" in client.post_calls[0][1]["X-Microsoft-OutputFormat"]
    assert AUDIO_MIME_TYPE == "audio/mp3"
