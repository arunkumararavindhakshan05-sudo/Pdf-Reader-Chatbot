import hashlib
import logging
import os
import zipfile
from typing import TypedDict

import streamlit as st
from dotenv import load_dotenv
from pypdf.errors import PdfReadError

from src.azure_speech import (
    AUDIO_MIME_TYPE as AZURE_AUDIO_MIME_TYPE,
)
from src.azure_speech import (
    AzureSpeechError,
    create_azure_speech_service,
)
from src.azure_speech import (
    is_configured as azure_speech_configured,
)
from src.injection_guard import InjectionFinding, sanitize_chunks, scan_chunks
from src.language import detect_language
from src.loaders import SUPPORTED_EXTENSIONS, describe_location, load_document
from src.ner import load_presidio_detector
from src.rag_service import RAGService, create_groq_model
from src.retriever import SearchResult, SemanticRetriever
from src.summarizer import (
    SummaryError,
    summarize_document,
    summary_chunk,
)
from src.table_engine import (
    TablePlanError,
    TableQAService,
    is_table_file,
    load_tables,
    looks_like_table_question,
    sensitive_column_values,
)
from src.voice_service import VoiceService, create_groq_audio_client

LOGGER = logging.getLogger(__name__)

TTS_VOICES = {
    "Hannah": "hannah",
    "Autumn": "autumn",
    "Diana": "diana",
    "Austin": "austin",
    "Daniel": "daniel",
    "Troy": "troy",
}


class ChatMessage(TypedDict, total=False):
    """One user or assistant message stored in Streamlit state."""

    role: str
    content: str
    evidence: list[SearchResult]
    audio: bytes
    audio_format: str
    masked_counts: dict[str, int]
    table_result: object


st.set_page_config(
    page_title="Document Reader Chatbot",
    page_icon="📄",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def read_document(
    file_name: str,
    file_bytes: bytes,
) -> tuple[list[dict], int, list[InjectionFinding]]:
    """Extract a document's chunks once and scan them for injected instructions."""
    chunks, section_count = load_document(file_name, file_bytes)
    findings = scan_chunks(chunks)
    chunks = sanitize_chunks(chunks)

    if not chunks:
        raise ValueError(
            "No readable text was found. A PDF may contain only scanned images."
        )

    return chunks, section_count, findings


@st.cache_data(show_spinner=False)
def build_summary(file_name: str, file_bytes: bytes, api_key: str) -> str:
    """Summarise a document once. Failures are not cached, so they are retried.

    Streamlit caches returned values but not exceptions, so keeping this in its
    own function means a temporary AI outage does not leave the document
    permanently without a summary.
    """
    chunks, _, _ = read_document(file_name, file_bytes)

    return summarize_document(
        chunks,
        model=create_groq_model(api_key),
        redact_personal_data=True,
        entity_detector=get_name_detector(),
    )


@st.cache_resource(show_spinner=False)
def build_document_index(
    file_name: str,
    file_bytes: bytes,
    summary: str,
) -> tuple[SemanticRetriever, int]:
    """Index a document, with its whole-document summary as one extra chunk.

    Storing the summary in the index is what lets questions about the document
    as a whole ("whose resume is this?", "summarise it") find an answer, while
    ordinary questions still retrieve the passage that answers them.
    """
    chunks, _, _ = read_document(file_name, file_bytes)

    if summary:
        chunks = [summary_chunk(summary), *chunks]

    return SemanticRetriever(chunks), len(chunks)


@st.cache_resource(show_spinner=False)
def build_tables(file_name: str, file_bytes: bytes) -> dict:
    """Load spreadsheet sheets as DataFrames for exact calculations."""
    if not is_table_file(file_name):
        return {}

    return load_tables(file_name, file_bytes)


@st.cache_resource(show_spinner=False)
def get_name_detector():
    """Load Microsoft Presidio once per container, if it is installed."""
    return load_presidio_detector()


def get_groq_api_key() -> str:
    """Read the Groq key from local environment or Streamlit secrets."""
    load_dotenv()

    environment_key = os.getenv("GROQ_API_KEY", "").strip()
    if environment_key:
        return environment_key

    try:
        return str(st.secrets["GROQ_API_KEY"]).strip()
    except (FileNotFoundError, KeyError):
        return ""


def render_evidence(evidence: list[SearchResult]) -> None:
    """Display retrieved document sections and similarity scores."""
    if not evidence:
        return

    with st.expander("View retrieved evidence and similarity scores"):
        for position, result in enumerate(evidence, start=1):
            st.markdown(
                f"**Match {position} — {describe_location(result)} — "
                f"score {result['score']:.3f}**"
            )
            st.write(result["text"])

            if position < len(evidence):
                st.divider()


def answer_language_for(answer: str, spoken_language: str | None) -> str:
    """Pick the language to speak an answer in, without asking the user."""
    if spoken_language:
        return spoken_language

    return detect_language(answer)


def render_masking_note(masked_counts: dict[str, int]) -> None:
    """Tell the user which personal data was hidden from the AI model."""
    if not masked_counts:
        return

    details = ", ".join(
        f"{count} {entity.replace('_', ' ').lower()}"
        for entity, count in sorted(masked_counts.items())
    )
    st.caption(f"Privacy: masked before sending to the AI model: {details}.")


def render_injection_warning(findings: list[InjectionFinding]) -> None:
    """Warn when the document contains text that tries to steer the AI."""
    if not findings:
        return

    st.warning(
        f"This document contains {len(findings)} "
        f"{'passage that looks' if len(findings) == 1 else 'passages that look'} "
        "like instructions to the AI. They are treated as plain document text, "
        "but check answers carefully."
    )

    with st.expander("View suspicious passages"):
        for finding in findings[:20]:
            st.markdown(
                f"**{finding.location}** — `{finding.rule}` ({finding.severity})"
            )
            st.write(finding.excerpt)


def render_chat_history(messages: list[ChatMessage]) -> None:
    """Render previous questions, answers, and generated answer audio."""
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                answer_audio = message.get("audio")
                if answer_audio:
                    st.audio(
                        answer_audio,
                        format=message.get("audio_format", "audio/wav"),
                    )

                table_result = message.get("table_result")
                if table_result is not None:
                    st.dataframe(table_result, hide_index=True)

                render_masking_note(message.get("masked_counts", {}))
                render_evidence(message.get("evidence", []))


st.title("Document Reader Chatbot")
st.caption(
    "Upload a PDF, Word, Excel, CSV, PowerPoint or text file and ask questions "
    "about it by typing or speaking."
)

api_key = get_groq_api_key()

with st.sidebar:
    st.header("Settings")

    redact_personal_data = st.toggle(
        "Mask personal data before sending to AI",
        value=True,
        help=(
            "Names, addresses, emails, phone numbers, Aadhaar, PAN, IFSC, card "
            "numbers and IP addresses are replaced with placeholders before the "
            "question and evidence reach the AI model. Answers show the real "
            "values again."
        ),
    )

    read_answers_aloud = st.toggle(
        "Read answers aloud",
        value=False,
        disabled=not api_key,
        help=(
            "Answers are spoken in the language you asked in. The language is "
            "detected automatically and a matching voice is chosen for you."
        ),
    )

    if read_answers_aloud and not azure_speech_configured():
        st.caption(
            "Only English voices are available. Add AZURE_SPEECH_KEY and "
            "AZURE_SPEECH_REGION to hear answers in other languages."
        )

    name_detector = get_name_detector() if redact_personal_data else None

    # Everything below is for tuning, not for everyday use, so it stays folded
    # away: the defaults are what most questions should run with.
    with st.expander("Advanced"):
        MASKABLE_TYPES = {
            "Names": "PERSON",
            "Places": "LOCATION",
            "Addresses": "ADDRESS",
            "Emails": "EMAIL",
            "Phone numbers": "PHONE",
            "Aadhaar": "AADHAAR",
            "PAN": "PAN",
            "Bank IFSC": "IFSC",
            "Card numbers": "CARD_NUMBER",
            "PIN codes": "PIN_CODE",
            "IP addresses": "IP_ADDRESS",
        }

        selected_types = st.multiselect(
            "Data types to mask",
            options=list(MASKABLE_TYPES),
            default=list(MASKABLE_TYPES),
            disabled=not redact_personal_data,
            help=(
                "Remove a type to let the AI see it, for example Names when you "
                "ask whose resume this is."
            ),
        )
        masked_entities = frozenset(MASKABLE_TYPES[label] for label in selected_types)

        if redact_personal_data and name_detector is None:
            st.caption(
                "Name detection (Microsoft Presidio) is not installed, so only "
                "pattern-based data, spreadsheet name columns and addresses "
                "with a PIN code are masked."
            )

        top_k = st.slider(
            "Number of candidate sections",
            min_value=1,
            max_value=10,
            value=5,
        )

        minimum_score = st.slider(
            "Minimum similarity score",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help=(
                "Higher values reduce unrelated evidence but may reject "
                "questions worded differently from the document."
            ),
        )

uploaded_file = st.file_uploader(
    "Upload one document",
    type=list(SUPPORTED_EXTENSIONS),
    accept_multiple_files=False,
)

if not api_key:
    st.warning(
        "GROQ_API_KEY is not configured. Add it to your local environment "
        "or Streamlit secrets before asking questions."
    )

if uploaded_file is None:
    st.info("Upload a document to create its searchable semantic index.")
    st.stop()

file_bytes = uploaded_file.getvalue()
document_hash = hashlib.sha256(file_bytes).hexdigest()

summary = ""

try:
    with st.spinner("Reading the document..."):
        _, section_count, injection_findings = read_document(
            uploaded_file.name, file_bytes
        )

    if api_key:
        try:
            with st.spinner("Summarising the document..."):
                summary = build_summary(uploaded_file.name, file_bytes, api_key)
        except (SummaryError, ValueError):
            LOGGER.exception("The document summary could not be generated.")
        except Exception:
            LOGGER.exception("Unexpected failure while summarising the document.")

    with st.spinner("Building the semantic index..."):
        retriever, chunk_count = build_document_index(
            uploaded_file.name, file_bytes, summary
        )
        tables = build_tables(uploaded_file.name, file_bytes)
except (PdfReadError, ValueError) as error:
    LOGGER.exception("The uploaded document could not be processed.")
    st.error(str(error))
    st.stop()
except (zipfile.BadZipFile, KeyError):
    LOGGER.exception("The uploaded Office file is damaged or not a real Office file.")
    st.error("The file could not be opened. It may be damaged or password-protected.")
    st.stop()
except Exception:
    LOGGER.exception("Unexpected document indexing failure.")
    st.error("The document index could not be created.")
    st.stop()

if st.session_state.get("active_document_hash") != document_hash:
    st.session_state.active_document_hash = document_hash
    st.session_state.messages = []
    st.session_state.last_audio_hash = None

if "messages" not in st.session_state:
    st.session_state.messages = []

st.success(
    f"Document ready: {section_count} "
    f"{'section' if section_count == 1 else 'sections'} (pages, sheets, slides or "
    f"paragraph groups) and {chunk_count} searchable "
    f"{'chunk' if chunk_count == 1 else 'chunks'}."
)
render_injection_warning(injection_findings)

if summary:
    with st.expander("Document summary"):
        st.markdown(summary.split("\n\n", 1)[-1])
        st.caption(
            "Generated once when the document was uploaded, and stored in the "
            "search index so questions about the whole document can find it."
        )

render_chat_history(st.session_state.messages)

st.subheader("Ask the document")
st.caption(
    "Record a voice question or type a question below. Microphone audio "
    "is transcribed with Groq Whisper."
)

audio_recording = st.audio_input(
    "Record a voice question",
    sample_rate=16000,
    key=f"voice-question-{document_hash}",
    disabled=not api_key,
)

voice_question = None
spoken_answer_language: str | None = None

if audio_recording is not None:
    audio_bytes = audio_recording.getvalue()
    audio_hash = hashlib.sha256(audio_bytes).hexdigest()

    if st.session_state.get("last_audio_hash") != audio_hash:
        try:
            with st.spinner("Transcribing your voice question..."):
                audio_client = create_groq_audio_client(api_key)
                voice_service = VoiceService(client=audio_client)
                # The summary gives Whisper the document's own vocabulary, so
                # names and technical terms are spelled the way they appear.
                st.session_state.pending_voice_question = voice_service.transcribe(
                    audio_bytes=audio_bytes,
                    filename=audio_recording.name or "question.wav",
                    context=summary,
                )
                # Whisper reports the language it heard; it beats guessing from
                # the transcript, and it needs no setting in the interface.
                st.session_state.detected_language = (
                    voice_service.last_detected_language
                )

            st.session_state.last_audio_hash = audio_hash
        except ValueError as error:
            LOGGER.exception("Voice recording validation failed.")
            st.error(str(error))
        except Exception:
            LOGGER.exception("Voice transcription failed.")
            st.error("The voice question could not be transcribed.")

pending_voice_question = st.session_state.get("pending_voice_question")

if pending_voice_question:
    st.caption("Check the transcription, correct it if needed, then send it.")
    edited_question = st.text_input(
        "Transcribed question",
        value=pending_voice_question,
        key=f"voice-text-{document_hash}-{st.session_state.get('last_audio_hash', '')}",
    )

    send_column, discard_column = st.columns([1, 1])

    if send_column.button("Send this question", type="primary"):
        voice_question = edited_question.strip()
        spoken_answer_language = st.session_state.get("detected_language")
        st.session_state.pending_voice_question = None

    if discard_column.button("Discard recording"):
        st.session_state.pending_voice_question = None
        st.rerun()

typed_question = st.chat_input(
    "Ask a question about the uploaded document",
    disabled=not api_key,
)

question = voice_question or typed_question

if question:
    user_message: ChatMessage = {
        "role": "user",
        "content": question,
    }
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            sensitive_values = (
                sensitive_column_values(tables) if redact_personal_data else {}
            )
            model = create_groq_model(api_key)
            table_answer = None

            if tables and looks_like_table_question(question):
                try:
                    with st.spinner("Calculating from the spreadsheet with pandas..."):
                        table_answer = TableQAService(
                            tables=tables,
                            model=model,
                            redact_personal_data=redact_personal_data,
                            entity_detector=name_detector,
                            masked_entities=masked_entities,
                        ).answer(question)
                except (TablePlanError, KeyError, TypeError, ValueError):
                    LOGGER.warning(
                        "Table query failed; falling back to retrieval.",
                        exc_info=True,
                    )
                    table_answer = None

            if table_answer is not None:
                result = {
                    "answer": table_answer.answer,
                    "evidence": [],
                    "masked_counts": table_answer.masked_counts,
                }
            else:
                with st.spinner("Retrieving evidence and generating an answer..."):
                    service = RAGService(
                        retriever=retriever,
                        model=model,
                        top_k=top_k,
                        minimum_score=minimum_score,
                        redact_personal_data=redact_personal_data,
                        entity_detector=name_detector,
                        sensitive_values=sensitive_values,
                        masked_entities=masked_entities,
                    )
                    result = service.answer(question)

            st.markdown(result["answer"])

            if table_answer is not None:
                st.dataframe(table_answer.result, hide_index=True)
                st.caption(
                    f"Calculated with pandas from sheet '{table_answer.plan.sheet}' "
                    "using all rows."
                )

            answer_audio: bytes | None = None
            answer_audio_format = "audio/wav"

            if read_answers_aloud:
                try:
                    with st.spinner("Generating the spoken answer..."):
                        # A spoken question carries the language Whisper heard;
                        # a typed one is identified from its script. Azure is
                        # used when configured, because its voices cover every
                        # language the app can answer in.
                        answer_language = answer_language_for(
                            result["answer"],
                            spoken_answer_language,
                        )

                        if azure_speech_configured():
                            answer_audio = create_azure_speech_service().synthesize(
                                result["answer"],
                                language=answer_language,
                            )
                            answer_audio_format = AZURE_AUDIO_MIME_TYPE
                        else:
                            audio_client = create_groq_audio_client(api_key)
                            voice_service = VoiceService(client=audio_client)
                            answer_audio = voice_service.synthesize(result["answer"])

                    st.audio(answer_audio, format=answer_audio_format)
                except AzureSpeechError:
                    LOGGER.exception("Azure speech generation failed.")
                    st.warning(
                        "The text answer is ready, but Azure Speech could not "
                        "generate audio. Check the key, region and quota."
                    )
                except Exception:
                    LOGGER.exception("Answer speech generation failed.")
                    st.warning(
                        "The text answer is ready, but its audio could not "
                        "be generated."
                    )

            masked_counts = result.get("masked_counts", {})
            render_masking_note(masked_counts)
            render_evidence(result["evidence"])

            assistant_message: ChatMessage = {
                "role": "assistant",
                "content": result["answer"],
                "evidence": result["evidence"],
            }

            if answer_audio:
                assistant_message["audio"] = answer_audio
                assistant_message["audio_format"] = answer_audio_format

            if masked_counts:
                assistant_message["masked_counts"] = masked_counts

            if table_answer is not None:
                assistant_message["table_result"] = table_answer.result

            st.session_state.messages.append(assistant_message)
        except Exception:
            LOGGER.exception("The RAG answer request failed.")
            st.error("The AI request failed. Please try again.")
