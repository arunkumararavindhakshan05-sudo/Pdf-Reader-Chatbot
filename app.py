import hashlib
import logging
import os
import zipfile
from typing import TypedDict

import streamlit as st
from dotenv import load_dotenv
from pypdf.errors import PdfReadError

from src.injection_guard import InjectionFinding, sanitize_chunks, scan_chunks
from src.loaders import SUPPORTED_EXTENSIONS, describe_location, load_document
from src.ner import load_presidio_detector
from src.rag_service import RAGService, create_groq_model
from src.retriever import SearchResult, SemanticRetriever
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
    masked_counts: dict[str, int]
    table_result: object


st.set_page_config(
    page_title="Document Reader Chatbot",
    page_icon="📄",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def build_document_index(
    file_name: str,
    file_bytes: bytes,
) -> tuple[SemanticRetriever, int, int, list[InjectionFinding]]:
    """Extract a document once, scan it, and cache its semantic search index."""
    chunks, section_count = load_document(file_name, file_bytes)
    findings = scan_chunks(chunks)
    chunks = sanitize_chunks(chunks)

    if not chunks:
        raise ValueError(
            "No readable text was found. A PDF may contain only scanned images."
        )

    retriever = SemanticRetriever(chunks)
    return retriever, section_count, len(chunks), findings


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
                    st.audio(answer_audio, format="audio/wav")

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
    st.header("Retrieval settings")

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
    )

    st.caption(
        "Higher similarity thresholds reduce unrelated evidence but may "
        "reject questions that use different wording."
    )

    st.divider()
    st.header("Privacy")

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

    name_detector = get_name_detector() if redact_personal_data else None

    if redact_personal_data and name_detector is None:
        st.caption(
            "Name detection (Microsoft Presidio) is not installed, so only "
            "pattern-based data, spreadsheet name columns and addresses with a "
            "PIN code are masked."
        )

    st.divider()
    st.header("Spoken answers")

    read_answers_aloud = st.toggle(
        "Read new answers aloud",
        value=False,
        disabled=not api_key,
        help=(
            "When enabled, each new answer is converted to WAV audio "
            "using Groq text-to-speech."
        ),
    )

    selected_voice_name = st.selectbox(
        "Answer voice",
        options=list(TTS_VOICES),
        index=0,
        disabled=not read_answers_aloud,
    )
    selected_voice = TTS_VOICES[selected_voice_name]

    st.caption(
        "Speech is generated only for new answers while this option is "
        "enabled. This avoids unnecessary API usage."
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

try:
    with st.spinner("Reading the document and building its semantic index..."):
        retriever, section_count, chunk_count, injection_findings = (
            build_document_index(uploaded_file.name, file_bytes)
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

if audio_recording is not None:
    audio_bytes = audio_recording.getvalue()
    audio_hash = hashlib.sha256(audio_bytes).hexdigest()

    if st.session_state.get("last_audio_hash") != audio_hash:
        try:
            with st.spinner("Transcribing your voice question..."):
                audio_client = create_groq_audio_client(api_key)
                voice_service = VoiceService(client=audio_client)
                voice_question = voice_service.transcribe(
                    audio_bytes=audio_bytes,
                    filename=audio_recording.name or "question.wav",
                )

            st.session_state.last_audio_hash = audio_hash
            st.success(f"Transcribed question: {voice_question}")
        except ValueError as error:
            LOGGER.exception("Voice recording validation failed.")
            st.error(str(error))
        except Exception:
            LOGGER.exception("Voice transcription failed.")
            st.error("The voice question could not be transcribed.")

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

            if read_answers_aloud:
                try:
                    with st.spinner("Generating the spoken answer..."):
                        audio_client = create_groq_audio_client(api_key)
                        voice_service = VoiceService(
                            client=audio_client,
                            tts_voice=selected_voice,
                        )
                        answer_audio = voice_service.synthesize(result["answer"])

                    st.audio(answer_audio, format="audio/wav")
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

            if masked_counts:
                assistant_message["masked_counts"] = masked_counts

            if table_answer is not None:
                assistant_message["table_result"] = table_answer.result

            st.session_state.messages.append(assistant_message)
        except Exception:
            LOGGER.exception("The RAG answer request failed.")
            st.error("The AI request failed. Please try again.")
