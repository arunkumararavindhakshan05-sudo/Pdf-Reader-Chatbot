import hashlib
import logging
import os
from typing import TypedDict

import streamlit as st
from dotenv import load_dotenv
from pypdf.errors import PdfReadError

from src.document_processor import extract_chunks
from src.rag_service import RAGService, create_groq_model
from src.retriever import SearchResult, SemanticRetriever
from src.voice_service import VoiceService, create_groq_audio_client

LOGGER = logging.getLogger(__name__)


class ChatMessage(TypedDict, total=False):
    """One user or assistant message stored in Streamlit state."""

    role: str
    content: str
    evidence: list[SearchResult]


st.set_page_config(
    page_title="PDF Intelligence Assistant",
    page_icon="📄",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def build_document_index(
    pdf_bytes: bytes,
) -> tuple[SemanticRetriever, int, int]:
    """Extract a PDF once and cache its semantic search index."""
    chunks, page_count = extract_chunks(pdf_bytes)

    if not chunks:
        raise ValueError(
            "No readable text was found. The PDF may contain scanned images."
        )

    retriever = SemanticRetriever(chunks)
    return retriever, page_count, len(chunks)


def get_groq_api_key() -> str:
    """Read the Groq key from local environment or Streamlit secrets."""
    load_dotenv()

    environment_key = os.getenv("GROQ_API_KEY", "").strip()
    if environment_key:
        return environment_key

    try:
        return str(st.secrets.get("GROQ_API_KEY", "")).strip()
    except Exception:  # noqa: BLE001
        return ""


def render_evidence(evidence: list[SearchResult]) -> None:
    """Display retrieved PDF sections and similarity scores."""
    if not evidence:
        return

    with st.expander("View retrieved evidence and similarity scores"):
        for position, result in enumerate(evidence, start=1):
            st.markdown(
                f"**Match {position} — Page {result['page']} — "
                f"score {result['score']:.3f}**"
            )
            st.write(result["text"])

            if position < len(evidence):
                st.divider()


def render_chat_history(messages: list[ChatMessage]) -> None:
    """Render previous questions and answers for the active PDF."""
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                render_evidence(message.get("evidence", []))


st.title("PDF Intelligence Assistant")
st.caption(
    "Ask grounded questions about a PDF using semantic retrieval, "
    "FAISS, Groq, and page-aware citations."
)

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

uploaded_pdf = st.file_uploader(
    "Upload one text-based PDF",
    type=["pdf"],
    accept_multiple_files=False,
)

api_key = get_groq_api_key()

if not api_key:
    st.warning(
        "GROQ_API_KEY is not configured. Add it to your local environment "
        "or Streamlit secrets before asking questions."
    )

if uploaded_pdf is None:
    st.info("Upload a PDF to create its searchable semantic index.")
    st.stop()

pdf_bytes = uploaded_pdf.getvalue()
document_hash = hashlib.sha256(pdf_bytes).hexdigest()

try:
    with st.spinner("Reading the PDF and building its semantic index..."):
        retriever, page_count, chunk_count = build_document_index(pdf_bytes)
except (PdfReadError, ValueError) as error:
    LOGGER.exception("The uploaded PDF could not be processed.")
    st.error(str(error))
    st.stop()
except Exception as error:  # noqa: BLE001
    LOGGER.exception("Unexpected document indexing failure.")
    st.error("The document index could not be created.")
    st.caption(f"{type(error).__name__}: {error}")
    st.stop()

if st.session_state.get("active_document_hash") != document_hash:
    st.session_state.active_document_hash = document_hash
    st.session_state.messages = []
    st.session_state.last_audio_hash = None

if "messages" not in st.session_state:
    st.session_state.messages = []

st.success(
    f"Indexed {chunk_count} searchable sections across {page_count} pages."
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
        except Exception as error:  # noqa: BLE001
            LOGGER.exception("Voice transcription failed.")
            st.error("The voice question could not be transcribed.")
            st.caption(f"{type(error).__name__}: {error}")

typed_question = st.chat_input(
    "Ask a question about the uploaded PDF",
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
            with st.spinner("Retrieving evidence and generating an answer..."):
                model = create_groq_model(api_key)
                service = RAGService(
                    retriever=retriever,
                    model=model,
                    top_k=top_k,
                    minimum_score=minimum_score,
                )
                result = service.answer(question)

            st.markdown(result["answer"])
            render_evidence(result["evidence"])

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "evidence": result["evidence"],
                }
            )
        except Exception as error:  # noqa: BLE001
            LOGGER.exception("The RAG answer request failed.")
            st.error("The AI request failed. Please try again.")
            st.caption(f"{type(error).__name__}: {error}")
