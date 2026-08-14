import numpy as np
import pytest

from src.document_processor import DocumentChunk
from src.retriever import SemanticRetriever


class FakeEmbeddingModel:
    """Return predictable embeddings without downloading an AI model."""

    def __init__(
        self,
        vectors: dict[str, list[float]],
    ) -> None:
        self.vectors = vectors

    def encode(
        self,
        sentences: list[str],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        assert convert_to_numpy is True
        assert normalize_embeddings is True

        return np.asarray(
            [self.vectors[sentence] for sentence in sentences],
            dtype=np.float32,
        )


def make_chunks() -> list[DocumentChunk]:
    """Create small document chunks for retriever tests."""
    return [
        {
            "text": "Machine learning discovers patterns in data.",
            "page": 3,
            "chunk": 1,
        },
        {
            "text": "Cooking pasta requires boiling water.",
            "page": 8,
            "chunk": 2,
        },
    ]


def make_model() -> FakeEmbeddingModel:
    """Create deterministic vectors for documents and questions."""
    return FakeEmbeddingModel(
        {
            "Machine learning discovers patterns in data.": [1.0, 0.0],
            "Cooking pasta requires boiling water.": [0.0, 1.0],
            "How does machine learning work?": [1.0, 0.0],
            "General question": [0.5, 0.5],
        }
    )


def test_search_returns_best_semantic_match() -> None:
    """The most semantically similar document should be ranked first."""
    retriever = SemanticRetriever(
        chunks=make_chunks(),
        model=make_model(),
    )

    results = retriever.search(
        "How does machine learning work?",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["page"] == 3
    assert results[0]["chunk"] == 1
    assert results[0]["text"].startswith("Machine learning")
    assert results[0]["score"] == pytest.approx(1.0)


def test_search_limits_results_to_available_chunks() -> None:
    """Requesting extra results should not create nonexistent matches."""
    retriever = SemanticRetriever(
        chunks=make_chunks(),
        model=make_model(),
    )

    results = retriever.search(
        "General question",
        top_k=10,
    )

    assert len(results) == 2


@pytest.mark.parametrize(
    "question",
    [
        "",
        "   ",
    ],
)
def test_search_rejects_empty_question(question: str) -> None:
    """Questions must contain visible text."""
    retriever = SemanticRetriever(
        chunks=make_chunks(),
        model=make_model(),
    )

    with pytest.raises(ValueError, match="cannot be empty"):
        retriever.search(question)


@pytest.mark.parametrize(
    "top_k",
    [
        0,
        -1,
    ],
)
def test_search_rejects_invalid_top_k(top_k: int) -> None:
    """The requested result count must be positive."""
    retriever = SemanticRetriever(
        chunks=make_chunks(),
        model=make_model(),
    )

    with pytest.raises(ValueError, match="greater than zero"):
        retriever.search(
            "General question",
            top_k=top_k,
        )


def test_retriever_rejects_missing_chunks() -> None:
    """A search index cannot be created without document content."""
    with pytest.raises(ValueError, match="At least one"):
        SemanticRetriever(
            chunks=[],
            model=make_model(),
        )


def test_retriever_rejects_empty_chunk_text() -> None:
    """Empty document sections should not enter the search index."""
    empty_chunk: DocumentChunk = {
        "text": "   ",
        "page": 1,
        "chunk": 1,
    }

    with pytest.raises(ValueError, match="empty text"):
        SemanticRetriever(
            chunks=[empty_chunk],
            model=make_model(),
        )