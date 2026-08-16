from typing import Protocol, TypedDict

import faiss
import numpy as np

from src.document_processor import DocumentChunk

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class SearchResult(TypedDict):
    """A retrieved PDF section and its semantic similarity score."""

    text: str
    page: int
    chunk: int
    score: float


class EmbeddingModel(Protocol):
    """Interface required from an embedding model."""

    def encode(
        self,
        sentences: list[str],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        """Convert text into numerical vectors."""
        ...


def load_embedding_model(model_name: str) -> EmbeddingModel:
    """Load the sentence-transformer model only when it is needed."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class SemanticRetriever:
    """Search PDF chunks using embeddings and FAISS."""

    def __init__(
        self,
        chunks: list[DocumentChunk],
        model: EmbeddingModel | None = None,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        if not chunks:
            raise ValueError("At least one document chunk is required.")

        if any(not chunk["text"].strip() for chunk in chunks):
            raise ValueError("Document chunks cannot contain empty text.")

        self._chunks = list(chunks)
        self._model = model or load_embedding_model(model_name)

        document_texts = [chunk["text"] for chunk in self._chunks]
        document_embeddings = self._encode(document_texts)

        if document_embeddings.shape[0] != len(self._chunks):
            raise ValueError(
                "The embedding model returned an unexpected number of vectors."
            )

        self._dimension = document_embeddings.shape[1]
        self._index = faiss.IndexFlatIP(self._dimension)
        self._index.add(document_embeddings)

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Create normalized, FAISS-compatible embeddings."""
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embeddings_array = np.asarray(embeddings, dtype=np.float32)

        if embeddings_array.ndim == 1:
            embeddings_array = embeddings_array.reshape(1, -1)

        if embeddings_array.ndim != 2 or embeddings_array.shape[1] == 0:
            raise ValueError(
                "The embedding model returned vectors with an invalid shape."
            )

        return np.ascontiguousarray(embeddings_array)

    def search(
        self,
        question: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Return the PDF chunks most relevant to a question."""
        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError("The search question cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        question_embedding = self._encode([cleaned_question])

        if question_embedding.shape[1] != self._dimension:
            raise ValueError(
                "The question embedding dimension does not match the index."
            )

        result_count = min(top_k, len(self._chunks))
        scores, indices = self._index.search(
            question_embedding,
            result_count,
        )

        results: list[SearchResult] = []

        for score, index in zip(
            scores[0],
            indices[0],
            strict=True,
        ):
            if index < 0:
                continue

            chunk = self._chunks[int(index)]

            results.append(
                {
                    "text": chunk["text"],
                    "page": chunk["page"],
                    "chunk": chunk["chunk"],
                    "score": float(score),
                }
            )

        return results
