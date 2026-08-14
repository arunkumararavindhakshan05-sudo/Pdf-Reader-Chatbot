from dataclasses import dataclass

import pytest

from src.rag_service import (
    DEFAULT_FALLBACK_ANSWER,
    RAGService,
    build_grounded_prompt,
    create_groq_model,
)
from src.retriever import SearchResult


@dataclass
class FakeResponse:
    """A predictable response returned by the fake chat model."""

    content: str


class FakeChatModel:
    """Record prompts and return a configured response."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> FakeResponse:
        self.prompts.append(prompt)
        return FakeResponse(content=self.response_text)


class FakeRetriever:
    """Return controlled evidence without running an embedding model."""

    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(
        self,
        question: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        self.calls.append((question, top_k))
        return self.results


def make_evidence(
    *,
    text: str = "Machine learning discovers patterns in data.",
    page: int = 3,
    chunk: int = 1,
    score: float = 0.9,
) -> SearchResult:
    """Create one retrieved evidence record."""
    return {
        "text": text,
        "page": page,
        "chunk": chunk,
        "score": score,
    }


def test_answer_uses_only_evidence_above_threshold() -> None:
    """Low-scoring evidence should not be sent to the language model."""
    strong_evidence = make_evidence(score=0.9)
    weak_evidence = make_evidence(
        text="This section is not sufficiently relevant.",
        page=9,
        score=0.1,
    )

    retriever = FakeRetriever([strong_evidence, weak_evidence])
    model = FakeChatModel(
        "Machine learning discovers patterns in data [Page 3]."
    )

    service = RAGService(
        retriever=retriever,
        model=model,
        top_k=4,
        minimum_score=0.25,
    )

    result = service.answer("What is machine learning?")

    assert retriever.calls == [("What is machine learning?", 4)]
    assert result["evidence"] == [strong_evidence]
    assert result["answer"].endswith("[Page 3].")

    assert len(model.prompts) == 1
    assert "What is machine learning?" in model.prompts[0]
    assert "[Page 3, chunk 1" in model.prompts[0]
    assert weak_evidence["text"] not in model.prompts[0]


def test_answer_returns_fallback_without_relevant_evidence() -> None:
    """The model should not run when retrieval confidence is too low."""
    retriever = FakeRetriever(
        [
            make_evidence(score=0.1),
        ]
    )
    model = FakeChatModel("This response should not be used.")

    service = RAGService(
        retriever=retriever,
        model=model,
        minimum_score=0.25,
    )

    result = service.answer("What is machine learning?")

    assert result == {
        "answer": DEFAULT_FALLBACK_ANSWER,
        "evidence": [],
    }
    assert model.prompts == []


@pytest.mark.parametrize(
    "question",
    [
        "",
        "   ",
    ],
)
def test_answer_rejects_empty_question(question: str) -> None:
    """Questions must contain visible text."""
    service = RAGService(
        retriever=FakeRetriever([]),
        model=FakeChatModel("Answer"),
    )

    with pytest.raises(ValueError, match="cannot be empty"):
        service.answer(question)


@pytest.mark.parametrize(
    "top_k",
    [
        0,
        -1,
    ],
)
def test_service_rejects_invalid_top_k(top_k: int) -> None:
    """The retrieval count must be positive."""
    with pytest.raises(ValueError, match="greater than zero"):
        RAGService(
            retriever=FakeRetriever([]),
            model=FakeChatModel("Answer"),
            top_k=top_k,
        )


@pytest.mark.parametrize(
    "minimum_score",
    [
        -1.1,
        1.1,
    ],
)
def test_service_rejects_invalid_minimum_score(
    minimum_score: float,
) -> None:
    """Cosine similarity thresholds must remain in their valid range."""
    with pytest.raises(ValueError, match="between"):
        RAGService(
            retriever=FakeRetriever([]),
            model=FakeChatModel("Answer"),
            minimum_score=minimum_score,
        )


@pytest.mark.parametrize(
    "response_text",
    [
        "",
        "   ",
    ],
)
def test_answer_rejects_empty_model_response(
    response_text: str,
) -> None:
    """An empty AI response should be reported as an error."""
    service = RAGService(
        retriever=FakeRetriever([make_evidence()]),
        model=FakeChatModel(response_text),
    )

    with pytest.raises(ValueError, match="empty response"):
        service.answer("What is machine learning?")


def test_prompt_requires_evidence() -> None:
    """A grounded prompt cannot be created without PDF evidence."""
    with pytest.raises(ValueError, match="At least one"):
        build_grounded_prompt(
            question="What is machine learning?",
            evidence=[],
        )


@pytest.mark.parametrize(
    "api_key",
    [
        "",
        "   ",
    ],
)
def test_create_groq_model_rejects_empty_key(api_key: str) -> None:
    """The application should fail clearly when its API key is missing."""
    with pytest.raises(ValueError, match="cannot be empty"):
        create_groq_model(api_key)