import re
from dataclasses import dataclass

import pytest

from src.rag_service import is_summary
from src.summarizer import (
    MAX_BATCHES,
    MAX_CHARS_PER_CHUNK,
    MAX_CHUNKS_PER_BATCH,
    SUMMARY_LOCATION,
    SUMMARY_PREFIX,
    SummaryError,
    build_batch_prompt,
    select_chunks,
    summarize_document,
    summary_chunk,
)


@dataclass
class FakeResponse:
    content: str


class ScriptedModel:
    """Return prepared replies and record every prompt."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> FakeResponse:
        self.prompts.append(prompt)
        return FakeResponse(
            self.replies[min(len(self.prompts) - 1, len(self.replies) - 1)]
        )


def make_chunks(count: int, text: str = "Line") -> list[dict]:
    return [
        {
            "text": f"{text} {number}",
            "page": number,
            "chunk": 1,
            "location": f"Page {number}",
        }
        for number in range(1, count + 1)
    ]


def test_short_document_is_one_batch() -> None:
    batches = select_chunks(make_chunks(4))

    assert len(batches) == 1
    assert len(batches[0]) == 4


def test_long_document_is_sampled_across_the_whole_file() -> None:
    chunks = make_chunks(500)

    batches = select_chunks(chunks)
    selected = [chunk for batch in batches for chunk in batch]

    assert len(batches) == MAX_BATCHES
    assert len(selected) == MAX_CHUNKS_PER_BATCH * MAX_BATCHES
    assert selected[0]["page"] == 1
    assert selected[-1]["page"] > 400


def test_select_chunks_rejects_empty_document() -> None:
    with pytest.raises(ValueError, match="At least one document chunk"):
        select_chunks([])


def test_single_batch_uses_one_model_call() -> None:
    model = ScriptedModel(["- A resume for Arun Kumar\n- Skills: Python"])

    summary = summarize_document(make_chunks(3), model)

    assert len(model.prompts) == 1
    assert summary.startswith(SUMMARY_PREFIX)
    assert "A resume for Arun Kumar" in summary
    assert "Page 1" in model.prompts[0]


def test_long_document_combines_partial_summaries() -> None:
    model = ScriptedModel(
        [
            "- partial",
            "- partial",
            "- partial",
            "- partial",
            "- partial",
            "- partial",
            "Combined summary",
        ]
    )

    summary = summarize_document(make_chunks(300), model)

    assert len(model.prompts) == MAX_BATCHES + 1
    assert "This is part 1 of 6" in model.prompts[0]
    assert "partial_summaries" in model.prompts[-1]
    assert "Combined summary" in summary


def test_personal_data_is_masked_for_the_model_and_restored_in_the_summary() -> None:
    chunks = [{"text": "Resume of Arun Kumar, arun@example.com", "page": 1, "chunk": 1}]
    model = ScriptedModel(["- Resume of <PERSON_1> (<EMAIL_1>)"])

    summary = summarize_document(
        chunks,
        model,
        redact_personal_data=True,
        entity_detector=lambda text: (
            [(text.index("Arun Kumar"), text.index("Arun Kumar") + 10, "PERSON")]
            if "Arun Kumar" in text
            else []
        ),
    )

    assert "Arun Kumar" not in model.prompts[0]
    assert "arun@example.com" not in model.prompts[0]
    assert "Resume of Arun Kumar (arun@example.com)" in summary


def test_empty_model_reply_is_rejected() -> None:
    with pytest.raises(SummaryError, match="empty summary"):
        summarize_document(make_chunks(1), ScriptedModel(["   "]))


def test_long_chunks_are_truncated_in_the_prompt() -> None:
    chunks = [{"text": "x" * (MAX_CHARS_PER_CHUNK + 500), "page": 1, "chunk": 1}]
    model = ScriptedModel(["- summary"])

    summarize_document(chunks, model)

    longest_run = max(len(run) for run in re.findall(r"x+", model.prompts[0]))
    assert longest_run == MAX_CHARS_PER_CHUNK


def test_batch_prompt_has_no_part_note_for_single_batch() -> None:
    assert "This is part" not in build_batch_prompt("text", 1, 1)


def test_summary_chunk_is_indexable_and_identifiable() -> None:
    chunk = summary_chunk("A resume for Arun Kumar")

    assert chunk["location"] == SUMMARY_LOCATION
    assert chunk["page"] == 0
    assert is_summary(chunk)
    assert not is_summary({"location": "Page 1"})


def test_summary_chunk_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        summary_chunk("   ")
