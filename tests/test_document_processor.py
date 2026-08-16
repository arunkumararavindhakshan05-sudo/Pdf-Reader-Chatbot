from unittest.mock import MagicMock, patch

import pytest

from src.document_processor import extract_chunks


def make_reader(page_texts: list[str | None]) -> MagicMock:
    """Create a fake PDF reader containing controlled page text."""
    reader = MagicMock()
    reader.pages = []

    for page_text in page_texts:
        page = MagicMock()
        page.extract_text.return_value = page_text
        reader.pages.append(page)

    return reader


def test_extract_chunks_keeps_page_metadata() -> None:
    """Readable chunks should retain their original page numbers."""
    fake_reader = make_reader(
        [
            "Machine learning finds useful patterns in data.",
            "   ",
            "Supervised learning uses labelled training examples.",
        ]
    )

    with patch(
        "src.document_processor.PdfReader",
        return_value=fake_reader,
    ):
        chunks, page_count = extract_chunks(b"fake-pdf-content")

    assert page_count == 3
    assert [chunk["page"] for chunk in chunks] == [1, 3]
    assert all(chunk["text"].strip() for chunk in chunks)
    assert all(chunk["chunk"] == 1 for chunk in chunks)


def test_extract_chunks_rejects_empty_content() -> None:
    """Empty PDF bytes should produce a clear validation error."""
    with pytest.raises(ValueError, match="cannot be empty"):
        extract_chunks(b"")


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap", "message"),
    [
        (0, 0, "greater than zero"),
        (100, -1, "non-negative"),
        (100, 100, "smaller than chunk size"),
        (100, 101, "smaller than chunk size"),
    ],
)
def test_extract_chunks_rejects_invalid_settings(
    chunk_size: int,
    chunk_overlap: int,
    message: str,
) -> None:
    """Invalid chunk configurations should fail before reading a PDF."""
    with pytest.raises(ValueError, match=message):
        extract_chunks(
            b"fake-pdf-content",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
