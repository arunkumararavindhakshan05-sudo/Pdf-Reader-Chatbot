from io import BytesIO
from typing import TypedDict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


class DocumentChunk(TypedDict):
    """A searchable section of a PDF with its source location."""

    text: str
    page: int
    chunk: int


def extract_chunks(
    pdf_bytes: bytes,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> tuple[list[DocumentChunk], int]:
    """Extract readable PDF text and split it into page-aware chunks."""
    if not pdf_bytes:
        raise ValueError("PDF content cannot be empty.")

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be non-negative and smaller than chunk size."
        )

    reader = PdfReader(BytesIO(pdf_bytes))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[DocumentChunk] = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        if not page_text or not page_text.strip():
            continue

        page_chunks = splitter.split_text(page_text.strip())

        for chunk_number, chunk_text in enumerate(
            page_chunks,
            start=1,
        ):
            chunks.append(
                {
                    "text": chunk_text,
                    "page": page_number,
                    "chunk": chunk_number,
                }
            )

    return chunks, len(reader.pages)