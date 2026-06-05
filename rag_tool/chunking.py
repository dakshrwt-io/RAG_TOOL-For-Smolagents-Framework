"""Text chunking and splitting utilities."""

from typing import List
from .models import Result


def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks of fixed size."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    cleaned = text.strip()
    if not cleaned:
        return []

    chunks: List[str] = []
    start = 0
    length = len(cleaned)
    while start < length:
        end = min(length, start + chunk_size)
        chunks.append(cleaned[start:end])
        if end == length:
            break
        start = end - overlap
    return chunks


def create_chunks_local(documents: List[dict], chunk_size: int, overlap: int) -> List[Result]:
    """Create Result chunks from documents using local text splitting."""
    results: List[Result] = []
    for doc in documents:
        metadata = {"source": doc["source"], "type": doc["type"]}
        for piece in split_text(doc["text"], chunk_size, overlap):
            results.append(Result(page_content=piece, metadata=metadata))
    return results
