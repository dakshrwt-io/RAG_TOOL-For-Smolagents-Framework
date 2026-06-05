"""RAG Tool - Modular Retrieval-Augmented Generation with Chroma and LLM Reranking.

Main API:
    from rag_tool import upload_doc, retrieve
    
    # Upload and embed documents
    upload_doc(["path/to/docs"])
    
    # Retrieve and rerank results
    results = retrieve("What is AI?")
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Union

from dotenv import load_dotenv
from openai import OpenAI

from .config import (
    DEFAULT_DB_DIR,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANK_MODEL,
    DEFAULT_AVERAGE_CHUNK_SIZE,
)
from .models import Result
from .documents import load_documents
from .chunking import create_chunks_local
from .embeddings import get_embeddings, get_query_embedding
from .storage import store_embeddings, query_collection
from .reranking import rerank as rerank_chunks


load_dotenv()


def upload_doc(
    input_paths: Iterable[Union[str, Path]],
    db_path: Optional[Union[str, Path]] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    average_chunk_size: int = DEFAULT_AVERAGE_CHUNK_SIZE,
    chunk_overlap: Optional[int] = None,
    reset_collection: bool = True,
    batch_size: int = 64,
    allowed_exts: Optional[set[str]] = None,
    openai_client: Optional[OpenAI] = None,
) -> dict:
    """Load docs, create local chunks, embed, and store in Chroma.
    
    Args:
        input_paths: File or directory paths to load
        db_path: Path to Chroma DB directory
        collection_name: Name of Chroma collection
        embedding_model: OpenAI embedding model
        average_chunk_size: Target chunk size in characters
        chunk_overlap: Character overlap between chunks (defaults to 25% of chunk_size)
        reset_collection: Delete collection if it exists
        batch_size: Batch size for embedding API calls
        allowed_exts: File extensions to load (default: .md, .txt)
        openai_client: OpenAI client instance
        
    Returns:
        Dictionary with document/chunk counts and storage info
    """
    db_path = Path(db_path) if db_path else DEFAULT_DB_DIR
    db_path.mkdir(parents=True, exist_ok=True)

    if allowed_exts is None:
        allowed_exts = {".md", ".txt"}

    documents = load_documents(input_paths, allowed_exts)
    overlap = chunk_overlap if chunk_overlap is not None else max(1, average_chunk_size // 4)
    chunks = create_chunks_local(documents, average_chunk_size, overlap)

    client = openai_client or OpenAI()
    texts = [chunk.page_content for chunk in chunks]
    vectors = get_embeddings(texts, embedding_model, client, batch_size)

    count = store_embeddings(chunks, db_path, collection_name, vectors, reset_collection)

    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "collection_name": collection_name,
        "db_path": str(db_path),
        "stored": count,
    }


def retrieve(
    query: str,
    db_path: Optional[Union[str, Path]] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    retrieval_k: int = 8,
    rerank_model: str = DEFAULT_RERANK_MODEL,
    rerank: bool = True,
    openai_client: Optional[OpenAI] = None,
) -> List[Result]:
    """Retrieve chunks from Chroma, then optionally rerank with an LLM.
    
    Args:
        query: Query string
        db_path: Path to Chroma DB directory
        collection_name: Name of Chroma collection
        embedding_model: OpenAI embedding model
        retrieval_k: Number of chunks to retrieve
        rerank_model: LLM model for reranking
        rerank: Whether to rerank results
        openai_client: OpenAI client instance
        
    Returns:
        List of Result objects with page content and metadata
    """
    db_path = Path(db_path) if db_path else DEFAULT_DB_DIR
    client = openai_client or OpenAI()
    
    query_embedding = get_query_embedding(query, embedding_model, client)
    documents, metadatas = query_collection(query_embedding, db_path, collection_name, retrieval_k)

    chunks = [
        Result(page_content=doc, metadata=meta)
        for doc, meta in zip(documents, metadatas)
    ]

    if rerank:
        return rerank_chunks(query, chunks, rerank_model)

    return chunks


__all__ = ["upload_doc", "retrieve", "Result"]
