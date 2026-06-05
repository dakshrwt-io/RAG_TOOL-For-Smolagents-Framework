"""Storage and retrieval operations using Chroma."""

from pathlib import Path
from typing import List
import uuid
from chromadb import PersistentClient
from .models import Result


def store_embeddings(
    chunks: List[Result],
    db_path: Path,
    collection_name: str,
    vectors: List[list[float]],
    reset_collection: bool = True,
) -> int:
    """Store chunks and their embeddings in Chroma database."""
    client = PersistentClient(path=str(db_path))
    if reset_collection and collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(collection_name)

    collection = client.get_or_create_collection(collection_name)

    texts = [chunk.page_content for chunk in chunks]
    ids = [str(uuid.uuid4()) for _ in chunks]
    metas = [chunk.metadata for chunk in chunks]

    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
    return collection.count()


def query_collection(
    query_embedding: List[float],
    db_path: Path,
    collection_name: str,
    retrieval_k: int = 8,
) -> tuple[List[str], List[dict]]:
    """Query Chroma collection with embedding and return documents and metadatas."""
    client = PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(collection_name)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=retrieval_k,
        include=["documents", "metadatas"],
    )

    return results["documents"][0], results["metadatas"][0]
