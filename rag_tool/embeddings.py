"""Embedding operations using OpenAI."""

from typing import List
from openai import OpenAI


def get_embeddings(texts: List[str], model: str, client: OpenAI, batch_size: int = 64) -> List[list[float]]:
    """Get embeddings for a list of texts using OpenAI API in batches."""
    vectors: List[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        embeddings = client.embeddings.create(model=model, input=batch).data
        vectors.extend([item.embedding for item in embeddings])
    return vectors


def get_query_embedding(query: str, model: str, client: OpenAI) -> List[float]:
    """Get embedding for a single query string."""
    embedding = client.embeddings.create(model=model, input=[query]).data[0].embedding
    return embedding
