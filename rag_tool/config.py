"""Configuration and default constants for RAG Tool."""

from pathlib import Path

DEFAULT_DB_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
DEFAULT_COLLECTION_NAME = "docs"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_RERANK_MODEL = "gpt-4.1-nano"
DEFAULT_AVERAGE_CHUNK_SIZE = 500
