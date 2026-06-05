"""Pydantic models for RAG Tool."""

from pydantic import BaseModel, Field
from typing import List


class Result(BaseModel):
    """Result object containing page content and metadata."""
    page_content: str
    metadata: dict


class RankOrder(BaseModel):
    """Model for LLM reranking response."""
    order: List[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )
