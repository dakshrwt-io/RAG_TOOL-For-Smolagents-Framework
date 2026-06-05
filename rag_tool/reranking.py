"""Reranking operations using LLM."""

import json
from typing import List
from pydantic import ValidationError
from litellm import completion
from .models import Result, RankOrder


def rerank(question: str, chunks: List[Result], model: str) -> List[Result]:
    """Rerank chunks by relevance to the question using an LLM."""
    if not chunks:
        return []

    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""

    user_prompt = (
        f"The user has asked the following question:\n\n{question}\n\n"
        "Order all the chunks of text by relevance to the question, from most relevant to least relevant. "
        "Include all the chunk ids you are provided with, reranked.\n\n"
        "Here are the chunks:\n\n"
    )
    for index, chunk in enumerate(chunks, start=1):
        user_prompt += f"# CHUNK ID: {index}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."

    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=RankOrder,
    )
    reply = response.choices[0].message.content

    try:
        order = RankOrder.model_validate_json(reply).order
    except (ValidationError, TypeError, ValueError):
        try:
            parsed = json.loads(reply) if isinstance(reply, str) else reply
            if isinstance(parsed, list):
                parsed = {"order": parsed}
            order = RankOrder.model_validate(parsed).order
        except (ValidationError, TypeError, ValueError, json.JSONDecodeError):
            # Fall back to retrieval order if the model reply is malformed.
            order = list(range(1, len(chunks) + 1))

    valid_order: List[int] = []
    seen = set()
    for idx in order:
        if 1 <= idx <= len(chunks) and idx not in seen:
            valid_order.append(idx)
            seen.add(idx)

    if len(valid_order) != len(chunks):
        for idx in range(1, len(chunks) + 1):
            if idx not in seen:
                valid_order.append(idx)

    return [chunks[i - 1] for i in valid_order]
