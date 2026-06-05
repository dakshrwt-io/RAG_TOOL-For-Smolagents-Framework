"""Backward compatibility wrapper for modular RAG Tool.

New code should import from the modular package:
    from rag_tool import upload_doc, retrieve
"""

from rag_tool import upload_doc, retrieve, Result

__all__ = ["upload_doc", "retrieve", "Result"]


