"""Backwards-compatible re-exports.

The implementation moved into focused modules. This shim keeps the import
paths used by the submitted README and evaluate.py working.
"""

from app.embeddings import build_embeddings  # noqa: F401
from app.engine import AviationRAGEngine, get_engine, reset_engine  # noqa: F401
from app.models import REFUSAL_MESSAGE, Decision, RetrievedChunk  # noqa: F401
from app.retrieval import LexicalIndex, Retriever  # noqa: F401

__all__ = [
    "REFUSAL_MESSAGE",
    "AviationRAGEngine",
    "Decision",
    "LexicalIndex",
    "RetrievedChunk",
    "Retriever",
    "build_embeddings",
    "get_engine",
    "reset_engine",
]
