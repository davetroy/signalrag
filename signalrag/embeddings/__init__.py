"""Embedding and indexing pipeline for Signal messages."""

from .embedder import Embedder
from .chunker import chunk_messages, chunk_conversation_windows
from .store import MessageStore
from .indexer import Indexer

__all__ = [
    "Embedder",
    "chunk_messages",
    "chunk_conversation_windows",
    "MessageStore",
    "Indexer",
]
