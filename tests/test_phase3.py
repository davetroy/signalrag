"""Phase 3 Tests: RAG query engine.

Tests that require a vector index use the mock data.
Tests that require an LLM API key are skipped if unavailable.
Tests that require a real Signal DB are marked with requires_signal.
"""

import os

import pytest

from signalrag.embeddings.embedder import Embedder
from signalrag.embeddings.chunker import chunk_messages, chunk_conversation_windows
from signalrag.embeddings.store import MessageStore
from signalrag.rag.retriever import Retriever


@pytest.fixture
def indexed_store(sample_messages, tmp_store_path):
    """Create a vector store with indexed sample messages."""
    store = MessageStore(path=tmp_store_path)
    emb = Embedder()

    single = chunk_messages(sample_messages)
    windows = chunk_conversation_windows(sample_messages, window_size=4, stride=2)
    all_chunks = single + windows
    vectors = emb.encode([c.text for c in all_chunks])
    store.create_or_replace(all_chunks, vectors)
    return store


def test_basic_search(indexed_store):
    retriever = Retriever(store=indexed_store)
    results = retriever.search("meeting schedule", limit=5)
    assert len(results) > 0


def test_filtered_search(indexed_store):
    retriever = Retriever(store=indexed_store)

    results = retriever.search("meeting", limit=3)
    assert len(results) > 0

    conv_id = results[0]["conversation_id"]
    filtered = retriever.search("meeting", limit=5, conversation_id=conv_id)
    assert all(r["conversation_id"] == conv_id for r in filtered)


def test_format_for_llm(indexed_store):
    retriever = Retriever(store=indexed_store)
    results = retriever.search("meeting schedule", limit=3)
    formatted = retriever.format_results_for_llm(results)
    assert "[Result 1]" in formatted
    assert "Conversation:" in formatted


@pytest.mark.requires_signal
def test_rag_ask_with_real_db():
    """Full RAG with LLM — requires real Signal DB and API key."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    from signalrag.rag.engine import RAGEngine

    engine = RAGEngine(llm_provider="anthropic")
    result = engine.ask("What have people discussed recently?", limit=5)
    assert "answer" in result
    assert len(result["answer"]) > 0
