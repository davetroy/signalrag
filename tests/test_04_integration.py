"""Phase 0 Test: End-to-end integration - read Signal messages, embed them, search.

Requires a real Signal Desktop installation on macOS.
"""

import pytest

pytestmark = pytest.mark.requires_signal


def test_integration():
    """Full pipeline: extract key -> fetch messages -> embed -> search."""
    from signalrag.db import SignalDB, get_all_messages_with_body
    from signalrag.embeddings.embedder import Embedder
    from signalrag.embeddings.chunker import chunk_messages
    from signalrag.embeddings.store import MessageStore
    from pathlib import Path
    import shutil

    store_path = Path("/tmp/signalrag_integration_test")
    if store_path.exists():
        shutil.rmtree(store_path)

    try:
        with SignalDB() as db:
            messages = get_all_messages_with_body(db, min_length=10, limit=200)

        assert len(messages) > 0

        emb = Embedder()
        chunks = chunk_messages(messages)
        vectors = emb.encode([c.text for c in chunks])

        store = MessageStore(path=store_path)
        count = store.create_or_replace(chunks, vectors)
        assert count > 0

        qvec = emb.encode_query("meeting schedule")
        results = store.search(qvec, limit=3)
        assert len(results) > 0
    finally:
        if store_path.exists():
            shutil.rmtree(store_path)
