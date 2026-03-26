"""Phase 2 Tests: Embedding & indexing pipeline.

Uses mock message data by default.
"""

from signalrag.embeddings.embedder import Embedder
from signalrag.embeddings.chunker import chunk_messages, chunk_conversation_windows, Chunk
from signalrag.embeddings.store import MessageStore


def test_embedder():
    emb = Embedder()
    assert emb.dimension == 384
    vecs = emb.encode(["hello world", "test message"])
    assert vecs.shape == (2, 384)
    # Verify normalization
    norm = (vecs[0] ** 2).sum() ** 0.5
    assert abs(norm - 1.0) < 0.01, f"Expected unit norm, got {norm}"
    q = emb.encode_query("hello")
    assert q.shape == (384,)


def test_chunker_single(sample_messages):
    chunks = chunk_messages(sample_messages)
    assert len(chunks) > 0
    assert all(c.chunk_type == "single" for c in chunks)
    assert all(len(c.message_ids) == 1 for c in chunks)
    assert all(c.text.strip() for c in chunks)


def test_chunker_windows(sample_messages):
    windows = chunk_conversation_windows(sample_messages, window_size=4, stride=2)
    assert len(windows) > 0
    assert all(c.chunk_type == "window" for c in windows)
    assert all(len(c.message_ids) <= 4 for c in windows)
    multi_line = [c for c in windows if "\n" in c.text]
    assert len(multi_line) > 0


def test_store_operations(sample_messages, tmp_store_path):
    store = MessageStore(path=tmp_store_path)
    emb = Embedder()

    chunks = chunk_messages(sample_messages)
    vectors = emb.encode([c.text for c in chunks])

    count = store.create_or_replace(chunks, vectors)
    assert count == len(chunks)
    assert store.exists()

    # Search
    qvec = emb.encode_query("meeting schedule plans")
    results = store.search(qvec, limit=5)
    assert len(results) > 0
    assert all("text" in r for r in results)
    assert all("_distance" in r for r in results)

    # Filtered search
    conv_id = chunks[0].conversation_id
    filtered = store.search(qvec, limit=5, conversation_id=conv_id)
    assert all(r["conversation_id"] == conv_id for r in filtered)
