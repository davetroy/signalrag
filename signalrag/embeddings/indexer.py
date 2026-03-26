"""Orchestrates the full indexing pipeline: DB -> chunks -> embeddings -> vector store."""

import json
import time
from pathlib import Path

from ..config import STATE_FILE, SIGNALRAG_DIR
from ..db.connection import SignalDB
from ..db.queries import get_all_messages_with_body, get_messages_since, count_messages
from .chunker import chunk_messages, chunk_conversation_windows
from .embedder import Embedder
from .store import MessageStore


class Indexer:
    """Builds and updates the vector index from Signal messages."""

    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        store: MessageStore | None = None,
    ):
        self.embedder = embedder or Embedder()
        self.store = store or MessageStore(dimension=self.embedder.dimension)

    def full_index(self, *, min_length: int = 5, progress_fn=None) -> dict:
        """Build the full index from scratch.

        Returns stats dict with counts and timing.
        """
        t0 = time.time()

        # Step 1: Read all messages
        if progress_fn:
            progress_fn("Reading messages from Signal database...")
        with SignalDB() as db:
            total_in_db = count_messages(db)
            messages = get_all_messages_with_body(db, min_length=min_length)

        if progress_fn:
            progress_fn(f"Read {len(messages)} messages with text (of {total_in_db} total)")

        # Step 2: Create chunks
        if progress_fn:
            progress_fn("Creating chunks...")
        single_chunks = chunk_messages(messages)
        window_chunks = chunk_conversation_windows(messages)
        all_chunks = single_chunks + window_chunks

        if progress_fn:
            progress_fn(
                f"Created {len(single_chunks)} single + {len(window_chunks)} window = "
                f"{len(all_chunks)} total chunks"
            )

        # Step 3: Generate embeddings in batches
        if progress_fn:
            progress_fn("Generating embeddings...")
        texts = [c.text for c in all_chunks]
        embeddings = self.embedder.encode(texts, batch_size=256, show_progress=bool(progress_fn))

        # Step 4: Store in vector DB
        if progress_fn:
            progress_fn("Writing to vector store...")
        stored = self.store.create_or_replace(all_chunks, embeddings)

        elapsed = time.time() - t0

        # Save state
        last_ts = max((m.sent_at for m in messages), default=0)
        self._save_state(last_ts, stored)

        stats = {
            "total_messages_in_db": total_in_db,
            "messages_with_text": len(messages),
            "single_chunks": len(single_chunks),
            "window_chunks": len(window_chunks),
            "total_chunks_indexed": stored,
            "embedding_dimension": self.embedder.dimension,
            "elapsed_seconds": round(elapsed, 1),
        }

        if progress_fn:
            progress_fn(f"Done. Indexed {stored} chunks in {elapsed:.1f}s")

        return stats

    def incremental_index(self, *, min_length: int = 5, progress_fn=None) -> dict:
        """Add only new messages since last index.

        Returns stats dict. Falls back to full_index if no prior state.
        """
        state = self._load_state()
        if not state or not self.store.exists():
            if progress_fn:
                progress_fn("No prior index found, doing full index...")
            return self.full_index(min_length=min_length, progress_fn=progress_fn)

        last_ts = state["last_timestamp"]
        t0 = time.time()

        with SignalDB() as db:
            messages = get_messages_since(db, last_ts, min_length=min_length)

        if not messages:
            if progress_fn:
                progress_fn("No new messages since last index.")
            return {"new_messages": 0, "new_chunks": 0, "elapsed_seconds": 0}

        if progress_fn:
            progress_fn(f"Found {len(messages)} new messages")

        single_chunks = chunk_messages(messages)
        window_chunks = chunk_conversation_windows(messages)
        all_chunks = single_chunks + window_chunks

        if all_chunks:
            texts = [c.text for c in all_chunks]
            embeddings = self.embedder.encode(texts, batch_size=256, show_progress=bool(progress_fn))
            self.store.add(all_chunks, embeddings)

        elapsed = time.time() - t0
        new_last_ts = max((m.sent_at for m in messages), default=last_ts)
        self._save_state(new_last_ts, state["total_chunks"] + len(all_chunks))

        stats = {
            "new_messages": len(messages),
            "new_single_chunks": len(single_chunks),
            "new_window_chunks": len(window_chunks),
            "new_chunks": len(all_chunks),
            "total_chunks": state["total_chunks"] + len(all_chunks),
            "elapsed_seconds": round(elapsed, 1),
        }

        if progress_fn:
            progress_fn(f"Added {len(all_chunks)} chunks in {elapsed:.1f}s")

        return stats

    def _save_state(self, last_timestamp: int, total_chunks: int):
        SIGNALRAG_DIR.mkdir(parents=True, exist_ok=True)
        state = {
            "last_timestamp": last_timestamp,
            "total_chunks": total_chunks,
            "model": self.embedder.model_name,
            "updated_at": time.time(),
        }
        STATE_FILE.write_text(json.dumps(state, indent=2))

    def _load_state(self) -> dict | None:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text())
        return None
