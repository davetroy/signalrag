"""LanceDB vector store for indexed Signal messages."""

import json
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa

from ..config import VECTORSTORE_DIR
from .chunker import Chunk

# LanceDB schema for message chunks
SCHEMA = pa.schema([
    pa.field("chunk_id", pa.string()),
    pa.field("text", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), 384)),
    pa.field("conversation_id", pa.string()),
    pa.field("conversation_name", pa.string()),
    pa.field("message_ids", pa.string()),       # JSON array
    pa.field("timestamp_start", pa.int64()),
    pa.field("timestamp_end", pa.int64()),
    pa.field("chunk_type", pa.string()),
    pa.field("sender_names", pa.string()),      # JSON array
    pa.field("message_type", pa.string()),
])

TABLE_NAME = "messages"


class MessageStore:
    """LanceDB-backed vector store for message chunks."""

    def __init__(self, *, path: Path | None = None, dimension: int = 384):
        self._path = path or VECTORSTORE_DIR
        self._path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self._path))
        self._dimension = dimension
        self._table = None

    @property
    def table(self):
        if self._table is None:
            if TABLE_NAME in self._db.table_names():
                self._table = self._db.open_table(TABLE_NAME)
            else:
                raise RuntimeError(
                    "Message index not found. Run 'signalrag index' first."
                )
        return self._table

    def create_or_replace(self, chunks: list[Chunk], vectors: np.ndarray):
        """Create (or replace) the index with the given chunks and vectors."""
        data = self._build_records(chunks, vectors)
        self._table = self._db.create_table(TABLE_NAME, data, schema=SCHEMA, mode="overwrite")
        return self._table.count_rows()

    def add(self, chunks: list[Chunk], vectors: np.ndarray):
        """Add new chunks to the existing index."""
        data = self._build_records(chunks, vectors)
        self.table.add(data)

    def search(
        self,
        query_vector: np.ndarray,
        *,
        limit: int = 10,
        conversation_id: str | None = None,
        chunk_type: str | None = None,
        since: int | None = None,
    ) -> list[dict]:
        """Search for similar chunks. Returns list of result dicts."""
        q = self.table.search(query_vector.tolist()).limit(limit)

        filters = []
        if conversation_id:
            filters.append(f"conversation_id = '{conversation_id}'")
        if chunk_type:
            filters.append(f"chunk_type = '{chunk_type}'")
        if since:
            filters.append(f"timestamp_start >= {since}")
        if filters:
            q = q.where(" AND ".join(filters))

        results = q.to_list()
        # Deserialize JSON fields
        for r in results:
            r["message_ids"] = json.loads(r.get("message_ids", "[]"))
            r["sender_names"] = json.loads(r.get("sender_names", "[]"))
        return results

    def literal_search(
        self,
        text_query: str,
        *,
        limit: int = 10,
        conversation_id: str | None = None,
        chunk_type: str | None = None,
        since: int | None = None,
    ) -> list[dict]:
        """Case-insensitive literal text search over indexed chunks."""
        escaped = text_query.replace("'", "''")
        filters = [f"lower(text) LIKE '%{escaped.lower()}%'"]
        if conversation_id:
            filters.append(f"conversation_id = '{conversation_id}'")
        if chunk_type:
            filters.append(f"chunk_type = '{chunk_type}'")
        if since:
            filters.append(f"timestamp_start >= {since}")

        where_clause = " AND ".join(filters)
        results = (
            self.table.search()
            .where(where_clause)
            .limit(limit)
            .to_list()
        )
        for r in results:
            r["message_ids"] = json.loads(r.get("message_ids", "[]"))
            r["sender_names"] = json.loads(r.get("sender_names", "[]"))
        return results

    def count(self) -> int:
        return self.table.count_rows()

    def exists(self) -> bool:
        return TABLE_NAME in self._db.table_names()

    def _build_records(self, chunks: list[Chunk], vectors: np.ndarray) -> list[dict]:
        records = []
        for i, chunk in enumerate(chunks):
            records.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text[:2000],  # cap length for storage
                "vector": vectors[i].tolist(),
                "conversation_id": chunk.conversation_id,
                "conversation_name": chunk.conversation_name,
                "message_ids": json.dumps(chunk.message_ids),
                "timestamp_start": chunk.timestamp_start,
                "timestamp_end": chunk.timestamp_end,
                "chunk_type": chunk.chunk_type,
                "sender_names": json.dumps(chunk.sender_names),
                "message_type": chunk.message_type,
            })
        return records
