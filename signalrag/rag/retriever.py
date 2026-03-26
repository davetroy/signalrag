"""Retrieval layer: vector search + context expansion."""

from datetime import datetime, timezone

from ..db.connection import SignalDB
from ..db.queries import get_messages
from ..embeddings.embedder import Embedder
from ..embeddings.store import MessageStore


class Retriever:
    """Retrieves relevant message chunks with optional context expansion."""

    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        store: MessageStore | None = None,
    ):
        self.embedder = embedder or Embedder()
        self.store = store or MessageStore()

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        conversation_id: str | None = None,
        chunk_type: str | None = None,
        since: int | None = None,
        literal: bool = False,
    ) -> list[dict]:
        """Search over indexed messages (semantic or literal)."""
        if literal:
            return self.store.literal_search(
                query,
                limit=limit,
                conversation_id=conversation_id,
                chunk_type=chunk_type,
                since=since,
            )
        query_vec = self.embedder.encode_query(query)
        return self.store.search(
            query_vec,
            limit=limit,
            conversation_id=conversation_id,
            chunk_type=chunk_type,
            since=since,
        )

    def search_with_context(
        self,
        query: str,
        *,
        limit: int = 10,
        context_messages: int = 5,
        conversation_id: str | None = None,
        since: int | None = None,
    ) -> list[dict]:
        """Search and expand each result with surrounding conversation context.

        For single-message hits, fetches neighboring messages from the DB
        to provide conversational context. Window chunks already have context.
        """
        results = self.search(
            query,
            limit=limit,
            conversation_id=conversation_id,
            since=since,
        )

        if context_messages <= 0:
            return results

        # Expand single chunks with surrounding messages
        expanded = []
        with SignalDB() as db:
            for r in results:
                if r["chunk_type"] == "window":
                    # Windows already have context
                    expanded.append(r)
                    continue

                # Fetch surrounding messages for single chunks
                conv_id = r["conversation_id"]
                all_msgs = get_messages(db, conv_id)
                if not all_msgs:
                    expanded.append(r)
                    continue

                # Find the message's position and extract context window
                target_ids = set(r["message_ids"])
                target_idx = None
                for i, m in enumerate(all_msgs):
                    if m.id in target_ids:
                        target_idx = i
                        break

                if target_idx is None:
                    expanded.append(r)
                    continue

                start = max(0, target_idx - context_messages)
                end = min(len(all_msgs), target_idx + context_messages + 1)
                context_window = all_msgs[start:end]

                context_lines = []
                for m in context_window:
                    if not m.body:
                        continue
                    sender = m.sender_name or ("me" if m.is_outgoing else "unknown")
                    ts = m.sent_date.strftime("%Y-%m-%d %H:%M")
                    marker = " >>>" if m.id in target_ids else "    "
                    context_lines.append(f"{marker} [{ts}] {sender}: {m.body.strip()}")

                r["context"] = "\n".join(context_lines)
                expanded.append(r)

        return expanded

    def format_results_for_llm(self, results: list[dict]) -> str:
        """Format retrieved results into a text block for LLM context."""
        sections = []
        for i, r in enumerate(results, 1):
            ts_start = datetime.fromtimestamp(
                r["timestamp_start"] / 1000, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M")

            header = (
                f"[Result {i}] "
                f"Conversation: {r['conversation_name']} | "
                f"Date: {ts_start} | "
                f"Type: {r['chunk_type']}"
            )

            if "context" in r:
                body = r["context"]
            else:
                body = r["text"]

            sections.append(f"{header}\n{body}")

        return "\n\n---\n\n".join(sections)
