"""Chunking strategies for Signal messages."""

from dataclasses import dataclass

from ..db.models import Message


@dataclass
class Chunk:
    """A unit of text ready for embedding and storage."""
    chunk_id: str
    text: str
    conversation_id: str
    conversation_name: str
    message_ids: list[int]
    timestamp_start: int  # ms
    timestamp_end: int    # ms
    chunk_type: str  # 'single' or 'window'
    sender_names: list[str]
    message_type: str  # 'incoming', 'outgoing', or 'mixed'


def chunk_messages(messages: list[Message]) -> list[Chunk]:
    """Create one chunk per message (for individual message search)."""
    chunks = []
    for m in messages:
        if not m.body or not m.body.strip():
            continue
        sender = m.sender_name or ("me" if m.is_outgoing else "unknown")
        chunks.append(Chunk(
            chunk_id=f"msg-{m.id}",
            text=m.body.strip(),
            conversation_id=m.conversation_id,
            conversation_name=m.conversation_name or "",
            message_ids=[m.id],
            timestamp_start=m.sent_at,
            timestamp_end=m.sent_at,
            chunk_type="single",
            sender_names=[sender],
            message_type=m.type,
        ))
    return chunks


def chunk_conversation_windows(
    messages: list[Message],
    *,
    window_size: int = 8,
    stride: int = 4,
) -> list[Chunk]:
    """Create sliding-window chunks over consecutive messages in a conversation.

    Groups messages by conversation, then creates overlapping windows of
    `window_size` messages with `stride` step. This captures conversational
    context that individual messages miss.
    """
    # Group by conversation
    by_conv: dict[str, list[Message]] = {}
    for m in messages:
        if not m.body or not m.body.strip():
            continue
        by_conv.setdefault(m.conversation_id, []).append(m)

    chunks = []
    for conv_id, conv_msgs in by_conv.items():
        # Sort by time within conversation
        conv_msgs.sort(key=lambda m: m.sent_at)

        if len(conv_msgs) < 3:
            # Too few messages for a window, skip (singles already cover these)
            continue

        for i in range(0, len(conv_msgs) - window_size + 1, stride):
            window = conv_msgs[i:i + window_size]

            # Build window text with sender attribution
            lines = []
            for m in window:
                sender = m.sender_name or ("me" if m.is_outgoing else "unknown")
                lines.append(f"{sender}: {m.body.strip()}")
            text = "\n".join(lines)

            senders = list({
                m.sender_name or ("me" if m.is_outgoing else "unknown")
                for m in window
            })
            types = {m.type for m in window}
            msg_type = "mixed" if len(types) > 1 else next(iter(types))

            chunks.append(Chunk(
                chunk_id=f"win-{conv_id[:8]}-{i}",
                text=text,
                conversation_id=conv_id,
                conversation_name=window[0].conversation_name or "",
                message_ids=[m.id for m in window],
                timestamp_start=window[0].sent_at,
                timestamp_end=window[-1].sent_at,
                chunk_type="window",
                sender_names=senders,
                message_type=msg_type,
            ))

    return chunks
