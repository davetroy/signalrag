"""Read-only query functions for the Signal database."""

import json

from .connection import SignalDB
from .models import Conversation, Message


def get_conversations(db: SignalDB, *, active_only: bool = True) -> list[Conversation]:
    """Get all conversations, ordered by most recently active."""
    where = "WHERE active_at IS NOT NULL" if active_only else ""
    rows = db.fetchall(f"""
        SELECT id, type, name, profileName, profileFullName,
               e164, serviceId, groupId, active_at
        FROM conversations
        {where}
        ORDER BY active_at DESC
    """)
    results = []
    for row in rows:
        conv = Conversation(
            id=row[0],
            type=row[1] or "private",
            name=row[2],
            profile_name=row[3],
            profile_full_name=row[4],
            phone=row[5],
            service_id=row[6],
            group_id=row[7],
            active_at=row[8],
        )
        results.append(conv)
    return results


def get_messages(
    db: SignalDB,
    conversation_id: str,
    *,
    limit: int | None = None,
    offset: int = 0,
) -> list[Message]:
    """Get messages for a specific conversation, ordered by time."""
    sql = """
        SELECT m.rowid, m.body, m.conversationId, m.type, m.sent_at,
               m.sourceServiceId, m.hasAttachments, m.expireTimer, m.json,
               c.name, c.profileName, c.profileFullName
        FROM messages m
        LEFT JOIN conversations c ON m.conversationId = c.id
        WHERE m.conversationId = ?
        ORDER BY m.sent_at ASC
    """
    if limit:
        sql += f" LIMIT {limit} OFFSET {offset}"

    return _rows_to_messages(db.fetchall(sql, (conversation_id,)))


def get_all_messages_with_body(
    db: SignalDB,
    *,
    min_length: int = 1,
    limit: int | None = None,
) -> list[Message]:
    """Get all messages that have body text, ordered by time."""
    sql = """
        SELECT m.rowid, m.body, m.conversationId, m.type, m.sent_at,
               m.sourceServiceId, m.hasAttachments, m.expireTimer, m.json,
               c.name, c.profileName, c.profileFullName
        FROM messages m
        LEFT JOIN conversations c ON m.conversationId = c.id
        WHERE m.body IS NOT NULL AND length(m.body) >= ?
        ORDER BY m.sent_at ASC
    """
    if limit:
        sql += f" LIMIT {limit}"

    return _rows_to_messages(db.fetchall(sql, (min_length,)))


def get_messages_since(
    db: SignalDB,
    since_timestamp: int,
    *,
    min_length: int = 1,
) -> list[Message]:
    """Get messages with body text newer than the given timestamp (ms)."""
    sql = """
        SELECT m.rowid, m.body, m.conversationId, m.type, m.sent_at,
               m.sourceServiceId, m.hasAttachments, m.expireTimer, m.json,
               c.name, c.profileName, c.profileFullName
        FROM messages m
        LEFT JOIN conversations c ON m.conversationId = c.id
        WHERE m.body IS NOT NULL AND length(m.body) >= ?
              AND m.sent_at > ?
        ORDER BY m.sent_at ASC
    """
    return _rows_to_messages(db.fetchall(sql, (min_length, since_timestamp)))


def count_messages(db: SignalDB) -> int:
    return db.fetchone("SELECT count(*) FROM messages")[0]


def count_conversations(db: SignalDB) -> int:
    return db.fetchone("SELECT count(*) FROM conversations WHERE active_at IS NOT NULL")[0]


def _rows_to_messages(rows: list) -> list[Message]:
    messages = []
    for row in rows:
        rowid, body, conv_id, mtype, sent_at, source_sid, has_attach, expire, json_str, conv_name, profile_name, profile_full = row
        # Parse JSON data if present
        json_data = {}
        if json_str:
            try:
                json_data = json.loads(json_str)
            except (json.JSONDecodeError, TypeError):
                pass

        # Determine sender name from JSON data for incoming messages
        sender_name = None
        if mtype == "incoming" and json_data:
            sender_name = json_data.get("contactName") or json_data.get("profileName")

        msg = Message(
            id=rowid,
            body=body or "",
            conversation_id=conv_id or "",
            type=mtype or "",
            sent_at=sent_at or 0,
            source_service_id=source_sid,
            has_attachments=bool(has_attach),
            expire_timer=expire,
            conversation_name=conv_name or profile_full or profile_name or "",
            sender_name=sender_name,
            json_data=json_data,
        )
        messages.append(msg)
    return messages
