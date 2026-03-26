"""Signal Desktop database access layer."""

from .key import extract_signal_key
from .connection import SignalDB
from .models import Message, Conversation
from .queries import (
    get_conversations,
    get_messages,
    get_all_messages_with_body,
    get_messages_since,
    count_messages,
    count_conversations,
)

__all__ = [
    "extract_signal_key",
    "SignalDB",
    "Message",
    "Conversation",
    "get_conversations",
    "get_messages",
    "get_all_messages_with_body",
    "get_messages_since",
    "count_messages",
    "count_conversations",
]
