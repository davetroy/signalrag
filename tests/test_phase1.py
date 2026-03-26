"""Phase 1 Tests: Database access layer.

Tests use a mock database by default. Run with --run-signal to test
against a real Signal Desktop installation.
"""

from signalrag.db.models import Message, Conversation
from signalrag.db.queries import (
    get_conversations,
    get_messages,
    get_all_messages_with_body,
    get_messages_since,
    count_messages,
    count_conversations,
)


def test_conversations(mock_signal_db):
    convs = get_conversations(mock_signal_db)
    assert len(convs) == 3
    assert all(isinstance(c, Conversation) for c in convs)

    private = [c for c in convs if c.type == "private"]
    groups = [c for c in convs if c.type == "group"]
    assert len(private) == 2
    assert len(groups) == 1

    for c in convs:
        assert c.display_name


def test_messages_for_conversation(mock_signal_db):
    msgs = get_messages(mock_signal_db, "conv-alice-001", limit=20)
    assert len(msgs) > 0
    assert all(isinstance(m, Message) for m in msgs)

    # Verify ordering (ascending by time)
    timestamps = [m.sent_at for m in msgs]
    assert timestamps == sorted(timestamps)


def test_all_messages_with_body(mock_signal_db):
    msgs = get_all_messages_with_body(mock_signal_db, min_length=10)
    assert len(msgs) > 0
    assert all(len(m.body) >= 10 for m in msgs)


def test_messages_since(mock_signal_db):
    # Get messages after the first few Alice messages
    msgs = get_messages_since(mock_signal_db, 1709100000000)
    assert isinstance(msgs, list)
    assert len(msgs) > 0
    assert all(m.sent_at > 1709100000000 for m in msgs)


def test_message_json_data(mock_signal_db):
    msgs = get_all_messages_with_body(mock_signal_db, limit=50)
    with_json = [m for m in msgs if m.json_data]
    assert len(with_json) > 0


def test_counts(mock_signal_db):
    mc = count_messages(mock_signal_db)
    cc = count_conversations(mock_signal_db)
    assert mc > 0
    assert cc > 0
