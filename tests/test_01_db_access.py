"""Phase 0 Test: Open and query the encrypted Signal database.

Requires a real Signal Desktop installation on macOS.
"""

import pytest

pytestmark = pytest.mark.requires_signal


def test_db_access():
    from signalrag.db import SignalDB, count_messages, count_conversations

    with SignalDB() as db:
        conv_count = count_conversations(db)
        msg_count = count_messages(db)

    assert conv_count > 0, "No conversations found"
    assert msg_count > 0, "No messages found"
