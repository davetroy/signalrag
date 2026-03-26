"""Shared test fixtures for SignalRAG.

Provides a mock Signal database so tests can run without a real
Signal Desktop installation. Tests that require the real database
are marked with @pytest.mark.requires_signal and skipped by default.
"""

import json
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from signalrag.db.models import Message, Conversation


# ---------------------------------------------------------------------------
# pytest markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_signal: test requires a real Signal Desktop installation",
    )


def pytest_collection_modifyitems(config, items):
    skip_signal = pytest.mark.skip(reason="requires Signal Desktop (use --run-signal)")
    for item in items:
        if "requires_signal" in item.keywords:
            if not config.getoption("--run-signal", default=False):
                item.add_marker(skip_signal)


def pytest_addoption(parser):
    parser.addoption(
        "--run-signal",
        action="store_true",
        default=False,
        help="Run tests that require a real Signal Desktop installation",
    )


# ---------------------------------------------------------------------------
# Synthetic message data
# ---------------------------------------------------------------------------

SYNTHETIC_CONVERSATIONS = [
    {
        "id": "conv-alice-001",
        "type": "private",
        "name": None,
        "profileName": "Alice",
        "profileFullName": "Alice Smith",
        "e164": "+15551234567",
        "serviceId": "aaaaaaaa-1111-2222-3333-444444444444",
        "groupId": None,
        "active_at": 1710000000000,
    },
    {
        "id": "conv-bob-002",
        "type": "private",
        "name": None,
        "profileName": "Bob",
        "profileFullName": "Bob Jones",
        "e164": "+15559876543",
        "serviceId": "bbbbbbbb-1111-2222-3333-444444444444",
        "groupId": None,
        "active_at": 1709900000000,
    },
    {
        "id": "conv-group-003",
        "type": "group",
        "name": "Project Team",
        "profileName": None,
        "profileFullName": None,
        "e164": None,
        "serviceId": None,
        "groupId": "group-project-team-xyz",
        "active_at": 1709800000000,
    },
]

SYNTHETIC_MESSAGES = [
    # Alice conversation
    {"body": "Hey, are we still meeting tomorrow at 3pm?", "conversationId": "conv-alice-001", "type": "incoming", "sent_at": 1709000000000, "sourceServiceId": "aaaaaaaa-1111-2222-3333-444444444444", "json": json.dumps({"contactName": "Alice Smith", "profileName": "Alice"})},
    {"body": "Yes, I'll be there. Should I bring the documents?", "conversationId": "conv-alice-001", "type": "outgoing", "sent_at": 1709000060000, "sourceServiceId": None, "json": json.dumps({})},
    {"body": "Please do. Also, have you seen the quarterly report?", "conversationId": "conv-alice-001", "type": "incoming", "sent_at": 1709000120000, "sourceServiceId": "aaaaaaaa-1111-2222-3333-444444444444", "json": json.dumps({"contactName": "Alice Smith"})},
    {"body": "I reviewed it yesterday. The numbers look good.", "conversationId": "conv-alice-001", "type": "outgoing", "sent_at": 1709000180000, "sourceServiceId": None, "json": json.dumps({})},
    {"body": "Great, let's discuss the budget projections too.", "conversationId": "conv-alice-001", "type": "incoming", "sent_at": 1709000240000, "sourceServiceId": "aaaaaaaa-1111-2222-3333-444444444444", "json": json.dumps({"contactName": "Alice Smith"})},
    {"body": "I can prepare a summary slide for the team.", "conversationId": "conv-alice-001", "type": "outgoing", "sent_at": 1709000300000, "sourceServiceId": None, "json": json.dumps({})},
    {"body": "Perfect. See you tomorrow then!", "conversationId": "conv-alice-001", "type": "incoming", "sent_at": 1709000360000, "sourceServiceId": "aaaaaaaa-1111-2222-3333-444444444444", "json": json.dumps({"contactName": "Alice Smith"})},
    {"body": "Looking forward to it. Have a good evening!", "conversationId": "conv-alice-001", "type": "outgoing", "sent_at": 1709000420000, "sourceServiceId": None, "json": json.dumps({})},

    # Bob conversation
    {"body": "Did you catch the flight to New York?", "conversationId": "conv-bob-002", "type": "incoming", "sent_at": 1709100000000, "sourceServiceId": "bbbbbbbb-1111-2222-3333-444444444444", "json": json.dumps({"contactName": "Bob Jones"})},
    {"body": "Yes, landed safely. The hotel is nice.", "conversationId": "conv-bob-002", "type": "outgoing", "sent_at": 1709100060000, "sourceServiceId": None, "json": json.dumps({})},
    {"body": "Have you tried the restaurant on 5th avenue?", "conversationId": "conv-bob-002", "type": "incoming", "sent_at": 1709100120000, "sourceServiceId": "bbbbbbbb-1111-2222-3333-444444444444", "json": json.dumps({"contactName": "Bob Jones"})},
    {"body": "Not yet. Planning to go tonight for dinner.", "conversationId": "conv-bob-002", "type": "outgoing", "sent_at": 1709100180000, "sourceServiceId": None, "json": json.dumps({})},
    {"body": "The conference starts at 9am sharp. Don't be late!", "conversationId": "conv-bob-002", "type": "incoming", "sent_at": 1709100240000, "sourceServiceId": "bbbbbbbb-1111-2222-3333-444444444444", "json": json.dumps({"contactName": "Bob Jones"})},
    {"body": "I'll set an alarm. Thanks for the reminder.", "conversationId": "conv-bob-002", "type": "outgoing", "sent_at": 1709100300000, "sourceServiceId": None, "json": json.dumps({})},

    # Group conversation
    {"body": "Team, the deadline for the proposal is next Friday.", "conversationId": "conv-group-003", "type": "incoming", "sent_at": 1709200000000, "sourceServiceId": "aaaaaaaa-1111-2222-3333-444444444444", "json": json.dumps({"contactName": "Alice Smith"})},
    {"body": "I can handle the technical section.", "conversationId": "conv-group-003", "type": "outgoing", "sent_at": 1709200060000, "sourceServiceId": None, "json": json.dumps({})},
    {"body": "I'll work on the budget and timeline.", "conversationId": "conv-group-003", "type": "incoming", "sent_at": 1709200120000, "sourceServiceId": "bbbbbbbb-1111-2222-3333-444444444444", "json": json.dumps({"contactName": "Bob Jones"})},
    {"body": "Let's sync up on Wednesday to review progress.", "conversationId": "conv-group-003", "type": "incoming", "sent_at": 1709200180000, "sourceServiceId": "aaaaaaaa-1111-2222-3333-444444444444", "json": json.dumps({"contactName": "Alice Smith"})},
    {"body": "Sounds good. I'll have a draft ready by then.", "conversationId": "conv-group-003", "type": "outgoing", "sent_at": 1709200240000, "sourceServiceId": None, "json": json.dumps({})},
    {"body": "Don't forget to include the risk assessment section.", "conversationId": "conv-group-003", "type": "incoming", "sent_at": 1709200300000, "sourceServiceId": "bbbbbbbb-1111-2222-3333-444444444444", "json": json.dumps({"contactName": "Bob Jones"})},
    {"body": "Good point. I'll add it to my section.", "conversationId": "conv-group-003", "type": "outgoing", "sent_at": 1709200360000, "sourceServiceId": None, "json": json.dumps({})},
    {"body": "Also, the client wants a demo. Can we prepare one?", "conversationId": "conv-group-003", "type": "incoming", "sent_at": 1709200420000, "sourceServiceId": "aaaaaaaa-1111-2222-3333-444444444444", "json": json.dumps({"contactName": "Alice Smith"})},
    {"body": "I can put together a prototype by Thursday.", "conversationId": "conv-group-003", "type": "outgoing", "sent_at": 1709200480000, "sourceServiceId": None, "json": json.dumps({})},
    {"body": "Excellent teamwork everyone!", "conversationId": "conv-group-003", "type": "incoming", "sent_at": 1709200540000, "sourceServiceId": "aaaaaaaa-1111-2222-3333-444444444444", "json": json.dumps({"contactName": "Alice Smith"})},
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_db_path(tmp_path):
    """Create a mock Signal database with synthetic data.

    Returns the path to the temporary SQLite database (unencrypted).
    """
    db_path = tmp_path / "mock_signal.sqlite"
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()

    # Create tables matching Signal Desktop schema
    c.execute("""
        CREATE TABLE conversations (
            id TEXT PRIMARY KEY,
            type TEXT,
            name TEXT,
            profileName TEXT,
            profileFullName TEXT,
            e164 TEXT,
            serviceId TEXT,
            groupId TEXT,
            active_at INTEGER
        )
    """)

    c.execute("""
        CREATE TABLE messages (
            rowid INTEGER PRIMARY KEY AUTOINCREMENT,
            body TEXT,
            conversationId TEXT,
            type TEXT,
            sent_at INTEGER,
            sourceServiceId TEXT,
            hasAttachments INTEGER DEFAULT 0,
            expireTimer INTEGER,
            json TEXT
        )
    """)

    # Insert synthetic conversations
    for conv in SYNTHETIC_CONVERSATIONS:
        c.execute(
            "INSERT INTO conversations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (conv["id"], conv["type"], conv["name"], conv["profileName"],
             conv["profileFullName"], conv["e164"], conv["serviceId"],
             conv["groupId"], conv["active_at"]),
        )

    # Insert synthetic messages
    for msg in SYNTHETIC_MESSAGES:
        c.execute(
            "INSERT INTO messages (body, conversationId, type, sent_at, sourceServiceId, json) VALUES (?, ?, ?, ?, ?, ?)",
            (msg["body"], msg["conversationId"], msg["type"], msg["sent_at"],
             msg["sourceServiceId"], msg["json"]),
        )

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def mock_signal_db(mock_db_path):
    """Provide a SignalDB-like connection to the mock database.

    Patches key extraction and uses a plain SQLite connection
    (no SQLCipher needed for the mock).
    """
    conn = sqlite3.connect(str(mock_db_path))

    class MockSignalDB:
        def __init__(self):
            self._conn = conn

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        @property
        def conn(self):
            return self._conn

        def cursor(self):
            return self._conn.cursor()

        def execute(self, sql, params=None):
            c = self.cursor()
            if params:
                c.execute(sql, params)
            else:
                c.execute(sql)
            return c

        def fetchall(self, sql, params=None):
            return self.execute(sql, params).fetchall()

        def fetchone(self, sql, params=None):
            return self.execute(sql, params).fetchone()

    db = MockSignalDB()
    yield db
    conn.close()


@pytest.fixture
def sample_messages():
    """Return a list of Message objects built from synthetic data."""
    messages = []
    for i, msg in enumerate(SYNTHETIC_MESSAGES):
        json_data = json.loads(msg["json"]) if msg["json"] else {}
        sender_name = None
        if msg["type"] == "incoming" and json_data:
            sender_name = json_data.get("contactName") or json_data.get("profileName")

        # Find conversation name
        conv_name = ""
        for conv in SYNTHETIC_CONVERSATIONS:
            if conv["id"] == msg["conversationId"]:
                conv_name = conv["name"] or conv["profileFullName"] or conv["profileName"] or ""
                break

        messages.append(Message(
            id=i + 1,
            body=msg["body"],
            conversation_id=msg["conversationId"],
            type=msg["type"],
            sent_at=msg["sent_at"],
            source_service_id=msg["sourceServiceId"],
            conversation_name=conv_name,
            sender_name=sender_name,
            json_data=json_data,
        ))
    return messages


@pytest.fixture
def tmp_store_path(tmp_path):
    """Return a temporary directory for vector store tests."""
    store_path = tmp_path / "vectorstore"
    yield store_path
    if store_path.exists():
        shutil.rmtree(store_path)
