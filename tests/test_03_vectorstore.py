"""Phase 0 Test: LanceDB vector store round-trip.

No Signal DB required — uses random vectors.
"""

import shutil

import lancedb
import numpy as np


def test_vectorstore(tmp_path):
    test_dir = tmp_path / "vectorstore"
    db = lancedb.connect(str(test_dir))

    data = [
        {
            "id": 1,
            "text": "Meeting tomorrow at 3pm in the conference room",
            "vector": np.random.rand(384).tolist(),
            "conversation_id": "conv-001",
            "sender": "Alice",
            "timestamp": 1700000000000,
        },
        {
            "id": 2,
            "text": "Can you review the document I sent?",
            "vector": np.random.rand(384).tolist(),
            "conversation_id": "conv-001",
            "sender": "Bob",
            "timestamp": 1700000060000,
        },
        {
            "id": 3,
            "text": "The election results are coming in",
            "vector": np.random.rand(384).tolist(),
            "conversation_id": "conv-002",
            "sender": "Charlie",
            "timestamp": 1700000120000,
        },
    ]

    table = db.create_table("messages", data, mode="overwrite")
    assert table.count_rows() == 3

    # Vector search
    query_vec = np.random.rand(384).tolist()
    results = table.search(query_vec).limit(2).to_list()
    assert len(results) == 2

    # Metadata filter
    results = (
        table.search(query_vec)
        .where("conversation_id = 'conv-001'")
        .limit(10)
        .to_list()
    )
    assert len(results) == 2

    # Insertion
    table.add([{
        "id": 4,
        "text": "New message added later",
        "vector": np.random.rand(384).tolist(),
        "conversation_id": "conv-002",
        "sender": "Dave",
        "timestamp": 1700000180000,
    }])
    assert table.count_rows() == 4
