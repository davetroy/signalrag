"""Phase 4 Tests: Graph analysis.

Requires a real Signal Desktop installation for meaningful graph data.
"""

import pytest

pytestmark = pytest.mark.requires_signal


def test_build_graph():
    from signalrag.db import SignalDB
    from signalrag.graph.builder import build_graph
    from signalrag.graph.analysis import conversation_stats

    with SignalDB() as db:
        G = build_graph(db, min_messages=5)

    stats = conversation_stats(G)
    assert stats["total_nodes"] > 0
    assert stats["total_edges"] > 0
    return G


def test_top_contacts():
    from signalrag.db import SignalDB
    from signalrag.graph.builder import build_graph
    from signalrag.graph.analysis import top_contacts

    with SignalDB() as db:
        G = build_graph(db, min_messages=5)

    top = top_contacts(G, n=10)
    assert len(top) > 0


def test_communities():
    from signalrag.db import SignalDB
    from signalrag.graph.builder import build_graph
    from signalrag.graph.analysis import detect_communities

    with SignalDB() as db:
        G = build_graph(db, min_messages=5)

    comms = detect_communities(G)
    assert len(comms) > 0
