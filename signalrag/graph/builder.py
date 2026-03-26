"""Build a NetworkX graph from Signal message data."""

from collections import defaultdict

import networkx as nx

from ..db.connection import SignalDB
from ..db.queries import get_conversations, get_all_messages_with_body


def build_graph(
    db: SignalDB,
    *,
    min_messages: int = 3,
    include_groups: bool = True,
) -> nx.Graph:
    """Build a communication graph from Signal conversations.

    Nodes represent contacts/participants.
    Edges represent communication links weighted by message count.

    For private conversations: direct edge between user and contact.
    For group conversations: edges between all co-members (co-occurrence).

    Args:
        db: Open SignalDB connection
        min_messages: Minimum messages for a conversation to be included
        include_groups: Whether to include group conversation edges
    """
    G = nx.Graph()
    convs = get_conversations(db)

    # Identify the DB owner
    owner_row = db.fetchone("SELECT ourServiceId FROM sessions LIMIT 1")
    owner_id = owner_row[0] if owner_row else "me"
    G.add_node(owner_id, label="me", type="self")

    # Get message counts per conversation
    msg_counts = {}
    rows = db.fetchall("""
        SELECT conversationId, count(*) as cnt
        FROM messages
        WHERE body IS NOT NULL AND body != ''
        GROUP BY conversationId
        HAVING cnt >= ?
    """, (min_messages,))
    for conv_id, cnt in rows:
        msg_counts[conv_id] = cnt

    # Get message direction counts per conversation
    direction_counts = {}
    rows = db.fetchall("""
        SELECT conversationId, type, count(*) as cnt
        FROM messages
        WHERE body IS NOT NULL AND body != ''
        GROUP BY conversationId, type
    """)
    for conv_id, msg_type, cnt in rows:
        direction_counts.setdefault(conv_id, {})[msg_type] = cnt

    for conv in convs:
        if conv.id not in msg_counts:
            continue

        total = msg_counts[conv.id]
        dirs = direction_counts.get(conv.id, {})
        incoming = dirs.get("incoming", 0)
        outgoing = dirs.get("outgoing", 0)

        if conv.type == "private":
            node_id = conv.service_id or conv.phone or conv.id
            G.add_node(node_id, label=conv.display_name, type="contact",
                       phone=conv.phone or "")
            G.add_edge(owner_id, node_id,
                       weight=total,
                       incoming=incoming,
                       outgoing=outgoing,
                       conversation_id=conv.id,
                       conversation_name=conv.display_name)

        elif conv.type == "group" and include_groups:
            # Get group members from the conversation JSON
            members = _get_group_members(db, conv)
            if not members:
                continue

            group_node = f"group:{conv.id[:12]}"
            G.add_node(group_node, label=conv.display_name, type="group",
                       member_count=len(members))

            # Connect owner to group
            G.add_edge(owner_id, group_node,
                       weight=outgoing,
                       conversation_id=conv.id,
                       conversation_name=conv.display_name)

            # Connect members to group
            for member_id, member_name in members:
                if member_id == owner_id:
                    continue
                if not G.has_node(member_id):
                    G.add_node(member_id, label=member_name, type="contact")
                G.add_edge(member_id, group_node,
                           weight=1,
                           conversation_id=conv.id,
                           conversation_name=conv.display_name)

    return G


def _get_group_members(db: SignalDB, conv) -> list[tuple[str, str]]:
    """Extract group member IDs and names from conversation JSON."""
    import json

    row = db.fetchone("SELECT json FROM conversations WHERE id = ?", (conv.id,))
    if not row or not row[0]:
        return []

    try:
        data = json.loads(row[0])
    except (json.JSONDecodeError, TypeError):
        return []

    members = []
    for member in data.get("membersV2", []):
        aci = member.get("aci", "")
        if aci:
            members.append((aci, ""))

    # Try to resolve names for member ACIs
    resolved = []
    for aci, _ in members:
        name_row = db.fetchone("""
            SELECT name, profileName, profileFullName, e164
            FROM conversations
            WHERE serviceId = ?
        """, (aci,))
        if name_row:
            name = name_row[0] or name_row[2] or name_row[1] or name_row[3] or aci[:8]
        else:
            name = aci[:8]
        resolved.append((aci, name))

    return resolved
