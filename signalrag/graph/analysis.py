"""Graph analysis functions for Signal communication patterns."""

from collections import defaultdict

import networkx as nx


def top_contacts(G: nx.Graph, n: int = 20) -> list[dict]:
    """Top contacts by total message volume (degree weighted by edge weight)."""
    contacts = []
    for node, data in G.nodes(data=True):
        if data.get("type") != "contact":
            continue
        total_weight = sum(
            edata.get("weight", 1)
            for _, _, edata in G.edges(node, data=True)
        )
        contacts.append({
            "id": node,
            "name": data.get("label", node[:8]),
            "total_messages": total_weight,
            "connections": G.degree(node),
        })
    contacts.sort(key=lambda c: c["total_messages"], reverse=True)
    return contacts[:n]


def bridging_contacts(G: nx.Graph, n: int = 20) -> list[dict]:
    """Contacts with highest betweenness centrality (bridge different clusters)."""
    bc = nx.betweenness_centrality(G, weight="weight")
    results = []
    for node, score in sorted(bc.items(), key=lambda x: -x[1]):
        data = G.nodes[node]
        if data.get("type") not in ("contact", "self"):
            continue
        results.append({
            "id": node,
            "name": data.get("label", node[:8]),
            "betweenness": round(score, 4),
            "connections": G.degree(node),
        })
        if len(results) >= n:
            break
    return results


def detect_communities(G: nx.Graph) -> list[dict]:
    """Detect communities using the Louvain algorithm.

    Returns list of communities with members and metadata.
    """
    import community as community_louvain

    # Build a contact-only graph excluding owner and group nodes.
    # The owner is connected to nearly everyone, which collapses communities
    # into one cluster. Removing the owner lets Louvain find structure among
    # contacts based on group co-membership.
    contact_graph = nx.Graph()

    # First, build edges from group co-membership (contacts in same groups)
    group_members = defaultdict(set)
    for u, v, data in G.edges(data=True):
        u_data = G.nodes[u]
        v_data = G.nodes[v]
        if u_data.get("type") == "group":
            group_members[u].add(v)
        elif v_data.get("type") == "group":
            group_members[v].add(u)

    # Create edges between co-members of groups
    for group_node, members in group_members.items():
        member_list = [m for m in members if G.nodes[m].get("type") != "self"]
        for i, m1 in enumerate(member_list):
            for m2 in member_list[i + 1:]:
                if contact_graph.has_edge(m1, m2):
                    contact_graph[m1][m2]["weight"] += 1
                else:
                    contact_graph.add_edge(m1, m2, weight=1)

    # Also add direct private conversation edges (excluding owner)
    for u, v, data in G.edges(data=True):
        u_type = G.nodes[u].get("type", "")
        v_type = G.nodes[v].get("type", "")
        if u_type == "self" or v_type == "self":
            continue
        if u_type == "group" or v_type == "group":
            continue
        if contact_graph.has_edge(u, v):
            contact_graph[u][v]["weight"] += data.get("weight", 1)
        else:
            contact_graph.add_edge(u, v, weight=data.get("weight", 1))

    # Copy node attributes
    for n in contact_graph.nodes():
        if G.has_node(n):
            contact_graph.nodes[n].update(G.nodes[n])

    if len(contact_graph.nodes) < 3:
        return []

    partition = community_louvain.best_partition(contact_graph, weight="weight")

    # Group by community
    communities_map = defaultdict(list)
    for node, comm_id in partition.items():
        data = contact_graph.nodes[node]
        communities_map[comm_id].append({
            "id": node,
            "name": data.get("label", node[:8]),
            "type": data.get("type", "unknown"),
        })

    # Build result sorted by community size
    communities = []
    for comm_id, members in sorted(communities_map.items(), key=lambda x: -len(x[1])):
        communities.append({
            "community_id": comm_id,
            "size": len(members),
            "members": members,
        })

    return communities


def conversation_stats(G: nx.Graph) -> dict:
    """High-level statistics about the communication graph."""
    node_types = defaultdict(int)
    for _, data in G.nodes(data=True):
        node_types[data.get("type", "unknown")] += 1

    total_messages = sum(
        data.get("weight", 0) for _, _, data in G.edges(data=True)
    )

    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "contacts": node_types.get("contact", 0),
        "groups": node_types.get("group", 0),
        "total_message_weight": total_messages,
        "density": round(nx.density(G), 6),
        "components": nx.number_connected_components(G),
    }
