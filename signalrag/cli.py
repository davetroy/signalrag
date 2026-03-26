"""SignalRAG command-line interface."""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


@click.group()
def cli():
    """SignalRAG: Personal intelligence database powered by Signal Desktop."""
    pass


@cli.command()
@click.option("--full", is_flag=True, help="Full re-index (default: incremental)")
@click.option("--min-length", default=5, help="Minimum message length to index")
def index(full, min_length):
    """Build or update the vector index."""
    from .embeddings.indexer import Indexer

    indexer = Indexer()

    if full:
        console.print("[bold]Full index build...[/bold]")
        stats = indexer.full_index(min_length=min_length, progress_fn=lambda m: console.print(f"  {m}"))
    else:
        console.print("[bold]Incremental index update...[/bold]")
        stats = indexer.incremental_index(min_length=min_length, progress_fn=lambda m: console.print(f"  {m}"))

    table = Table(title="Index Stats")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in stats.items():
        table.add_row(k.replace("_", " ").title(), str(v))
    console.print(table)


@cli.command()
@click.argument("query")
@click.option("--limit", "-n", default=10, help="Number of results")
@click.option("--conversation", "-c", default=None, help="Filter by conversation name")
@click.option("--since", "-s", default=None, help="Only messages after date (YYYY-MM-DD)")
@click.option("--type", "chunk_type", type=click.Choice(["single", "window"]), default=None)
@click.option("--literal", "-l", is_flag=True, help="Exact text match instead of semantic search")
def search(query, limit, conversation, since, chunk_type, literal):
    """Semantic search over indexed messages."""
    from .rag.engine import RAGEngine
    from .db import SignalDB, get_conversations

    engine = RAGEngine()
    conv_id = None

    if conversation:
        conv_id = _resolve_conversation(conversation)
        if not conv_id:
            return

    since_ts = _parse_date(since) if since else None

    results = engine.search_only(query, limit=limit, conversation_id=conv_id, since=since_ts, literal=literal)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    for i, r in enumerate(results, 1):
        ts = datetime.fromtimestamp(r["timestamp_start"] / 1000, tz=timezone.utc)
        tag = "W" if r["chunk_type"] == "window" else "S"
        dist = r.get("_distance", 0)

        header = f"[bold cyan]{i}.[/bold cyan] [{tag}] [dim]{ts.strftime('%Y-%m-%d %H:%M')}[/dim] [bold]{r['conversation_name']}[/bold] [dim](d={dist:.3f})[/dim]"
        console.print(header)

        text = r["text"][:300]
        if r["chunk_type"] == "window":
            for line in text.split("\n"):
                console.print(f"    {line}")
        else:
            console.print(f"    {text}")
        console.print()


@cli.command()
@click.argument("query")
@click.option("--limit", "-n", default=10, help="Number of chunks to retrieve")
@click.option("--context", "-x", default=5, help="Surrounding messages per hit")
@click.option("--conversation", "-c", default=None, help="Filter by conversation name")
@click.option("--since", "-s", default=None, help="Only messages after date (YYYY-MM-DD)")
@click.option("--mode", "-m", type=click.Choice(["query", "summary", "timeline"]), default="query")
@click.option("--provider", "-p", default=None, help="LLM provider (ollama, anthropic, openai)")
@click.option("--model", default=None, help="LLM model name")
def ask(query, limit, context, conversation, since, mode, provider, model):
    """Ask a question using RAG (retrieval + LLM)."""
    from .rag.engine import RAGEngine

    kwargs = {}
    if provider:
        kwargs["llm_provider"] = provider
    if model:
        kwargs["llm_model"] = model

    engine = RAGEngine(**kwargs)
    conv_id = None

    if conversation:
        conv_id = _resolve_conversation(conversation)
        if not conv_id:
            return

    since_ts = _parse_date(since) if since else None

    with console.status("[bold green]Thinking..."):
        result = engine.ask(
            query,
            limit=limit,
            context_messages=context,
            conversation_id=conv_id,
            since=since_ts,
            mode=mode,
        )

    console.print(Panel(Markdown(result["answer"]), title="Answer", border_style="green"))

    if result["sources"]:
        table = Table(title="Sources", show_lines=False)
        table.add_column("Conversation", style="cyan")
        table.add_column("Date", style="dim")
        table.add_column("Distance", style="yellow")
        for s in result["sources"][:10]:
            ts = datetime.fromtimestamp(s["timestamp"] / 1000, tz=timezone.utc)
            table.add_row(s["conversation"], ts.strftime("%Y-%m-%d %H:%M"), f"{s['distance']:.3f}")
        console.print(table)


@cli.command()
@click.option("--limit", "-n", default=30, help="Number of conversations to show")
@click.option("--type", "conv_type", type=click.Choice(["private", "group", "all"]), default="all")
def conversations(limit, conv_type):
    """List Signal conversations."""
    from .db import SignalDB, get_conversations, count_messages

    with SignalDB() as db:
        convs = get_conversations(db)

        if conv_type != "all":
            convs = [c for c in convs if c.type == conv_type]

        table = Table(title=f"Conversations ({len(convs)} total)")
        table.add_column("#", style="dim", width=4)
        table.add_column("Type", width=7)
        table.add_column("Name", style="cyan")
        table.add_column("Last Active", style="dim")

        for i, c in enumerate(convs[:limit], 1):
            active = c.active_date.strftime("%Y-%m-%d") if c.active_date else "?"
            table.add_row(str(i), c.type, c.display_name, active)

        console.print(table)

        if len(convs) > limit:
            console.print(f"[dim]Showing {limit} of {len(convs)}. Use --limit to see more.[/dim]")


@cli.command()
@click.option("--top", "-n", default=15, help="Number of top contacts")
def graph(top):
    """Show communication graph analysis."""
    from .db import SignalDB
    from .graph.builder import build_graph
    from .graph.analysis import top_contacts, bridging_contacts, detect_communities, conversation_stats

    with console.status("[bold green]Building graph..."):
        with SignalDB() as db:
            G = build_graph(db, min_messages=5)

    stats = conversation_stats(G)
    console.print(Panel(
        f"Nodes: {stats['total_nodes']} ({stats['contacts']} contacts, {stats['groups']} groups)\n"
        f"Edges: {stats['total_edges']}  |  Message weight: {stats['total_message_weight']}  |  "
        f"Components: {stats['components']}",
        title="Graph Overview",
    ))

    # Top contacts
    tc = top_contacts(G, n=top)
    table = Table(title="Top Contacts (by message volume)")
    table.add_column("#", style="dim", width=4)
    table.add_column("Name", style="cyan")
    table.add_column("Messages", justify="right", style="green")
    table.add_column("Connections", justify="right")
    for i, c in enumerate(tc, 1):
        table.add_row(str(i), c["name"], str(c["total_messages"]), str(c["connections"]))
    console.print(table)

    # Bridging contacts
    bc = bridging_contacts(G, n=10)
    table = Table(title="Bridging Contacts (betweenness centrality)")
    table.add_column("#", style="dim", width=4)
    table.add_column("Name", style="cyan")
    table.add_column("Betweenness", justify="right", style="yellow")
    table.add_column("Connections", justify="right")
    for i, b in enumerate(bc, 1):
        table.add_row(str(i), b["name"], f"{b['betweenness']:.4f}", str(b["connections"]))
    console.print(table)

    # Communities
    comms = detect_communities(G)
    table = Table(title=f"Communities ({len(comms)} detected)")
    table.add_column("ID", style="dim", width=4)
    table.add_column("Size", justify="right", width=6)
    table.add_column("Members", style="cyan")
    for c in comms[:10]:
        names = [m["name"] for m in c["members"][:5]]
        suffix = f" +{c['size'] - 5} more" if c["size"] > 5 else ""
        table.add_row(str(c["community_id"]), str(c["size"]), ", ".join(names) + suffix)
    console.print(table)


@cli.command()
@click.argument("output", type=click.Path())
@click.option("--format", "fmt", type=click.Choice(["parquet", "csv"]), default="parquet")
@click.option("--no-vectors", is_flag=True, help="Exclude embedding vectors")
def export(output, fmt, no_vectors):
    """Export indexed data to Parquet or CSV."""
    from .export import to_parquet, to_csv

    with console.status("[bold green]Exporting..."):
        if fmt == "parquet":
            path = to_parquet(output, include_vectors=not no_vectors)
        else:
            path = to_csv(output)

    size_mb = path.stat().st_size / 1024 ** 2
    console.print(f"Exported to [bold]{path}[/bold] ({size_mb:.1f} MB)")
    if fmt == "parquet" and not no_vectors:
        console.print("[dim]Compatible with: embedding-atlas {path} --text text --vector vector[/dim]")


@cli.command()
def stats():
    """Show database and index statistics."""
    from .db import SignalDB, count_messages, count_conversations
    from .embeddings.store import MessageStore
    import json
    from .config import STATE_FILE

    with SignalDB() as db:
        mc = count_messages(db)
        cc = count_conversations(db)

    table = Table(title="SignalRAG Stats")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Messages in DB", f"{mc:,}")
    table.add_row("Active Conversations", f"{cc:,}")

    store = MessageStore()
    if store.exists():
        table.add_row("Indexed Chunks", f"{store.count():,}")

    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())
        ts = datetime.fromtimestamp(state["updated_at"], tz=timezone.utc)
        table.add_row("Last Indexed", ts.strftime("%Y-%m-%d %H:%M UTC"))
        table.add_row("Embedding Model", state.get("model", "?"))

    console.print(table)


def _resolve_conversation(name_fragment: str) -> str | None:
    """Find a conversation ID by name substring match."""
    from .db import SignalDB, get_conversations

    with SignalDB() as db:
        convs = get_conversations(db)

    matches = [c for c in convs if name_fragment.lower() in c.display_name.lower()]
    if not matches:
        console.print(f"[red]No conversation matching '{name_fragment}'[/red]")
        return None
    if len(matches) > 1:
        console.print(f"[yellow]Multiple matches for '{name_fragment}':[/yellow]")
        for c in matches[:10]:
            console.print(f"  - {c.display_name}")
        console.print("[yellow]Please be more specific.[/yellow]")
        return None
    return matches[0].id


def _parse_date(date_str: str) -> int:
    """Parse YYYY-MM-DD to timestamp in milliseconds."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def main():
    cli()


if __name__ == "__main__":
    main()
