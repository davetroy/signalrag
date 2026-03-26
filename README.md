# SignalRAG

Turn your Signal Desktop database into a searchable, analyzable intelligence store using vector embeddings, RAG, and graph analysis. Entirely local-first and privacy-preserving.

> **Platform: macOS only.** Key extraction depends on macOS Keychain. Linux/Windows support would require alternative key extraction — PRs welcome.

---

## Security Warning

**Read this before using SignalRAG.**

- This tool **accesses your Signal encryption keys** via the macOS Keychain. Your system will prompt you to authorize access on first run.
- The vector index and any exported data are stored **unencrypted** on disk at `~/.signalrag/`.
- **Only run this on machines you physically control.** Full-disk encryption (FileVault) is strongly recommended.
- **Never commit or share** your `~/.signalrag/` directory, `.parquet` exports, or any output files — they contain your message content.
- This is a **local-only, single-user** research tool. Do not deploy it as a service or expose it over a network.
- The tool opens Signal's database in **read-only** mode via a temporary copy. It never writes to Signal's database.

---

## Prerequisites

- **macOS** with [Signal Desktop](https://signal.org/download/) installed, configured, and linked to your phone
- **Python 3.12+**
- **SQLCipher** (via Homebrew):
  ```bash
  brew install sqlcipher
  ```
- **Optional — Local LLM** (for privacy-preserving RAG queries):
  ```bash
  brew install ollama
  ollama pull llama3.1:8b
  ```
- **Optional — Cloud LLM**: Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your environment for cloud-based RAG queries. NOTE: this exposes your Signal data to external service providers. You are responsible for the security of your data.

## Installation

```bash
git clone https://github.com/davetroy/signalrag.git
cd signalrag
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,llm]"
```

## Quick Start

```bash
# Build the vector index (first run takes a few minutes)
signalrag index --full

# Semantic search across all messages
signalrag search "meeting about the project"

# Ask a question using RAG (retrieval + LLM synthesis)
signalrag ask "What did Alice say about the deadline?"

# Filter by conversation and date
signalrag search "travel plans" --conversation "Bob" --since 2024-01-01

# Ask with a specific LLM provider
signalrag ask "summarize recent activity" --provider ollama --model llama3.1:8b

# List your conversations
signalrag conversations

# Communication graph analysis (top contacts, bridging nodes, communities)
signalrag graph

# Export indexed data
signalrag export output.parquet
signalrag export output.csv --format csv --no-vectors

# View database and index stats
signalrag stats
```

After the initial full index, use `signalrag index` (without `--full`) for fast incremental updates.

## How It Works

```
Signal Desktop DB (SQLCipher)
         │
         ▼
┌──────────────────┐
│  Key Extraction  │  macOS Keychain → PBKDF2 → AES-CBC decrypt
│   (db/key.py)    │
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Database Layer  │  Read-only SQLCipher connection (temp copy)
│      (db/)       │  → Conversations, Messages, Contacts
└────────┬─────────┘
         ▼
┌──────────────────┐
│    Embeddings    │  sentence-transformers (all-MiniLM-L6-v2)
│  (embeddings/)   │  → Single messages + conversation windows
└────────┬─────────┘
         ▼
┌──────────────────┐
│   Vector Store   │  LanceDB (embedded, zero-config)
│  (embeddings/)   │  → ~/.signalrag/vectorstore/
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌─────────┐
│  RAG   │ │  Graph  │
│ (rag/) │ │ (graph/)│
│        │ │         │
│Retrieve│ │NetworkX │
│ + LLM  │ │analysis │
└────────┘ └─────────┘
```

- **Database layer**: Extracts the SQLCipher key from macOS Keychain, copies Signal's DB to a temp directory (avoids lock contention), opens read-only
- **Embeddings**: Chunks messages individually and as sliding conversation windows (8 messages, stride 4). Embeds with `all-MiniLM-L6-v2` (384 dimensions)
- **Vector store**: LanceDB for fast similarity search with metadata filtering
- **RAG engine**: Retrieves relevant chunks, expands with surrounding context, synthesizes answers via LLM (Ollama, Anthropic, or OpenAI)
- **Graph analysis**: Builds a communication graph with NetworkX. Computes top contacts, betweenness centrality (bridging nodes), and Louvain community detection

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `SIGNALRAG_LLM_PROVIDER` | `ollama` | LLM backend: `ollama`, `anthropic`, or `openai` |
| `SIGNALRAG_LLM_MODEL` | `llama3.1:8b` | Model name for the chosen provider |
| `ANTHROPIC_API_KEY` | — | Required if using `anthropic` provider |
| `OPENAI_API_KEY` | — | Required if using `openai` provider |

## Claude Code Integration

This repo includes a [Claude Code](https://claude.ai/claude-code) custom slash command. After installing SignalRAG, use `/signalrag <query>` inside Claude Code to search your Signal messages directly from the conversation.

The command checks index freshness, runs semantic search, and summarizes results. It supports conversation and date filters.

## Contributing

Contributions are welcome. Some areas that could use help:

- **Linux/Windows support**: Alternative key extraction methods for non-macOS platforms
- **Additional embedding models**: Support for other models via Ollama or HuggingFace
- **Visualization**: Interactive graph visualization, timeline views
- **Performance**: Faster indexing for very large databases
- **Testing**: Expanded test coverage with mock fixtures

## License

[MIT](LICENSE)
