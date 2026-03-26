# SignalRAG Search

Run a semantic search over your indexed Signal messages using the signalrag CLI.

## Usage

`/signalrag <query>` — search for a topic, person, or phrase in Signal messages.

Arguments after the command name are treated as the search query: `$ARGUMENTS`

## Instructions

0. **Check index freshness first.** Run `signalrag stats` and check the "Last Indexed" date. If the index is more than 1 day old, run `signalrag index` to update it before searching. Inform the user that you're updating the index.

1. Run the search using the signalrag CLI:

```
signalrag search "$ARGUMENTS" --limit 10
```

2. Present the results to the user in a clear, readable format.
3. Summarize the key findings — which conversations mentioned the topic, approximate date ranges, and any notable content.
4. If the user wants more results, re-run with a higher `--limit`.
5. If the user wants to filter by conversation, add `-c <name>`.
6. If the user wants to filter by date, add `-s YYYY-MM-DD`.
