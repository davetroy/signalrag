"""RAG query engine: orchestrates retrieval + LLM generation."""

import os

from .retriever import Retriever
from .prompt import SYSTEM_PROMPT, QUERY_TEMPLATE, SUMMARY_TEMPLATE, TIMELINE_TEMPLATE
from ..embeddings.embedder import Embedder
from ..embeddings.store import MessageStore


class RAGEngine:
    """End-to-end RAG: query -> retrieve -> prompt -> LLM answer."""

    def __init__(
        self,
        *,
        retriever: Retriever | None = None,
        llm_provider: str | None = None,
        llm_model: str | None = None,
    ):
        self.retriever = retriever or Retriever()
        self._provider = llm_provider or os.environ.get("SIGNALRAG_LLM_PROVIDER", "anthropic")
        self._model = llm_model or os.environ.get("SIGNALRAG_LLM_MODEL", self._default_model())
        self._client = None

    def _default_model(self) -> str:
        defaults = {
            "anthropic": "claude-sonnet-4-20250514",
            "openai": "gpt-4o",
            "ollama": "llama3.1:8b",
        }
        return defaults.get(self._provider, "claude-sonnet-4-20250514")

    def ask(
        self,
        query: str,
        *,
        limit: int = 10,
        context_messages: int = 5,
        conversation_id: str | None = None,
        since: int | None = None,
        mode: str = "query",
    ) -> dict:
        """Ask a question over Signal messages.

        Args:
            query: The user's question
            limit: Number of chunks to retrieve
            context_messages: Surrounding messages to include per hit
            conversation_id: Filter to a specific conversation
            since: Only search messages after this timestamp (ms)
            mode: 'query' (default), 'summary', or 'timeline'

        Returns dict with 'answer', 'sources', and 'retrieval_results'.
        """
        # Step 1: Retrieve
        results = self.retriever.search_with_context(
            query,
            limit=limit,
            context_messages=context_messages,
            conversation_id=conversation_id,
            since=since,
        )

        if not results:
            return {
                "answer": "No relevant messages found for your query.",
                "sources": [],
                "retrieval_results": [],
            }

        # Step 2: Build prompt
        context = self.retriever.format_results_for_llm(results)
        templates = {
            "query": QUERY_TEMPLATE,
            "summary": SUMMARY_TEMPLATE,
            "timeline": TIMELINE_TEMPLATE,
        }
        template = templates.get(mode, QUERY_TEMPLATE)
        user_message = template.format(context=context, query=query)

        # Step 3: Call LLM
        answer = self._call_llm(user_message)

        # Step 4: Build sources
        sources = []
        seen = set()
        for r in results:
            key = (r["conversation_name"], r["timestamp_start"])
            if key not in seen:
                seen.add(key)
                sources.append({
                    "conversation": r["conversation_name"],
                    "timestamp": r["timestamp_start"],
                    "chunk_type": r["chunk_type"],
                    "distance": r.get("_distance"),
                })

        return {
            "answer": answer,
            "sources": sources,
            "retrieval_results": results,
        }

    def search_only(
        self,
        query: str,
        *,
        limit: int = 10,
        conversation_id: str | None = None,
        since: int | None = None,
        literal: bool = False,
    ) -> list[dict]:
        """Search without LLM — semantic or literal text match."""
        return self.retriever.search(
            query,
            limit=limit,
            conversation_id=conversation_id,
            since=since,
            literal=literal,
        )

    def _call_llm(self, user_message: str) -> str:
        if self._provider == "anthropic":
            return self._call_anthropic(user_message)
        elif self._provider == "openai":
            return self._call_openai(user_message)
        elif self._provider == "ollama":
            return self._call_ollama(user_message)
        else:
            raise ValueError(f"Unknown LLM provider: {self._provider}")

    def _call_anthropic(self, user_message: str) -> str:
        import anthropic
        if self._client is None:
            self._client = anthropic.Anthropic()
        response = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text

    def _call_openai(self, user_message: str) -> str:
        from openai import OpenAI
        if self._client is None:
            self._client = OpenAI()
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content

    def _call_ollama(self, user_message: str) -> str:
        from openai import OpenAI
        if self._client is None:
            self._client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
            )
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content
