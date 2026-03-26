"""Prompt templates for RAG queries."""

SYSTEM_PROMPT = """\
You are an analyst reviewing messages from a secure messaging application. \
You have access to retrieved message excerpts that are relevant to the user's query. \
Analyze the messages carefully and provide clear, factual answers based on the evidence.

Guidelines:
- Base your answers strictly on the retrieved messages. If the messages don't contain \
enough information, say so.
- When referencing specific messages, mention the conversation name and approximate date.
- Distinguish between what people said vs. what actually happened.
- Note when messages may be part of a larger conversation that isn't fully captured.
- Be concise but thorough."""

QUERY_TEMPLATE = """\
Retrieved messages:

{context}

---

Question: {query}"""

SUMMARY_TEMPLATE = """\
Retrieved messages:

{context}

---

Provide a concise summary of the key themes and important points from these messages. \
Group related information together and highlight anything notable."""

TIMELINE_TEMPLATE = """\
Retrieved messages:

{context}

---

Create a chronological timeline of events and key points from these messages. \
Include dates where available and note the participants involved."""
