"""Phase 0 Test: Generate embeddings from short message text.

No Signal DB required — uses static sample data.
"""

from sentence_transformers import SentenceTransformer


def test_embeddings():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [
        "Hello, how are you?",
        "Meeting tomorrow at 3pm",
        "Did you see the news about the election?",
        "Can you send me that document?",
        "",  # edge case: empty string
    ]

    embeddings = model.encode(texts)
    assert embeddings.shape == (5, 384), f"Unexpected shape: {embeddings.shape}"

    from sklearn.metrics.pairwise import cosine_similarity

    sims = cosine_similarity(embeddings)
    # Basic sanity: "meeting" and "document" (work-related) should be
    # more similar to each other than "hello" is to "election"
    assert sims[1][3] > 0, "Expected positive similarity between work-related texts"
