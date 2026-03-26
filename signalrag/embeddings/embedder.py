"""Embedding model wrapper."""

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    """Wraps a sentence-transformers model for encoding text."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], *, batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        """Encode a list of texts into embeddings.

        Returns numpy array of shape (len(texts), dimension).
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string. Returns 1D array."""
        return self.model.encode(
            [query],
            normalize_embeddings=True,
        )[0]
