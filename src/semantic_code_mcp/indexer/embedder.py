"""Embedding generation with sentence-transformers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

log = structlog.get_logger()


class Embedder:
    """Generates embeddings for code chunks using sentence-transformers.

    Requires a pre-loaded SentenceTransformer model. Model loading
    should happen once at container initialization.
    """

    def __init__(self, model: SentenceTransformer) -> None:
        """Initialize the embedder with a pre-loaded model.

        Args:
            model: Pre-loaded SentenceTransformer model.
        """
        self._model = model

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        dim = self._model.get_sentence_embedding_dimension()
        assert isinstance(dim, int)
        return dim

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        log.debug("embedding_batch", count=len(texts))
        embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [e.tolist() for e in embeddings]
