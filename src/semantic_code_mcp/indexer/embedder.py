"""Embedding generation with sentence-transformers."""

import structlog
from sentence_transformers import SentenceTransformer

from semantic_code_mcp.config import Settings

log = structlog.get_logger()


class Embedder:
    """Generates embeddings for code chunks using sentence-transformers.

    The model is loaded lazily on first use to avoid slow startup.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the embedder.

        Args:
            settings: Application settings with model configuration.
        """
        self.model_name = settings.embedding_model
        self.device = settings.embedding_device
        self._model: SentenceTransformer | None = None
        self._embedding_dim: int | None = None

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension. Loads model if needed."""
        if self._embedding_dim is None:
            self.load()
        return self._embedding_dim  # type: ignore[return-value]

    def load(self) -> None:
        """Explicitly load the model.

        This can be called to pre-load the model before first use.
        """
        if self._model is not None:
            return

        log.info("loading_embedding_model", model=self.model_name, device=self.device)

        device = None if self.device == "auto" else self.device
        self._model = SentenceTransformer(self.model_name, device=device)
        self._embedding_dim = self._model.get_sentence_embedding_dimension()

        log.info(
            "embedding_model_loaded",
            model=self.model_name,
            embedding_dim=self._embedding_dim,
        )

    def _ensure_loaded(self) -> SentenceTransformer:
        """Ensure model is loaded and return it."""
        if self._model is None:
            self.load()
        return self._model  # type: ignore[return-value]

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        model = self._ensure_loaded()
        embedding = model.encode(text, convert_to_numpy=True)
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

        model = self._ensure_loaded()
        log.debug("embedding_batch", count=len(texts))

        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [e.tolist() for e in embeddings]
