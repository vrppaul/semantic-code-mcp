"""Embeddings with sentence-transformers, the model loaded on first use.

Loading the model imports torch and reads the weights: seconds an `index_status`
call never needs to pay. So nothing loads until the first text is embedded, and that
happens in a worker thread (the services embed through `asyncio.to_thread`), never
on the event loop. A lock keeps two first calls from loading the model twice.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, override

import numpy
import structlog

from semantic_code_mcp.domain.interfaces import Embedder, Vector, VectorMatrix

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

log = structlog.get_logger()


class SentenceTransformerEmbedder(Embedder):
    """An embedder over a sentence-transformers model, loaded when first needed."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None
        self._loading = threading.Lock()

    @property
    @override
    def is_loaded(self) -> bool:
        return self._model is not None

    @override
    def load(self) -> None:
        self._loaded_model()

    @override
    def embed_query(self, text: str) -> Vector:
        vector = self._loaded_model().encode(text, convert_to_numpy=True)
        return numpy.asarray(vector, dtype=numpy.float32)

    @override
    def embed_documents(self, texts: list[str]) -> VectorMatrix:
        # For no texts the model gives a flat empty array, not rows of its width
        log.debug("embedding_documents", count=len(texts))
        vectors = self._loaded_model().encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return numpy.asarray(vectors, dtype=numpy.float32)

    def _loaded_model(self) -> SentenceTransformer:
        """The model, loading it on the first call."""
        with self._loading:
            if self._model is None:
                self._model = load_model(self._model_name)
            return self._model


def load_model(model_name: str) -> SentenceTransformer:
    """Load a sentence-transformers model by name; imports torch on first call."""
    # Deferred: importing sentence-transformers pulls in torch, seconds of start-up
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    log.info("loading_embedding_model", model=model_name)
    model = SentenceTransformer(model_name)
    log.info("embedding_model_loaded", model=model_name)
    return model
