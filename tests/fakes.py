"""Hand-written stand-ins for the services' parts: each keeps what it was handed."""

import hashlib
from collections import defaultdict
from typing import override

from semantic_code_mcp.domain.interfaces import Embedder, VectorStore
from semantic_code_mcp.domain.models import (
    ChunkWithEmbedding,
    SearchResult,
    StoreStatistics,
)

EMBEDDING_DIMENSION = 384
BYTE_RANGE = 255
# A digest byte scaled into [-1, 1]
VECTOR_SPAN = 2
VECTOR_OFFSET = 1


class FakeEmbedder(Embedder):
    """Vectors without a model: each text maps to a fixed vector derived from its hash.

    It starts loaded unless told otherwise; one that starts unloaded loads on `load()`
    or on its first text, as the real one does.
    """

    def __init__(self, *, loaded: bool = True) -> None:
        self.embedded_texts: list[str] = []
        self._loaded = loaded

    @property
    @override
    def is_loaded(self) -> bool:
        return self._loaded

    @override
    def load(self) -> None:
        self._loaded = True

    @override
    def embed_text(self, text: str) -> list[float]:
        self._loaded = True
        self.embedded_texts.append(text)
        return vector_of(text)

    @override
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self._loaded = True
        self.embedded_texts.extend(texts)
        return [vector_of(text) for text in texts]


class FakeVectorStore(VectorStore):
    """A store that answers every search with the hits it was given and keeps what is written."""

    def __init__(self, hits: list[SearchResult]) -> None:
        self.hits = hits
        self.chunks_by_file: dict[str, list[ChunkWithEmbedding]] = defaultdict(list)
        self.deleted_files: list[str] = []

    @override
    def add_chunks(self, items: list[ChunkWithEmbedding]) -> None:
        for item in items:
            self.chunks_by_file[item.chunk.file_path].append(item)

    @override
    def search_hybrid(
        self,
        query_embedding: list[float],
        query_text: str,
        limit: int,
        vector_weight: float,
    ) -> list[SearchResult]:
        return self.hits[:limit]

    @override
    def delete_by_files(self, file_paths: list[str]) -> None:
        self.deleted_files.extend(file_paths)
        for file_path in file_paths:
            self.chunks_by_file.pop(file_path, None)

    @override
    def statistics(self) -> StoreStatistics:
        return StoreStatistics(
            indexed_files=sorted(self.chunks_by_file),
            chunks_count=sum(len(chunks) for chunks in self.chunks_by_file.values()),
        )

    @override
    def clear(self) -> None:
        self.chunks_by_file.clear()


def vector_of(text: str) -> list[float]:
    """A fixed vector for a text, its components in [-1, 1]."""
    digest = hashlib.sha256(text.encode()).digest()
    repeated = (digest * (EMBEDDING_DIMENSION // len(digest) + 1))[:EMBEDDING_DIMENSION]
    return [byte / BYTE_RANGE * VECTOR_SPAN - VECTOR_OFFSET for byte in repeated]


class ProgressRecorder:
    """A progress callback that keeps every progress value and message it is handed."""

    def __init__(self) -> None:
        self.progress_values: list[float] = []
        self.messages: list[str] = []

    async def __call__(self, progress: float, _total: float, message: str) -> None:
        self.progress_values.append(progress)
        self.messages.append(message)
