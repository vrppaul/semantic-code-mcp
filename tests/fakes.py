"""Hand-written stand-ins for the services' parts: each keeps what it was handed."""

import hashlib
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
    """Vectors without a model: each text maps to a fixed vector derived from its hash."""

    def __init__(self) -> None:
        self.embedded_texts: list[str] = []

    @property
    @override
    def is_loaded(self) -> bool:
        return True

    @override
    def embed_text(self, text: str) -> list[float]:
        self.embedded_texts.append(text)
        return vector_of(text)

    @override
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.embedded_texts.extend(texts)
        return [vector_of(text) for text in texts]


class FakeVectorStore(VectorStore):
    """A store that answers every search with the hits it was given and keeps what is written."""

    def __init__(self, hits: list[SearchResult]) -> None:
        self.hits = hits
        self.added: list[ChunkWithEmbedding] = []
        self.deleted_files: list[str] = []
        self.searched_limits: list[int] = []

    @override
    def add_chunks(self, items: list[ChunkWithEmbedding]) -> None:
        self.added.extend(items)

    @override
    def search_hybrid(
        self,
        query_embedding: list[float],
        query_text: str,
        limit: int,
        vector_weight: float,
    ) -> list[SearchResult]:
        self.searched_limits.append(limit)
        return self.hits[:limit]

    @override
    def delete_by_files(self, file_paths: list[str]) -> None:
        self.deleted_files.extend(file_paths)
        self.added = [item for item in self.added if item.chunk.file_path not in file_paths]

    @override
    def statistics(self) -> StoreStatistics:
        files = sorted({item.chunk.file_path for item in self.added})
        return StoreStatistics(indexed_files=files, chunks_count=len(self.added))

    @override
    def clear(self) -> None:
        self.added = []


def vector_of(text: str) -> list[float]:
    """A fixed vector for a text, its components in [-1, 1]."""
    digest = hashlib.sha256(text.encode()).digest()
    repeated = (digest * (EMBEDDING_DIMENSION // len(digest) + 1))[:EMBEDDING_DIMENSION]
    return [byte / BYTE_RANGE * VECTOR_SPAN - VECTOR_OFFSET for byte in repeated]


class ProgressRecorder:
    """A progress callback that keeps every progress value it is handed."""

    def __init__(self) -> None:
        self.progress_values: list[float] = []

    async def __call__(self, progress: float, _total: float, _message: str) -> None:
        self.progress_values.append(progress)
