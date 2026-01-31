"""Protocols for dependency injection."""

from collections.abc import Awaitable, Callable
from typing import Protocol

from semantic_code_mcp.models import Chunk, ChunkWithEmbedding, SearchResult

# Matches MCP's ctx.report_progress(progress, total, message) signature
ProgressCallback = Callable[[float, float, str], Awaitable[None]]


class VectorStoreProtocol(Protocol):
    """Interface for vector storage."""

    def add_chunks(self, items: list[ChunkWithEmbedding]) -> None:
        """Add chunks with embeddings to the store."""
        ...

    def search_hybrid(
        self,
        query_embedding: list[float],
        query_text: str,
        limit: int,
        vector_weight: float,
    ) -> list[SearchResult]:
        """Search using hybrid vector + full-text search."""
        ...

    def delete_by_file(self, file_path: str) -> None:
        """Delete all chunks for a specific file."""
        ...

    def get_indexed_files(self) -> list[str]:
        """Get list of all indexed file paths."""
        ...

    def count(self) -> int:
        """Count total chunks in the store."""
        ...

    def clear(self) -> None:
        """Delete all chunks from the store."""
        ...


class EmbedderProtocol(Protocol):
    """Interface for embedding generation."""

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        ...


class ChunkerProtocol(Protocol):
    """Interface for code chunking."""

    def chunk_file(self, file_path: str) -> list[Chunk]:
        """Extract chunks from a source file."""
        ...
