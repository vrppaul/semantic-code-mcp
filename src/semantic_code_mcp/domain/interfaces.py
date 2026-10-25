"""The interfaces the services meet their parts at.

Each is an abstract class its implementations inherit, so a class missing a member
fails when it is defined, not when a request first reaches the gap.
"""

from abc import ABC, abstractmethod

from semantic_code_mcp.domain.models import (
    Chunk,
    ChunkWithEmbedding,
    SearchResult,
    StoreStatistics,
)


class Embedder(ABC):
    """Turns text into vectors."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model is in memory, so the next call does not wait for it."""

    @abstractmethod
    def load(self) -> None:
        """Load the model now, if it is not in memory; blocks while it loads."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """The vector of one text; loads the model on first use."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """The vectors of many texts, in order; loads the model on first use."""


class VectorStore(ABC):
    """Keeps chunks with their vectors and finds them again."""

    @abstractmethod
    def add_chunks(self, items: list[ChunkWithEmbedding]) -> None:
        """Store chunks with their vectors."""

    @abstractmethod
    def search_hybrid(
        self,
        query_embedding: list[float],
        query_text: str,
        limit: int,
        vector_weight: float,
    ) -> list[SearchResult]:
        """The best chunks by vector similarity and full-text match combined."""

    @abstractmethod
    def delete_by_files(self, file_paths: list[str]) -> None:
        """Remove every chunk of the given files."""

    @abstractmethod
    def statistics(self) -> StoreStatistics:
        """The files the store has chunks of, and how many chunks."""

    @abstractmethod
    def clear(self) -> None:
        """Remove every chunk."""


class Chunker(ABC):
    """Splits source files into chunks."""

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """The file extensions this chunker reads, such as `.py`."""

    @abstractmethod
    def chunk_file(self, file_path: str) -> list[Chunk]:
        """The chunks of one file; empty when the file cannot be read."""
