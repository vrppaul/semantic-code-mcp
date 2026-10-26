"""The interfaces the services meet their parts at.

Each is an abstract class its implementations inherit, so a class missing a member
fails when it is defined, not when a request first reaches the gap.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy
from numpy.typing import NDArray

from semantic_code_mcp.domain.models import (
    Chunk,
    IndexCounts,
    IndexManifest,
    IndexUpdate,
    SearchResult,
    SourceFile,
)

# The vector of one text
type Vector = NDArray[numpy.float32]
# The vectors of many texts, one row per text, in their order
type VectorMatrix = NDArray[numpy.float32]


class Embedder(ABC):
    """Turns text into vectors."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model's name, which fixes what the vectors mean and how wide they are."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model is in memory, so the next call does not wait for it."""

    @abstractmethod
    def load(self) -> None:
        """Load the model now, if it is not in memory; blocks while it loads."""

    @abstractmethod
    def dimension(self) -> int:
        """How wide the model's vectors are; loads the model on first use."""

    @abstractmethod
    def embed_query(self, text: str) -> Vector:
        """The vector of a search query; loads the model on first use."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> VectorMatrix:
        """The vectors of at least one text to be searched, one row each; loads the model
        on first use."""


class IndexStore(ABC):
    """Keeps one project's index of one identity: the chunks with their vectors, the record
    of each file they came from, and the manifest of what built them; finds chunks again."""

    @abstractmethod
    def manifest(self) -> IndexManifest | None:
        """What built the index and when it was last written; None before the first write."""

    @abstractmethod
    def indexed_files(self) -> list[SourceFile]:
        """The record of every file the index holds, as the scan that indexed it found it."""

    @abstractmethod
    def counts(self) -> IndexCounts:
        """How many files and chunks the index holds."""

    @abstractmethod
    def chunked_paths(self) -> set[Path]:
        """The paths of the files the index holds chunks of, recorded or not: a write that
        stopped between a file's chunks and its record leaves them unrecorded."""

    @abstractmethod
    def stored_vectors(self, chunk_ids: list[str]) -> dict[str, Vector]:
        """The stored vector of each of these chunks the index holds, by chunk id."""

    @abstractmethod
    def keep_up(self) -> None:
        """Fold what was written into the index's own indexes now, as after a run of many
        writes, rather than when enough writes have gathered."""

    @abstractmethod
    def write(self, update: IndexUpdate, vectors: VectorMatrix) -> None:
        """Apply one write of an index run as a unit no other writer interleaves with, in
        this process or another, a cancelled request's thread included: the chunks of the
        files written take the place of those files' and the removed files' chunks, so a
        reader finds a file's old chunks or its new ones; then their records; the manifest
        last. The vector of each chunk is in its row of `vectors`.

        Vectors of another width than those the index holds (a model swapped under the same
        name) start the index again from this write: the files it does not hold are left
        unrecorded, so the next run indexes them."""

    @abstractmethod
    def search_hybrid(
        self,
        query_vector: Vector,
        query_text: str,
        limit: int,
        vector_weight: float,
    ) -> list[SearchResult]:
        """The best chunks by vector similarity and full-text match combined."""


class Chunker(ABC):
    """Splits source files into chunks."""

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """The file extensions this chunker reads, such as `.py`."""

    @property
    @abstractmethod
    def version(self) -> int:
        """The version of the chunks it makes; each version keeps an index of its own."""

    @abstractmethod
    def chunk_file(self, project_root: Path, path: Path) -> list[Chunk]:
        """The chunks of the file at `path` from the project root, each naming that path;
        empty when the file cannot be read."""
