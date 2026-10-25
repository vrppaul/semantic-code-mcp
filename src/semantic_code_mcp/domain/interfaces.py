"""The interfaces the services meet their parts at.

Each is an abstract class its implementations inherit, so a class missing a member
fails when it is defined, not when a request first reaches the gap.
"""

from abc import ABC, abstractmethod

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
    def write(self, update: IndexUpdate, vectors: VectorMatrix) -> None:
        """Apply one index run's write as a unit no other writer interleaves with, in this
        process or another, a cancelled request's thread included; the manifest is written
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
        """The version of the chunks it makes; an index of another version is rebuilt."""

    @abstractmethod
    def chunk_file(self, file_path: str) -> list[Chunk]:
        """The chunks of one file; empty when the file cannot be read."""
