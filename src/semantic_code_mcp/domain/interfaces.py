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
    IndexedFiles,
    IndexManifest,
    IndexUpdate,
    SearchCandidates,
    SearchQuery,
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
        """The model's name, which with its revision fixes what the vectors mean and how
        wide they are."""

    @property
    @abstractmethod
    def revision(self) -> str | None:
        """The revision of the model's weights it loads; None where nothing pins one, as
        for a local path."""

    @property
    @abstractmethod
    def version(self) -> int:
        """The version of how it makes a text's vector of a model's files, bumped by any
        change to that, as reading a model's lowercasing was: each keeps an index of its
        own."""

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
    def indexed_files(self) -> IndexedFiles:
        """The record of every file the index holds, and the paths it holds chunks of with
        no record."""

    @abstractmethod
    def counts(self) -> IndexCounts:
        """How many files and chunks the index holds."""

    @abstractmethod
    def stored_vectors(self, chunk_ids: list[str]) -> dict[str, Vector]:
        """The stored vector of each of these chunks the index holds, by chunk id."""

    @abstractmethod
    def write(self, update: IndexUpdate, vectors: VectorMatrix) -> None:
        """Apply one write of an index run as a unit no other writer interleaves with, in
        this process or another, a cancelled request's thread included: the chunks of the
        files written take the place of those files' and the removed files' chunks, so a
        reader finds a file's old chunks or its new ones, never neither, and the same write
        records those files and what built the index. The vector of each chunk is in its row
        of `vectors`.
        A write stopped midway leaves its files' old records, so the next run writes them
        again; the run's last write also does what the store has to at a run's end.

        Vectors of another width than those the index holds (a model replaced under a name
        nothing pins) start the index again from this write: the files it does not hold are left
        unrecorded, so the next run indexes them."""

    @abstractmethod
    def search_candidates(
        self, query_vector: Vector, query: SearchQuery, per_channel: int
    ) -> SearchCandidates:
        """The `per_channel` chunks of the query's scope nearest its vector and the
        `per_channel` whose words best match its text, each with the cosine similarity of
        its vector to the query's, both read from one state of the index."""


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
