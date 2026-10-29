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
    SearchScope,
    VectorIdentity,
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
    """Keeps one project's index of one identity: the chunks and their words, the record of
    each file they came from, the manifest of what built them, and beside the chunks the
    vectors of each model that embedded them, kept apart by the model's identity; finds
    chunks again."""

    @abstractmethod
    def manifest(self) -> IndexManifest | None:
        """What built the index and when it was last written; None before the first write."""

    @abstractmethod
    def try_hold_for_run(self) -> bool:
        """Take the index for one run of words, so no other run, in this process or another,
        builds it meanwhile; whether it was free. Never waits, so a run cancelled as it asks
        never holds it after; `release_run` frees it, and so does its process ending. The
        run's own writes, and another session's vectors, do not wait for it."""

    @abstractmethod
    def release_run(self) -> None:
        """Free the index a run held."""

    @abstractmethod
    def run_held(self) -> bool:
        """Whether a run holds the index now, this session's or another's; never waits, and
        leaves every run's hold as it was, though a run's try in the same moment is turned
        away for a retry: `index_status` asks it, never a search."""

    @abstractmethod
    def indexed_files(self) -> IndexedFiles:
        """The stamp of every file the index holds, by its path's text, and the paths it holds
        chunks of with no record."""

    @abstractmethod
    def counts(self, model: VectorIdentity) -> IndexCounts:
        """How many files and chunks the index holds, and how many of those chunks have the
        model's vectors."""

    @abstractmethod
    def write(self, update: IndexUpdate) -> None:
        """Apply one write of an index run as a unit no other writer interleaves with, in
        this process or another, a cancelled request's thread included: the chunks of the
        files written take the place of those files' and the removed files' chunks, so a
        reader finds a file's old chunks or its new ones, never neither, and the same write
        records those files and what built the index. A chunk the write keeps, by its id,
        keeps the vectors every model gave it; those of the chunks it drops go with them.
        A write stopped midway leaves its files' old records, so the next run writes them
        again; the run's last write also does what the store has to at a run's end."""

    @abstractmethod
    def chunks_without_vectors(self, model: VectorIdentity) -> list[Chunk]:
        """Every chunk the index holds that has no vector of the model, in the order they
        were written."""

    @abstractmethod
    def add_vectors(
        self, model: VectorIdentity, chunk_ids: list[str], vectors: VectorMatrix
    ) -> None:
        """Keep the model's vector of each chunk, the row of `vectors` at its id's position,
        as a unit, as `write` is; an id the index no longer holds is skipped, as a write of
        words may have dropped it meanwhile. Vectors of another width than the model's stored
        ones (a model replaced under a name nothing pins) take the place of every vector of
        that model, the words and other models' vectors untouched."""

    @abstractmethod
    def remove_vectors(self, model: VectorIdentity) -> None:
        """Drop every vector of the model, so its next search by meaning embeds the chunks
        again."""

    @abstractmethod
    def search_words(self, query: SearchQuery, limit: int) -> list[Chunk]:
        """The `limit` chunks of the query's scope whose words best match its text, best
        first; none for a text of no word."""

    @abstractmethod
    def chunks_named(self, folded_name: str, scope: SearchScope) -> list[Chunk]:
        """The chunks of the scope whose name, case-folded, is this one, in the order of
        their paths and lines."""

    @abstractmethod
    def search(
        self, model: VectorIdentity, query_vector: Vector, query: SearchQuery, per_channel: int
    ) -> SearchCandidates:
        """The `per_channel` chunks of the query's scope whose vectors of the model lie
        nearest its vector and the `per_channel` whose words best match its text, each with
        the cosine similarity of its vector to the query's where it has one, both read from
        one state of the index."""


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
