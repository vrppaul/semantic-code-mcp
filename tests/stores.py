"""What the store tests share: chunks, their records and writes of them, vectors near and far
from the query and the models that gave them, the searches of each channel, and the stores the
contract's tests run over."""

from pathlib import Path

import numpy
import pytest

from semantic_code_mcp.domain.interfaces import IndexStore
from semantic_code_mcp.domain.models import (
    WHOLE_PROJECT,
    Candidate,
    Chunk,
    ChunkKind,
    IndexIdentity,
    IndexUpdate,
    Language,
    SearchCandidates,
    SearchQuery,
    SourceFile,
    VectorIdentity,
)
from semantic_code_mcp.storage.index_directories import DirectoryStore, IndexDirectory
from semantic_code_mcp.storage.lancedb import LanceDBIndexStore
from semantic_code_mcp.storage.sqlite import SqliteIndexStore
from tests.fakes import EMBEDDING_DIMENSION, FAKE_MODEL

# A store's class, built on an index's directory in the layout the class names
type StoreClass = type[DirectoryStore]
# Every store the contract's tests run over
STORE_CLASSES = [
    pytest.param(LanceDBIndexStore, id="lancedb"),
    pytest.param(SqliteIndexStore, id="sqlite"),
]
# What a search finds in an index that holds no chunks
NO_CANDIDATES = SearchCandidates(by_vector=[], by_keyword=[])

TOWARD_QUERY = [0.9] * EMBEDDING_DIMENSION
QUERY = numpy.full(EMBEDDING_DIMENSION, 1.0, dtype=numpy.float32)
AWAY_FROM_QUERY = [-0.9] * EMBEDDING_DIMENSION
ALTERNATING = [0.1, -0.1] * (EMBEDDING_DIMENSION // 2)
LIMIT = 10
# A text no chunk's words match, so a search finds by vector alone
NO_WORDS_MATCH = "zzzz"
LOCK_TIMEOUT_SECONDS = 5.0
IDENTITY = IndexIdentity(chunker_version=1)
OTHER_IDENTITY = IndexIdentity(chunker_version=2)
# The model whose vectors the tests add, and another, whose vectors an index keeps apart
MODEL = FAKE_MODEL
OTHER_MODEL = VectorIdentity(model_name="other-model", model_revision=None, embedder_version=1)


def index_directory_of(
    store_class: StoreClass,
    project_directory: Path,
    identity: IndexIdentity,
    lock_timeout_seconds: float = LOCK_TIMEOUT_SECONDS,
) -> IndexDirectory:
    """The directory of a project's index of an identity in a store's layout."""
    return IndexDirectory(
        project_directory,
        identity,
        store_class.LAYOUT,
        lock_timeout_seconds=lock_timeout_seconds,
    )


def open_store(
    store_class: StoreClass, project_directory: Path, identity: IndexIdentity
) -> IndexStore:
    """A store of a project's index of an identity, as another session would open it."""
    return store_class(index_directory_of(store_class, project_directory, identity))


def function_chunk(name: str, path: str, content: str) -> Chunk:
    return Chunk(
        chunk_id=f"{path}#{name}",
        path=Path(path),
        language=Language.python,
        kind=ChunkKind.function,
        name=name,
        parent=None,
        line_start=1,
        line_end=3,
        content=content,
    )


def record_of(path: str) -> SourceFile:
    return SourceFile(path=Path(path), mtime_ns=1, size_bytes=1)


def update_of(chunks: list[Chunk], removed_paths: list[Path]) -> IndexUpdate:
    """A write that ends its run, as a run of one write does: the store keeps the index up
    in it."""
    return IndexUpdate(
        files=[record_of(chunk.path.as_posix()) for chunk in chunks],
        chunks=chunks,
        removed_paths=removed_paths,
        last_of_run=True,
    )


def vectors_of(*rows: list[float]) -> numpy.ndarray:
    return numpy.array(rows, dtype=numpy.float32)


def fill(store: IndexStore, chunks: list[Chunk], vectors: numpy.ndarray) -> None:
    """Write the chunks in a run of one write, then add the model's vectors of them, a row
    each."""
    store.write(update_of(chunks, removed_paths=[]))
    store.add_vectors(MODEL, [chunk.chunk_id for chunk in chunks], vectors)


def query_of(text: str) -> SearchQuery:
    """A query of the whole project."""
    return SearchQuery(text=text, limit=LIMIT, scope=WHOLE_PROJECT)


def nearest(store: IndexStore, query: numpy.ndarray = QUERY) -> list[Candidate]:
    """What the vector channel of a search by the model's vectors finds."""
    return store.search(MODEL, query, query_of(NO_WORDS_MATCH), LIMIT).by_vector


def matching(store: IndexStore, text: str) -> list[Chunk]:
    """What a search by words for the text finds."""
    return store.search_words(query_of(text), LIMIT)
