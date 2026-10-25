"""Tests for the LanceDB store: what is written comes back, by vector and by words."""

from datetime import UTC, datetime
from pathlib import Path

import numpy
import pytest
from filelock import FileLock, Timeout

from semantic_code_mcp.domain.errors import IndexRebuiltElsewhereError, IndexWriteError
from semantic_code_mcp.domain.models import (
    Chunk,
    ChunkType,
    IndexCounts,
    IndexIdentity,
    IndexManifest,
    IndexUpdate,
    SourceFile,
    WriteScope,
)
from semantic_code_mcp.storage.lancedb import WRITE_LOCK_FILE, LanceDBIndexStore
from tests.fakes import EMBEDDING_DIMENSION

TOWARD_QUERY = [0.9] * EMBEDDING_DIMENSION
QUERY = numpy.full(EMBEDDING_DIMENSION, 1.0, dtype=numpy.float32)
AWAY_FROM_QUERY = [-0.9] * EMBEDDING_DIMENSION
ALTERNATING = [0.1, -0.1] * (EMBEDDING_DIMENSION // 2)
LIMIT = 10
VECTOR_WEIGHT = 0.5
LOCK_TIMEOUT_SECONDS = 5.0
# Long enough to tell a write that waits from one that does not, short for a test
SHORT_LOCK_TIMEOUT_SECONDS = 0.2
MANIFEST = IndexManifest(
    identity=IndexIdentity(model_name="fake-model", chunker_version=1, schema_version=1),
    indexed_at=datetime(2026, 10, 25, tzinfo=UTC),
)


def function_chunk(name: str, file_path: str, content: str) -> Chunk:
    return Chunk(
        file_path=file_path,
        line_start=1,
        line_end=3,
        content=content,
        chunk_type=ChunkType.function,
        name=name,
    )


def record_of(file_path: str) -> SourceFile:
    return SourceFile(path=Path(file_path), mtime_ns=1, size_bytes=1)


def update_of(
    chunks: list[Chunk], removed_paths: list[Path], scope: WriteScope = WriteScope.changed_files
) -> IndexUpdate:
    return IndexUpdate(
        scope=scope,
        files=[record_of(chunk.file_path) for chunk in chunks],
        chunks=chunks,
        removed_paths=removed_paths,
        manifest=MANIFEST,
    )


def vectors_of(*rows: list[float]) -> numpy.ndarray:
    return numpy.array(rows, dtype=numpy.float32)


@pytest.fixture
def store(tmp_path: Path) -> LanceDBIndexStore:
    return LanceDBIndexStore(tmp_path / "index", lock_timeout_seconds=LOCK_TIMEOUT_SECONDS)


@pytest.fixture
def filled_store(store: LanceDBIndexStore) -> LanceDBIndexStore:
    chunks = [
        function_chunk("near", "/project/near.py", "def near(): return elapsed_time()"),
        function_chunk(
            "keyword",
            "/project/indexer.py",
            'log.debug("timing", duration_ms=round(time.time() - started))',
        ),
        function_chunk("far", "/project/far.py", "def far(): return None"),
    ]
    store.write(
        update_of(chunks, removed_paths=[]),
        vectors_of(TOWARD_QUERY, ALTERNATING, AWAY_FROM_QUERY),
    )
    return store


def test_vector_search_ranks_the_nearest_chunk_first(filled_store: LanceDBIndexStore):
    results = filled_store.search_vector(QUERY, LIMIT)

    assert [result.name for result in results] == ["near", "keyword", "far"]


def test_full_text_search_finds_an_exact_identifier(filled_store: LanceDBIndexStore):
    results = filled_store.search_full_text("duration_ms", LIMIT)

    assert [result.name for result in results] == ["keyword"]


def test_full_text_search_finds_chunks_of_a_later_write_and_not_removed_ones(
    filled_store: LanceDBIndexStore,
):
    later = function_chunk("later", "/project/later.py", "def later(): return retry_budget")
    filled_store.write(
        update_of([later], removed_paths=[Path("/project/indexer.py")]), vectors_of(TOWARD_QUERY)
    )

    assert [result.name for result in filled_store.search_full_text("retry_budget", LIMIT)] == [
        "later"
    ]
    assert filled_store.search_full_text("duration_ms", LIMIT) == []


def test_hybrid_search_returns_both_the_nearest_and_the_keyword_match(
    filled_store: LanceDBIndexStore,
):
    results = filled_store.search_hybrid(QUERY, "duration_ms", 2, VECTOR_WEIGHT)

    assert {result.name for result in results} == {"near", "keyword"}


def test_an_empty_store_finds_nothing(store: LanceDBIndexStore):
    assert store.search_hybrid(QUERY, "anything", LIMIT, VECTOR_WEIGHT) == []


def test_removing_files_takes_only_their_chunks_and_records(filled_store: LanceDBIndexStore):
    filled_store.write(
        update_of([], removed_paths=[Path("/project/near.py"), Path("/project/it's quoted.py")]),
        numpy.empty((0, 0), dtype=numpy.float32),
    )

    assert sorted(str(file.path) for file in filled_store.indexed_files()) == [
        "/project/far.py",
        "/project/indexer.py",
    ]
    assert filled_store.counts().chunks == 2


def test_a_write_of_everything_replaces_every_chunk_and_record(filled_store: LanceDBIndexStore):
    only = function_chunk("only", "/project/only.py", "def only(): pass")

    filled_store.write(
        update_of([only], removed_paths=[], scope=WriteScope.everything), vectors_of(TOWARD_QUERY)
    )

    assert [result.name for result in filled_store.search_vector(QUERY, LIMIT)] == ["only"]
    assert [str(file.path) for file in filled_store.indexed_files()] == ["/project/only.py"]


def test_the_manifest_written_comes_back(store: LanceDBIndexStore):
    before = store.manifest()
    store.write(update_of([], removed_paths=[]), numpy.empty((0, 0), dtype=numpy.float32))

    assert (before, store.manifest()) == (None, MANIFEST)


def test_a_write_waits_while_another_process_writes(tmp_path: Path):
    directory = tmp_path / "index"
    store = LanceDBIndexStore(directory, lock_timeout_seconds=SHORT_LOCK_TIMEOUT_SECONDS)
    directory.mkdir()
    chunk = function_chunk("held", "/project/held.py", "def held(): pass")

    with FileLock(directory / WRITE_LOCK_FILE), pytest.raises(Timeout):
        store.write(update_of([chunk], removed_paths=[]), vectors_of(TOWARD_QUERY))

    assert store.counts().chunks == 0


def test_a_store_sees_what_another_store_on_its_directory_wrote(tmp_path: Path):
    """Two server processes on one project: each reads what the other has written."""
    directory = tmp_path / "index"
    reader = LanceDBIndexStore(directory, lock_timeout_seconds=LOCK_TIMEOUT_SECONDS)
    writer = LanceDBIndexStore(directory, lock_timeout_seconds=LOCK_TIMEOUT_SECONDS)
    first = function_chunk("first", "/project/first.py", "def first(): pass")
    second = function_chunk("second", "/project/second.py", "def second(): pass")
    writer.write(update_of([first], removed_paths=[]), vectors_of(TOWARD_QUERY))
    before = reader.counts().chunks

    writer.write(update_of([second], removed_paths=[]), vectors_of(AWAY_FROM_QUERY))

    assert (before, reader.counts().chunks) == (1, 2)


def test_vectors_of_another_width_are_refused_before_anything_changes(
    filled_store: LanceDBIndexStore,
):
    """A model swapped under the same name: the index holds vectors of another width."""
    edited = function_chunk("near", "/project/near.py", "def near(): return elapsed_time()")

    with pytest.raises(IndexWriteError, match="force=true"):
        filled_store.write(
            update_of([edited], removed_paths=[Path("/project/far.py")]),
            vectors_of(TOWARD_QUERY * 2),
        )

    assert filled_store.counts() == IndexCounts(files=3, chunks=3)


def test_a_write_planned_before_another_process_rebuilt_the_index_is_refused(
    filled_store: LanceDBIndexStore,
):
    other_model = IndexManifest(
        identity=IndexIdentity(model_name="other-model", chunker_version=1, schema_version=1),
        indexed_at=MANIFEST.indexed_at,
    )
    edited = function_chunk("near", "/project/near.py", "def near(): return elapsed_time()")
    update = IndexUpdate(
        scope=WriteScope.changed_files,
        files=[record_of(edited.file_path)],
        chunks=[edited],
        removed_paths=[],
        manifest=other_model,
    )

    with pytest.raises(IndexRebuiltElsewhereError):
        filled_store.write(update, vectors_of(TOWARD_QUERY))

    assert filled_store.manifest() == MANIFEST
