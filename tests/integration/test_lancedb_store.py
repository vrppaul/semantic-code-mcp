"""Tests for the LanceDB store: what is written comes back, by vector and by words."""

import os
from collections.abc import Callable
from pathlib import Path

import lancedb
import numpy
import pyarrow
import pytest
from filelock import FileLock, Timeout

from semantic_code_mcp.domain.models import (
    Chunk,
    ChunkType,
    IndexCounts,
    IndexIdentity,
    IndexUpdate,
    SourceFile,
    WriteScope,
)
from semantic_code_mcp.storage.lancedb import (
    FILES_TABLE,
    MANIFEST_FILE,
    SCHEMA_VERSION,
    WRITE_LOCK_FILE,
    LanceDBIndexStore,
)
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
# Modes for a table's files: read and enter only, so a write to it fails; and writable again
READ_ONLY = 0o555
WRITABLE = 0o755
IDENTITY = IndexIdentity(model_name="fake-model", chunker_version=1)
OTHER_IDENTITY = IndexIdentity(model_name="other-model", chunker_version=1)
# A time long before any test runs: an index last written then is long unused
LONG_AGO_SECONDS = 1_000_000_000
# What 0.4 kept in a project's index directory: its chunks table, and its record of mtimes
LEGACY_TABLE = "chunks"
LEGACY_RECORDS = "file_mtimes.json"
LEGACY_ROW = {
    "vector": [0.5] * EMBEDDING_DIMENSION,
    "file_path": "/elsewhere/old.py",
    "line_start": 1,
    "line_end": 2,
    "content": "def old(): pass",
    "chunk_type": "function",
    "name": "old",
}


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
    )


def vectors_of(*rows: list[float]) -> numpy.ndarray:
    return numpy.array(rows, dtype=numpy.float32)


@pytest.fixture
def store(tmp_path: Path) -> LanceDBIndexStore:
    return LanceDBIndexStore(
        tmp_path / "index", IDENTITY, lock_timeout_seconds=LOCK_TIMEOUT_SECONDS
    )


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


def test_a_write_records_what_built_the_index(store: LanceDBIndexStore):
    before = store.manifest()
    store.write(update_of([], removed_paths=[]), numpy.empty((0, 0), dtype=numpy.float32))
    after = store.manifest()

    assert (before, after.identity, after.schema_version) == (None, IDENTITY, SCHEMA_VERSION)


def test_a_write_waits_while_another_process_writes(tmp_path: Path):
    store = LanceDBIndexStore(
        tmp_path / "index", IDENTITY, lock_timeout_seconds=SHORT_LOCK_TIMEOUT_SECONDS
    )
    store.directory.mkdir(parents=True)
    chunk = function_chunk("held", "/project/held.py", "def held(): pass")

    with FileLock(store.directory / WRITE_LOCK_FILE), pytest.raises(Timeout):
        store.write(update_of([chunk], removed_paths=[]), vectors_of(TOWARD_QUERY))

    assert store.counts().chunks == 0


def test_a_store_sees_what_another_store_on_its_directory_wrote(tmp_path: Path):
    """Two server processes on one project: each reads what the other has written."""
    directory = tmp_path / "index"
    reader = LanceDBIndexStore(directory, IDENTITY, lock_timeout_seconds=LOCK_TIMEOUT_SECONDS)
    writer = LanceDBIndexStore(directory, IDENTITY, lock_timeout_seconds=LOCK_TIMEOUT_SECONDS)
    first = function_chunk("first", "/project/first.py", "def first(): pass")
    second = function_chunk("second", "/project/second.py", "def second(): pass")
    writer.write(update_of([first], removed_paths=[]), vectors_of(TOWARD_QUERY))
    before = reader.counts().chunks

    writer.write(update_of([second], removed_paths=[]), vectors_of(AWAY_FROM_QUERY))

    assert (before, reader.counts().chunks) == (1, 2)


def test_vectors_of_another_width_replace_the_index_and_leave_the_other_files_unrecorded(
    filled_store: LanceDBIndexStore,
):
    """A model swapped under the same name gives vectors of another width: the index starts
    again from the files written, and the next run indexes the others, unrecorded now."""
    edited = function_chunk("near", "/project/near.py", "def near(): return elapsed_time()")

    filled_store.write(update_of([edited], removed_paths=[]), vectors_of(TOWARD_QUERY * 2))

    wide_query = numpy.full(2 * EMBEDDING_DIMENSION, 1.0, dtype=numpy.float32)
    assert filled_store.counts() == IndexCounts(files=1, chunks=1)
    assert [result.name for result in filled_store.search_vector(wide_query, LIMIT)] == ["near"]


def test_an_index_of_another_identity_on_the_project_is_kept_apart(
    filled_store: LanceDBIndexStore, tmp_path: Path
):
    """Two models, or two versions of the chunker, on one project never write each other's index."""
    other = LanceDBIndexStore(
        tmp_path / "index", OTHER_IDENTITY, lock_timeout_seconds=LOCK_TIMEOUT_SECONDS
    )
    chunk = function_chunk("other", "/project/other.py", "def other(): pass")

    other.write(update_of([chunk], removed_paths=[]), vectors_of(AWAY_FROM_QUERY))

    assert (filled_store.counts(), other.counts()) == (
        IndexCounts(files=3, chunks=3),
        IndexCounts(files=1, chunks=1),
    )


def leave_an_index_of_another_identity(project_directory: Path) -> list[str]:
    """Another model's index of the project; the names it made in the project's directory."""
    other = LanceDBIndexStore(
        project_directory, OTHER_IDENTITY, lock_timeout_seconds=LOCK_TIMEOUT_SECONDS
    )
    chunk = function_chunk("other", "/project/other.py", "def other(): pass")
    other.write(update_of([chunk], removed_paths=[]), vectors_of(AWAY_FROM_QUERY))
    return [other.directory.name]


def leave_an_index_of_version_0_4(project_directory: Path) -> list[str]:
    """What 0.4 kept in the project's directory: a chunks table and a record of mtimes."""
    lancedb.connect(str(project_directory)).create_table(
        LEGACY_TABLE, data=pyarrow.Table.from_pylist([LEGACY_ROW])
    )
    (project_directory / LEGACY_RECORDS).write_text("{}")
    return [f"{LEGACY_TABLE}.lance", LEGACY_RECORDS]


LEFT_INDEXES = [
    pytest.param(leave_an_index_of_another_identity, id="another identity"),
    pytest.param(leave_an_index_of_version_0_4, id="version 0.4"),
]


@pytest.mark.parametrize("leave_index", LEFT_INDEXES)
def test_an_index_written_within_the_week_is_left_alone(
    store: LanceDBIndexStore, tmp_path: Path, leave_index: Callable[[Path], list[str]]
):
    """Another session may still use it; this store reads and writes its own index only."""
    left = leave_index(tmp_path / "index")
    chunk = function_chunk("own", "/project/own.py", "def own(): pass")

    store.write(update_of([chunk], removed_paths=[]), vectors_of(TOWARD_QUERY))

    assert names_in(tmp_path / "index") == sorted([store.directory.name, *left])
    assert store.counts() == IndexCounts(files=1, chunks=1)


@pytest.mark.parametrize("leave_index", LEFT_INDEXES)
def test_an_index_nobody_wrote_for_a_week_is_removed_by_the_next_write(
    store: LanceDBIndexStore, tmp_path: Path, leave_index: Callable[[Path], list[str]]
):
    leave_index(tmp_path / "index")
    set_mtime_of_tree(tmp_path / "index", LONG_AGO_SECONDS)
    chunk = function_chunk("own", "/project/own.py", "def own(): pass")

    store.write(update_of([chunk], removed_paths=[]), vectors_of(TOWARD_QUERY))

    assert names_in(tmp_path / "index") == [store.directory.name]


def names_in(directory: Path) -> list[str]:
    return sorted(entry.name for entry in directory.iterdir())


def set_mtime_of_tree(directory: Path, seconds: int) -> None:
    """Set the modification time of a directory and of everything under it."""
    for path in [directory, *directory.rglob("*")]:
        os.utime(path, (seconds, seconds))


def write_bytes_that_are_not_utf8(manifest: Path) -> None:
    manifest.write_bytes(b'{"identity": "\xff"}')


def make_a_directory(manifest: Path) -> None:
    manifest.mkdir()


@pytest.mark.parametrize(
    "spoil",
    [
        pytest.param(write_bytes_that_are_not_utf8, id="not UTF-8"),
        pytest.param(make_a_directory, id="a directory"),
    ],
)
def test_a_manifest_that_cannot_be_read_counts_as_none(
    filled_store: LanceDBIndexStore, spoil: Callable[[Path], None]
):
    manifest = filled_store.directory / MANIFEST_FILE
    manifest.unlink()
    spoil(manifest)

    assert filled_store.manifest() is None


def test_a_whole_write_that_fails_midway_never_leaves_records_of_missing_chunks(
    filled_store: LanceDBIndexStore,
):
    """Records go before chunks: a write stopped between them leaves chunks, never records
    of chunks that are gone, which would read as an up-to-date, empty index."""
    files_table_directory = filled_store.directory / f"{FILES_TABLE}.lance"
    set_mode_of_tree(files_table_directory, READ_ONLY)
    only = function_chunk("only", "/project/only.py", "def only(): pass")

    with pytest.raises((OSError, RuntimeError)):
        filled_store.write(
            update_of([only], removed_paths=[], scope=WriteScope.everything),
            vectors_of(TOWARD_QUERY),
        )
    set_mode_of_tree(files_table_directory, WRITABLE)

    assert filled_store.counts() == IndexCounts(files=3, chunks=3)


def set_mode_of_tree(directory: Path, mode: int) -> None:
    """Set the mode of a directory, everything under it, and their directories."""
    for path in [directory, *directory.rglob("*")]:
        path.chmod(mode)
