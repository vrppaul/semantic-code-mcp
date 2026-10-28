"""Tests of what the SQLite store does of its own, beside the contract every store holds
(`test_index_store.py`): a write applied whole or not at all, a first build out of every
reader's sight until it ends, and what its schema's version names."""

import sqlite3
import tempfile
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from semantic_code_mcp.domain.models import Chunk, IndexUpdate
from semantic_code_mcp.storage.search_text import search_text
from semantic_code_mcp.storage.sqlite import (
    DAMAGED_SUFFIX,
    DATABASE_FILE,
    SqliteIndexStore,
    SqliteStoreError,
)
from tests.stores import (
    IDENTITY,
    MODEL,
    TOWARD_QUERY,
    fill,
    function_chunk,
    index_directory_of,
    matching,
    record_of,
    update_of,
    vectors_of,
)

# The chunks LanceDB's recorded schema keeps, and what each version of this store's schema
# stored of them: each chunk's search text and the statements that made its tables
CHUNKS_RECORD = Path(__file__).parent.parent / "data" / "schema_outputs.json"
RECORDED_SCHEMA = Path(__file__).parent.parent / "data" / "sqlite_schema_outputs.json"
RECORD_INDENT = 2
# What SQLite answers for a database whose readers read beside its writer
WRITE_AHEAD_LOG_MODE = "wal"
JOURNAL_MODE = "PRAGMA journal_mode"
# Every table, index and full-text index of a database, by name, with the statement that
# made it; FTS5's shadow tables follow from its own
SCHEMA_STATEMENTS = (
    "SELECT name, sql FROM sqlite_schema WHERE sql IS NOT NULL AND name NOT LIKE 'words_%' "
    "ORDER BY name"
)


class RecordedChunks(BaseModel):
    """The chunks a recorded schema keeps."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    chunks: list[Chunk]


class StoredSchema(BaseModel):
    """What the store keeps of some chunks: the text its full-text index reads of each, and
    the statements that made its tables."""

    model_config = ConfigDict(frozen=True)

    search_texts: list[str]
    statements: list[str]


class RecordedSchema(BaseModel):
    """What each version of the store's schema stored of the recorded chunks."""

    model_config = ConfigDict(frozen=True)

    schemas: dict[int, StoredSchema]


def stored_schema(chunks: list[Chunk], project_directory: Path) -> StoredSchema:
    """What the store keeps of the chunks once it has written them."""
    index_directory = index_directory_of(SqliteIndexStore, project_directory, IDENTITY)
    store = SqliteIndexStore(index_directory)
    store.write(update_of(chunks, removed_paths=[]))
    with sqlite3.connect(index_directory.path / DATABASE_FILE) as connection:
        rows = connection.execute(SCHEMA_STATEMENTS).fetchall()
    return StoredSchema(
        search_texts=[search_text(chunk) for chunk in chunks],
        statements=[statement for _name, statement in rows],
    )


def record_schema() -> None:
    """Record what the store keeps under its schema's current version, unless that version is
    recorded already. Run it by hand from the repository's root, through `uv run python -c`
    and an import of this module."""
    chunks = RecordedChunks.model_validate_json(CHUNKS_RECORD.read_text()).chunks
    recorded = RecordedSchema.model_validate_json(RECORDED_SCHEMA.read_text())
    with tempfile.TemporaryDirectory() as directory:
        schema = stored_schema(chunks, Path(directory))
    updated = RecordedSchema(
        schemas={SqliteIndexStore.LAYOUT.schema_version: schema} | recorded.schemas
    )
    RECORDED_SCHEMA.write_text(updated.model_dump_json(indent=RECORD_INDENT) + "\n")


def test_the_schema_s_version_names_what_the_store_keeps(tmp_path: Path):
    """An index holds the tables of one version of the store's schema, so a change to them or
    to what one is made from, the search text among them, needs a new version, whose output
    `record_schema` records; none recorded fails here."""
    chunks = RecordedChunks.model_validate_json(CHUNKS_RECORD.read_text()).chunks
    recorded = RecordedSchema.model_validate_json(RECORDED_SCHEMA.read_text())

    schema = stored_schema(chunks, tmp_path / "index")

    assert schema == recorded.schemas.get(SqliteIndexStore.LAYOUT.schema_version)


def test_a_write_that_fails_midway_leaves_the_index_as_it_was(tmp_path: Path):
    """A write is one transaction: the removal it began with is undone with it, and the
    caller learns why."""
    store = SqliteIndexStore(index_directory_of(SqliteIndexStore, tmp_path / "index", IDENTITY))
    kept = function_chunk("kept", "kept.py", "def kept(): pass")
    fill(store, [kept], vectors_of(TOWARD_QUERY))
    twice = function_chunk("twice", "twice.py", "def twice(): pass")
    # The same chunk twice fails at its second insert, after the removal ran
    update = IndexUpdate(
        files=[record_of("twice.py")],
        chunks=[twice, twice],
        removed_paths=[Path("kept.py")],
        last_of_run=True,
    )

    with pytest.raises(SqliteStoreError):
        store.write(update)

    assert (
        store.indexed_files().records,
        [chunk.name for chunk in matching(store, "kept")],
        store.counts(MODEL).chunks_with_vectors,
    ) == ([record_of("kept.py")], ["kept"], 1)


def test_a_word_a_question_repeats_counts_once(tmp_path: Path):
    """Each word of a question is one more term FTS5 scores every chunk by: a word repeated
    would weigh more, and cost as much again."""
    store = SqliteIndexStore(index_directory_of(SqliteIndexStore, tmp_path / "index", IDENTITY))
    chunks = [
        function_chunk("retrying", "retrying.py", "def retrying(): retry retry"),
        function_chunk("backing_off", "backing_off.py", "def backing_off(): " + "backoff " * 6),
        function_chunk("both", "both.py", "def both(): retry backoff once"),
    ]
    store.write(update_of(chunks, removed_paths=[]))

    once = [chunk.name for chunk in matching(store, "retry backoff")]
    repeated = [chunk.name for chunk in matching(store, "retry Retry retry RETRY retry backoff")]

    assert repeated == once


def session_store(project_directory: Path) -> SqliteIndexStore:
    """A store of the index, as a session's process opens it, with a directory of its own."""
    return SqliteIndexStore(index_directory_of(SqliteIndexStore, project_directory, IDENTITY))


def within_the_run(chunk: Chunk) -> IndexUpdate:
    """A write of a run that more writes follow."""
    return IndexUpdate(
        files=[record_of(chunk.path.as_posix())],
        chunks=[chunk],
        removed_paths=[],
        last_of_run=False,
    )


def test_a_first_build_shows_nothing_until_its_last_write_then_everything(tmp_path: Path):
    """A first index of words is built in a database of its own, without a journal, and
    renamed into place whole, readers beside its writer from then on (ADR 009, D12); its
    manifest comes with it, so no status names an index that is not there."""
    builder = session_store(tmp_path / "index")
    reader = session_store(tmp_path / "index")
    builder.try_hold_for_run()
    builder.write(within_the_run(function_chunk("first", "first.py", "def first(): pass")))
    during = (reader.counts(MODEL).chunks, reader.manifest())

    builder.write(update_of([function_chunk("last", "last.py", "def last(): pass")], []))
    builder.release_run()

    with sqlite3.connect(builder_database(tmp_path / "index")) as connection:
        mode = connection.execute(JOURNAL_MODE).fetchone()[0]
    assert (during, reader.counts(MODEL).chunks, reader.manifest() is None, mode) == (
        (0, None),
        2,
        False,
        WRITE_AHEAD_LOG_MODE,
    )


def test_a_first_build_stopped_midway_leaves_no_index_and_the_next_builds_whole(
    tmp_path: Path,
):
    """A cancelled first index, or a crash, leaves only its file, which the next deletes."""
    stopped = session_store(tmp_path / "index")
    stopped.try_hold_for_run()
    stopped.write(within_the_run(function_chunk("first", "first.py", "def first(): pass")))
    stopped.release_run()
    left = stopped.counts(MODEL).chunks
    next_build = session_store(tmp_path / "index")
    next_build.try_hold_for_run()

    next_build.write(update_of([function_chunk("last", "last.py", "def last(): pass")], []))
    next_build.release_run()

    assert (left, [chunk.name for chunk in matching(next_build, "def")]) == (0, ["last"])


# A database a torn write leaves, or a sync tool, or a hand: no database, and no tables
DAMAGES = [
    pytest.param(b"\x8f" * 65536, id="not a database"),
    pytest.param(b"", id="truncated to nothing"),
]


def damaged_index(project_directory: Path, damage: bytes) -> Path:
    """An index built, then its database damaged; the database's path."""
    built = session_store(project_directory)
    built.write(update_of([function_chunk("first", "first.py", "def first(): pass")], []))
    database = builder_database(project_directory)
    database.write_bytes(damage)
    return database


def set_aside_of(database: Path) -> list[Path]:
    return list(database.parent.glob(f"{DATABASE_FILE}{DAMAGED_SUFFIX}*"))


@pytest.mark.parametrize("damage", DAMAGES)
def test_a_read_of_a_damaged_database_finds_no_index_and_sets_the_file_aside(
    tmp_path: Path, damage: bytes
):
    """Otherwise every call would fail on it for good, as nothing would build it again."""
    database = damaged_index(tmp_path / "index", damage)

    chunks = session_store(tmp_path / "index").counts(MODEL).chunks

    assert (chunks, len(set_aside_of(database)), database.exists()) == (0, 1, False)


@pytest.mark.parametrize("damage", DAMAGES)
def test_a_run_that_meets_a_damaged_database_first_builds_the_index_whole(
    tmp_path: Path, damage: bytes
):
    database = damaged_index(tmp_path / "index", damage)
    store = session_store(tmp_path / "index")

    store.try_hold_for_run()
    store.write(update_of([function_chunk("rebuilt", "rebuilt.py", "def rebuilt(): pass")], []))
    store.release_run()

    assert ([chunk.name for chunk in matching(store, "def")], len(set_aside_of(database))) == (
        ["rebuilt"],
        1,
    )


def builder_database(project_directory: Path) -> Path:
    """The database file of the index in a project's index directory."""
    return index_directory_of(SqliteIndexStore, project_directory, IDENTITY).path / DATABASE_FILE
