"""Tests of what the SQLite store does of its own, beside the contract every store holds
(`test_index_store.py`): a write applied whole or not at all, and what its schema's version
names."""

import sqlite3
import tempfile
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from semantic_code_mcp.domain.models import Chunk, IndexUpdate
from semantic_code_mcp.storage.search_text import search_text
from semantic_code_mcp.storage.sqlite import DATABASE_FILE, SqliteIndexStore, SqliteStoreError
from tests.stores import (
    IDENTITY,
    TOWARD_QUERY,
    function_chunk,
    index_directory_of,
    nearest,
    record_of,
    update_of,
    vectors_of,
)

# The chunks LanceDB's recorded schema keeps, and what each version of this store's schema
# stored of them: each chunk's search text and the statements that made its tables
CHUNKS_RECORD = Path(__file__).parent.parent / "data" / "schema_outputs.json"
RECORDED_SCHEMA = Path(__file__).parent.parent / "data" / "sqlite_schema_outputs.json"
RECORD_INDENT = 2
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
    store.write(update_of(chunks, removed_paths=[]), vectors_of(*([TOWARD_QUERY] * len(chunks))))
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
    store.write(update_of([kept], removed_paths=[]), vectors_of(TOWARD_QUERY))
    twice = function_chunk("twice", "twice.py", "def twice(): pass")
    # The same chunk twice fails at its second insert, after the removal ran
    update = IndexUpdate(
        files=[record_of("twice.py")],
        chunks=[twice, twice],
        removed_paths=[Path("kept.py")],
        last_of_run=True,
    )

    with pytest.raises(SqliteStoreError):
        store.write(update, vectors_of(TOWARD_QUERY, TOWARD_QUERY))

    assert (
        store.indexed_files().records,
        [candidate.chunk.name for candidate in nearest(store)],
    ) == ([record_of("kept.py")], ["kept"])
