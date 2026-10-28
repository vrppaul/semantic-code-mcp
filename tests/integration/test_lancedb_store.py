"""Tests of what the LanceDB store does of its own, beside the contract every store holds
(`test_index_store.py`): its tables' schema, its upkeep of the full-text index, its fragments
and versions, and the order it lays a write in."""

import tempfile
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import lancedb
import numpy
import pytest
from pydantic import BaseModel, ConfigDict

from semantic_code_mcp.domain.models import (
    Chunk,
    IndexUpdate,
    SourceFile,
)
from semantic_code_mcp.storage import lancedb as lancedb_store
from semantic_code_mcp.storage.index_directories import IndexDirectory
from semantic_code_mcp.storage.lancedb import (
    CHUNKS_TABLE,
    FILES_TABLE,
    REPLACED_ROWS_SHARE_BEFORE_COMPACTION,
    ROWS_PER_FRAGMENT,
    UPKEEPS_PER_COMPACTION,
    LanceDBIndexStore,
)
from semantic_code_mcp.storage.search_text import search_text
from tests.lancedb_tables import (
    TABLE_INDEX_VERSIONS,
    chunks_fragments,
    chunks_outside_the_full_text_index,
    full_text_index_documents,
    newest_full_text_index_files,
)
from tests.stores import (
    ALTERNATING,
    AWAY_FROM_QUERY,
    IDENTITY,
    LIMIT,
    QUERY,
    TOWARD_QUERY,
    function_chunk,
    index_directory_of,
    query_of,
    record_of,
    update_of,
    vectors_of,
)

# A few chunks, and what each version of the store's schema stored of them: each chunk's
# search text and each table's columns
RECORDED_SCHEMA = Path(__file__).parent.parent / "data" / "schema_outputs.json"
RECORD_INDENT = 2
# A retention no version is young enough for, and one every version a test makes is: under
# the first upkeep prunes every version but the latest, under the second none
NO_RETENTION = timedelta(0)
LONGER_THAN_ANY_TEST = timedelta(hours=1)
# The version a table's first write makes
FIRST_VERSION = 1
# A table of a hundred chunks: a file of two that a test edits again and again, and the
# rest, one file each
OFTEN_EDITED_FILE = "often_edited.py"
OFTEN_EDITED_NAMES = ("often_first", "often_second")
OTHER_CHUNKS = 98
# How many edits of that file replace no more rows than upkeep lets weigh in the index
EDITS_UNDER_THE_SHARE = int(
    REPLACED_ROWS_SHARE_BEFORE_COMPACTION * (OTHER_CHUNKS + len(OFTEN_EDITED_NAMES))
) // len(OFTEN_EDITED_NAMES)
# Modes for a table's files: read and enter only, so a write to it fails; and writable again
READ_ONLY = 0o555
WRITABLE = 0o755


class StoredColumn(BaseModel):
    """A column of a table, by its name and its Arrow type."""

    model_config = ConfigDict(frozen=True)

    name: str
    type: str


class StoredTable(BaseModel):
    """A table of an index, by its name and its columns in their order."""

    model_config = ConfigDict(frozen=True)

    name: str
    columns: list[StoredColumn]


class StoredSchema(BaseModel):
    """What the store keeps of some chunks: the text its keyword channel searches of each,
    and its tables' columns."""

    model_config = ConfigDict(frozen=True)

    search_texts: list[str]
    tables: list[StoredTable]


class RecordedSchema(BaseModel):
    """Chunks, and what each version of the store's schema stored of them."""

    model_config = ConfigDict(frozen=True)

    chunks: list[Chunk]
    schemas: dict[int, StoredSchema]


def stored_schema(chunks: list[Chunk], index_directory: IndexDirectory) -> StoredSchema:
    """What the store keeps of the chunks once it has written them."""
    store = LanceDBIndexStore(index_directory)
    store.write(update_of(chunks, removed_paths=[]), vectors_of(*([TOWARD_QUERY] * len(chunks))))
    database = lancedb.connect(str(index_directory.path))
    tables = [
        StoredTable(
            name=name,
            columns=[
                StoredColumn(name=field.name, type=str(field.type))
                for field in database.open_table(name).schema
            ],
        )
        for name in (CHUNKS_TABLE, FILES_TABLE)
    ]
    return StoredSchema(search_texts=[search_text(chunk) for chunk in chunks], tables=tables)


def record_schema() -> None:
    """Record what the store keeps of the chunks under its schema's current version, unless
    that version is recorded already: it keeps what it stored, so a change of columns or of
    what one is made from needs a new version. Run it by hand from the repository's root,
    through `uv run python -c` and an import of this module."""
    recorded = RecordedSchema.model_validate_json(RECORDED_SCHEMA.read_text())
    with tempfile.TemporaryDirectory() as directory:
        schema = stored_schema(
            recorded.chunks, index_directory_of(LanceDBIndexStore, Path(directory), IDENTITY)
        )
    updated = RecordedSchema(
        chunks=recorded.chunks,
        schemas={LanceDBIndexStore.LAYOUT.schema_version: schema} | recorded.schemas,
    )
    RECORDED_SCHEMA.write_text(updated.model_dump_json(indent=RECORD_INDENT) + "\n")


@pytest.fixture
def index_directory(tmp_path: Path) -> IndexDirectory:
    return index_directory_of(LanceDBIndexStore, tmp_path / "index", IDENTITY)


@pytest.fixture
def store(index_directory: IndexDirectory) -> LanceDBIndexStore:
    return LanceDBIndexStore(index_directory)


@pytest.fixture
def filled_store(store: LanceDBIndexStore) -> LanceDBIndexStore:
    chunks = [
        function_chunk("near", "near.py", "def near(): return elapsed_time()"),
        function_chunk(
            "keyword",
            "indexer.py",
            'log.debug("timing", duration_ms=round(time.time() - started))',
        ),
        function_chunk("far", "far.py", "def far(): return None"),
    ]
    store.write(
        update_of(chunks, removed_paths=[]),
        vectors_of(TOWARD_QUERY, ALTERNATING, AWAY_FROM_QUERY),
    )
    return store


def test_the_schema_s_version_names_what_the_store_keeps(index_directory: IndexDirectory):
    """An index holds the tables of one version of the store's schema, so a change to their
    columns or to what one is made from, the search text among them, needs a new version,
    whose output `record_schema` records; none recorded fails here."""
    recorded = RecordedSchema.model_validate_json(RECORDED_SCHEMA.read_text())

    schema = stored_schema(recorded.chunks, index_directory)

    assert schema == recorded.schemas.get(LanceDBIndexStore.LAYOUT.schema_version)


def test_a_search_warns_of_nothing(
    filled_store: LanceDBIndexStore, capfd: pytest.CaptureFixture[str]
):
    """A gate on a library, not a behaviour of ours: LanceDB writes its warnings to the
    process's standard error, which an MCP client logs as the server's, so a release of it
    that warns on every search shows here."""
    capfd.readouterr()

    filled_store.search_candidates(QUERY, query_of("duration_ms"), LIMIT)

    assert capfd.readouterr().err == ""


def test_a_write_whose_chunks_fail_leaves_its_files_records_as_they_were(
    index_directory: IndexDirectory, filled_store: LanceDBIndexStore
):
    """Chunks go before records: a write stopped between them leaves the old records, which
    the next run reads as files changed, never new records over chunks that are not there."""
    chunks_table_directory = index_directory.path / f"{CHUNKS_TABLE}.lance"
    set_mode_of_tree(chunks_table_directory, READ_ONLY)
    edited = function_chunk("near", "near.py", "def near(): return elapsed_time() * 2")
    update = IndexUpdate(
        files=[SourceFile(path=Path("near.py"), mtime_ns=2, size_bytes=2)],
        chunks=[edited],
        removed_paths=[],
        last_of_run=True,
    )

    with pytest.raises((OSError, RuntimeError)):
        filled_store.write(update, vectors_of(TOWARD_QUERY))
    set_mode_of_tree(chunks_table_directory, WRITABLE)

    assert sorted(filled_store.indexed_files().records, key=lambda file: file.path) == [
        record_of("far.py"),
        record_of("indexer.py"),
        record_of("near.py"),
    ]


def test_a_run_s_last_write_whose_fold_fails_leaves_its_files_records_as_they_were(
    index_directory: IndexDirectory, filled_store: LanceDBIndexStore
):
    """The fold comes before the records: a write stopped there leaves the old records, which
    the next run reads as files changed, and writes and folds again."""
    indices_directory = index_directory.path / f"{CHUNKS_TABLE}.lance" / TABLE_INDEX_VERSIONS
    set_mode_of_tree(indices_directory, READ_ONLY)
    edited = function_chunk("near", "near.py", "def near(): return elapsed_time() * 2")
    update = IndexUpdate(
        files=[SourceFile(path=Path("near.py"), mtime_ns=2, size_bytes=2)],
        chunks=[edited],
        removed_paths=[],
        last_of_run=True,
    )

    with pytest.raises((OSError, RuntimeError)):
        filled_store.write(update, vectors_of(TOWARD_QUERY))
    set_mode_of_tree(indices_directory, WRITABLE)

    assert sorted(filled_store.indexed_files().records, key=lambda file: file.path) == [
        record_of("far.py"),
        record_of("indexer.py"),
        record_of("near.py"),
    ]


def test_a_file_s_new_chunks_take_the_place_of_its_old_ones_in_one_commit(
    index_directory: IndexDirectory, filled_store: LanceDBIndexStore
):
    """A reader sees the file's old chunks or its new ones, never neither."""
    chunks_table = lancedb.connect(str(index_directory.path)).open_table(CHUNKS_TABLE)
    version_before = chunks_table.version
    edited = function_chunk("near", "near.py", "def near(): return elapsed_time() * 2")
    # A write within its run, so no upkeep adds versions of its own
    update = IndexUpdate(
        files=[record_of("near.py")], chunks=[edited], removed_paths=[], last_of_run=False
    )

    filled_store.write(update, vectors_of(TOWARD_QUERY))

    chunks_table.checkout_latest()
    assert (chunks_table.version, filled_store.counts().chunks) == (version_before + 1, 3)


def test_chunks_with_no_record_come_back_as_unrecorded_paths(
    index_directory: IndexDirectory, filled_store: LanceDBIndexStore
):
    """A write stopped between a file's chunks and its record leaves them so."""
    files_table = lancedb.connect(str(index_directory.path)).open_table(FILES_TABLE)
    files_table.delete("path = 'near.py'")

    assert filled_store.indexed_files().unrecorded_paths == [Path("near.py")]


def test_a_run_s_last_write_folds_every_chunk_into_the_full_text_index(
    index_directory: IndexDirectory, filled_store: LanceDBIndexStore
):
    """The first write builds the full-text index whole; a later one adds rows it has not
    folded in, which score on another scale."""
    later = function_chunk("later", "later.py", "def later(): return retry_budget")

    filled_store.write(update_of([later], removed_paths=[]), vectors_of(TOWARD_QUERY))

    assert chunks_outside_the_full_text_index(index_directory) == 0


def set_mode_of_tree(directory: Path, mode: int) -> None:
    """Set the mode of a directory, everything under it, and their directories."""
    for path in [directory, *directory.rglob("*")]:
        path.chmod(mode)


def write_a_new_file(store: LanceDBIndexStore, number: int) -> None:
    """Write the numbered new file of one chunk, which the full-text index has not folded
    in; it replaces no row, so only the countdown, not the rows replaced, calls for upkeep's
    compaction."""
    added = function_chunk(f"added_{number}", f"added_{number}.py", f"def added_{number}(): pass")
    store.write(update_of([added], removed_paths=[]), vectors_of(TOWARD_QUERY))


def same_store_after_its_upkeeps(
    store: LanceDBIndexStore, _index_directory: IndexDirectory
) -> LanceDBIndexStore:
    """The store, after a run of one new file for each but the last of the upkeeps between
    two compactions."""
    for number in range(UPKEEPS_PER_COMPACTION - 1):
        write_a_new_file(store, number)
    return store


def store_of_the_next_session(
    store: LanceDBIndexStore, index_directory: IndexDirectory
) -> LanceDBIndexStore:
    """A new store on the same index, as the next session opens it, after the earlier
    session's store folded in a part."""
    write_a_new_file(store, 0)
    return LanceDBIndexStore(index_directory)


@pytest.mark.parametrize(
    "store_due_a_compaction",
    [
        pytest.param(same_store_after_its_upkeeps, id="the same store, once in a while"),
        pytest.param(store_of_the_next_session, id="a new store, at its first upkeep"),
    ],
)
def test_upkeep_rebuilds_the_full_text_index_before_its_parts_pile_up(
    index_directory: IndexDirectory,
    filled_store: LanceDBIndexStore,
    store_due_a_compaction: Callable[[LanceDBIndexStore, IndexDirectory], LanceDBIndexStore],
):
    """Each upkeep folds new rows in as a part of their own, which no later upkeep merges,
    so now and then it compacts and rebuilds the index whole. A store cannot know how many
    parts an earlier session left, so it does so first."""
    rebuilt = newest_full_text_index_files(index_directory)
    store = store_due_a_compaction(filled_store, index_directory)

    write_a_new_file(store, UPKEEPS_PER_COMPACTION - 1)

    assert newest_full_text_index_files(index_directory) == rebuilt


def test_an_upkeep_between_compactions_rewrites_no_fragment(
    index_directory: IndexDirectory, filled_store: LanceDBIndexStore
):
    """LanceDB's own optimize compacts at every call, merging every fragment under a million
    rows into one, so at 100k chunks each edit rewrote the whole table, in about 0.67 s."""
    write_a_new_file(filled_store, 0)
    kept = [fragment.fragment_id for fragment in chunks_fragments(index_directory)]
    later = function_chunk("later", "later.py", "def later(): pass")

    filled_store.write(update_of([later], removed_paths=[]), vectors_of(TOWARD_QUERY))

    fragments = [fragment.fragment_id for fragment in chunks_fragments(index_directory)]
    assert fragments[: len(kept)] == kept


def write_the_often_edited_file(store: LanceDBIndexStore, edit: int) -> None:
    """Write the numbered edit of the file a test edits again and again, of two chunks."""
    chunks = [
        function_chunk(name, OFTEN_EDITED_FILE, f"def {name}(): return {edit}")
        for name in OFTEN_EDITED_NAMES
    ]
    update = IndexUpdate(
        files=[record_of(OFTEN_EDITED_FILE)], chunks=chunks, removed_paths=[], last_of_run=True
    )
    store.write(update, vectors_of(*[TOWARD_QUERY] * len(chunks)))


@pytest.mark.parametrize(
    ("edits", "replaced_rows_counted"),
    [
        pytest.param(
            EDITS_UNDER_THE_SHARE,
            EDITS_UNDER_THE_SHARE * len(OFTEN_EDITED_NAMES),
            id="replaced rows under the share, kept",
        ),
        pytest.param(EDITS_UNDER_THE_SHARE + 1, 0, id="replaced rows past the share, dropped"),
    ],
)
def test_upkeep_compacts_once_replaced_rows_weigh_in_the_full_text_index(
    index_directory: IndexDirectory,
    store: LanceDBIndexStore,
    edits: int,
    replaced_rows_counted: int,
):
    """The full-text index counts the rows writes replaced in its word statistics until a
    rebuild: one file edited again and again moved answers of click's. Its rows sit in the
    fragment the last edit wrote, which Lance drops whole at the next, so only the index's
    own count shows them."""
    others = [
        function_chunk(f"other_{number}", f"other_{number}.py", f"def other_{number}(): pass")
        for number in range(OTHER_CHUNKS)
    ]
    store.write(update_of(others, removed_paths=[]), vectors_of(*[TOWARD_QUERY] * len(others)))
    for edit in range(edits):
        write_the_often_edited_file(store, edit)

    write_the_often_edited_file(store, edits)

    replaced = full_text_index_documents(index_directory) - store.counts().chunks
    assert replaced == replaced_rows_counted


def test_compaction_merges_small_fragments_and_leaves_full_ones(
    index_directory: IndexDirectory, store: LanceDBIndexStore
):
    """Each write leaves a fragment of its own, which compaction merges with its small
    neighbours folded into the indexes alike, up to ROWS_PER_FRAGMENT rows; a fragment that
    holds as many stays as it is, however large the table grows."""
    full = [
        function_chunk(f"full_{number}", f"full_{number}.py", f"def full_{number}(): pass")
        for number in range(ROWS_PER_FRAGMENT)
    ]
    store.write(update_of(full, removed_paths=[]), vectors_of(*[TOWARD_QUERY] * len(full)))
    full_fragment = chunks_fragments(index_directory)[0].fragment_id
    write_a_new_file(store, 0)
    later = function_chunk("later", "later.py", "def later(): pass")
    store.write(update_of([later], removed_paths=[]), vectors_of(TOWARD_QUERY))
    next_session = LanceDBIndexStore(index_directory)

    # A run that only ends: its upkeep, the session's first, compacts what earlier runs left
    next_session.write(update_of([], removed_paths=[]), numpy.empty((0, 0), dtype=numpy.float32))

    fragments = chunks_fragments(index_directory)
    assert (fragments[0].fragment_id, len(fragments)) == (full_fragment, 2)


def only_the_latest(latest: int) -> list[int]:
    """The versions upkeep leaves when none is young enough to keep: the latest alone."""
    return [latest]


def every_version(latest: int) -> list[int]:
    """The versions upkeep leaves when all are young enough to keep: each, from the first."""
    return list(range(FIRST_VERSION, latest + 1))


@pytest.mark.parametrize(
    ("retention", "versions_left"),
    [
        pytest.param(NO_RETENTION, only_the_latest, id="none young enough to keep"),
        pytest.param(LONGER_THAN_ANY_TEST, every_version, id="all young enough to keep"),
    ],
)
def test_upkeep_prunes_the_versions_older_than_any_read_can_last(
    index_directory: IndexDirectory,
    filled_store: LanceDBIndexStore,
    monkeypatch: pytest.MonkeyPatch,
    retention: timedelta,
    versions_left: Callable[[int], list[int]],
):
    """Else each write would keep a copy of what it replaced, and the index grow for ever;
    a version younger than the retention stays, as a read may still be on it."""
    monkeypatch.setattr(lancedb_store, "OLD_VERSION_RETENTION", retention)

    write_a_new_file(filled_store, 0)

    database = lancedb.connect(str(index_directory.path))
    tables = [database.open_table(name) for name in (CHUNKS_TABLE, FILES_TABLE)]
    assert [[entry["version"] for entry in table.list_versions()] for table in tables] == [
        versions_left(table.version) for table in tables
    ]
