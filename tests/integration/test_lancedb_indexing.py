"""Tests of what an index run leaves in the LanceDB store of its own, beside what it leaves in
every store (`test_indexing.py`): the run's chunks folded into the full-text index, a run
cancelled as its last write lands included, the files a cancelled first index wrote, which
SQLite's whole first build does not keep, and the chunks a write stopped before their records
left behind, which a store writing in one transaction never leaves."""

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

import lancedb
import pytest

from semantic_code_mcp.app import build_chunker
from semantic_code_mcp.config import Settings
from semantic_code_mcp.domain.models import IndexCounts
from semantic_code_mcp.services.indexing import IndexingService
from semantic_code_mcp.storage.index_directories import IndexDirectory
from semantic_code_mcp.storage.lancedb import FILES_TABLE, LanceDBIndexStore
from tests.fakes import FAKE_MODEL, StoreCancelledAfterItsWrite, StoreHeldInItsWrite
from tests.helpers import indexing_service
from tests.indexing import (
    ADDED_FUNCTION,
    FILES_OF_FUNCTIONS,
    FIRST_MODULE,
    FUNCTIONS_PER_FILE,
    IDENTITY,
    NO_PROGRESS,
    first_index,
    functions_in_files,
    touch_later,
)
from tests.lancedb_tables import chunks_outside_the_full_text_index
from tests.stores import index_directory_of

# A filter every row of a table matches
MATCH_EVERY_ROW = "true"
# How long a test waits for a held write to begin and to land, far beyond either
HELD_WRITE_TIMEOUT_SECONDS = 30.0


@pytest.fixture
def index_directory(tmp_path: Path) -> IndexDirectory:
    return index_directory_of(LanceDBIndexStore, tmp_path / "index", IDENTITY)


@pytest.fixture
def store(index_directory: IndexDirectory) -> LanceDBIndexStore:
    return LanceDBIndexStore(index_directory)


def delete_records(index_directory: IndexDirectory, condition: str) -> None:
    """Drop the records of the files a filter matches, as a write stopped between those
    files' chunks and their records leaves the index."""
    database = lancedb.connect(str(index_directory.path))
    database.open_table(FILES_TABLE).delete(condition)


async def test_files_whose_records_were_lost_are_indexed_again_without_doubled_chunks(
    test_settings: Settings,
    sample_project: Path,
    index_directory: IndexDirectory,
    store: LanceDBIndexStore,
):
    indexing = indexing_service(store, test_settings, build_chunker())
    await indexing.index(sample_project, force=False, progress=NO_PROGRESS)
    delete_records(index_directory, MATCH_EVERY_ROW)

    report = await indexing.index(sample_project, force=False, progress=NO_PROGRESS)

    assert (report.files_indexed, store.counts(FAKE_MODEL)) == (
        2,
        IndexCounts(files=2, chunks=6, chunks_with_vectors=0),
    )


async def test_chunks_of_a_file_deleted_before_its_record_was_written_leave_the_index(
    test_settings: Settings,
    sample_project: Path,
    index_directory: IndexDirectory,
    store: LanceDBIndexStore,
):
    """A write that stopped between a new file's chunks and its record left them behind."""
    (sample_project / "gone.py").write_text("def gone():\n    return 0\n")
    indexing = indexing_service(store, test_settings, build_chunker())
    await indexing.index(sample_project, force=False, progress=NO_PROGRESS)
    delete_records(index_directory, "path = 'gone.py'")
    (sample_project / "gone.py").unlink()

    status = await indexing.status(sample_project)
    await indexing.index(sample_project, force=False, progress=NO_PROGRESS)

    assert (status.stale_files_count, store.counts(FAKE_MODEL)) == (
        1,
        IndexCounts(files=2, chunks=6, chunks_with_vectors=0),
    )


async def forced_run(indexing: IndexingService, project: Path) -> None:
    """A first index, then a forced run, which writes every row again."""
    await first_index(indexing, project)
    await indexing.index(project, force=True, progress=NO_PROGRESS)


async def edit_of_one_file(indexing: IndexingService, project: Path) -> None:
    """A first index, then a run after one file was edited, which writes that file only."""
    await first_index(indexing, project)
    edited = project / FIRST_MODULE
    edited.write_text(edited.read_text() + ADDED_FUNCTION)
    touch_later(edited)
    await indexing.index(project, force=False, progress=NO_PROGRESS)


@pytest.mark.parametrize(
    "runs",
    [
        pytest.param(first_index, id="a first index of several writes"),
        pytest.param(forced_run, id="a forced run"),
        pytest.param(edit_of_one_file, id="an edit of one file"),
    ],
)
async def test_a_run_that_wrote_leaves_every_chunk_in_the_full_text_index(
    test_settings: Settings,
    tmp_path: Path,
    index_directory: IndexDirectory,
    store: LanceDBIndexStore,
    runs: Callable[[IndexingService, Path], Awaitable[None]],
):
    """Rows the full-text index has not folded in score on another scale, so the chunks a
    run wrote, the ones an agent asks about next, would rank apart until upkeep."""
    project = functions_in_files(tmp_path, FILES_OF_FUNCTIONS, FUNCTIONS_PER_FILE)
    indexing = indexing_service(store, test_settings, build_chunker())

    await runs(indexing, project)

    assert chunks_outside_the_full_text_index(index_directory) == 0


async def test_a_run_cancelled_while_its_last_write_lands_leaves_every_chunk_in_the_full_text_index(
    test_settings: Settings,
    tmp_path: Path,
    index_directory: IndexDirectory,
    store: LanceDBIndexStore,
):
    """A client that hangs up as a run writes: the write still lands in its thread, and the
    chunks it wrote must score on the full-text index's scale, not another."""
    project = functions_in_files(tmp_path, FILES_OF_FUNCTIONS, FUNCTIONS_PER_FILE)
    await first_index(indexing_service(store, test_settings, build_chunker()), project)
    edited = project / FIRST_MODULE
    edited.write_text(edited.read_text() + ADDED_FUNCTION)
    touch_later(edited)
    held = StoreHeldInItsWrite(store)
    indexing = indexing_service(held, test_settings, build_chunker())
    run = asyncio.create_task(indexing.index(project, force=False, progress=NO_PROGRESS))

    write_began = await asyncio.to_thread(held.writing.wait, HELD_WRITE_TIMEOUT_SECONDS)
    run.cancel()
    held.release.set()
    with pytest.raises(asyncio.CancelledError):
        await run
    write_landed = await asyncio.to_thread(held.written.wait, HELD_WRITE_TIMEOUT_SECONDS)

    assert (write_began, write_landed, chunks_outside_the_full_text_index(index_directory)) == (
        True,
        True,
        0,
    )


async def test_a_cancelled_first_index_keeps_the_files_it_wrote(
    test_settings: Settings, tmp_path: Path, store: LanceDBIndexStore
):
    project = functions_in_files(tmp_path, FILES_OF_FUNCTIONS, FUNCTIONS_PER_FILE)
    cancelled = StoreCancelledAfterItsWrite(store)
    indexing = indexing_service(cancelled, test_settings, build_chunker())

    with pytest.raises(asyncio.CancelledError):
        await indexing.index(project, force=False, progress=NO_PROGRESS)

    assert store.counts(FAKE_MODEL) == IndexCounts(
        files=2, chunks=2 * FUNCTIONS_PER_FILE, chunks_with_vectors=0
    )
