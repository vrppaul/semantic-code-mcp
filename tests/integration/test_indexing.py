"""Tests for indexing a project into a real LanceDB store."""

import asyncio
import os
import threading
from collections.abc import Awaitable, Callable
from functools import partial
from pathlib import Path

import lancedb
import numpy
import pytest

from semantic_code_mcp.app import (
    WRITE_LOCK_TIMEOUT_SECONDS,
    build_chunker,
    build_registry,
    open_index_store,
)
from semantic_code_mcp.chunkers.composite import CHUNKER_VERSION
from semantic_code_mcp.config import LOCAL_INDEX_DIRECTORY, Settings, get_index_path
from semantic_code_mcp.domain.interfaces import Chunker
from semantic_code_mcp.domain.models import IndexCounts, IndexIdentity
from semantic_code_mcp.embedding import SentenceTransformerEmbedder
from semantic_code_mcp.services.indexing import EMBEDDING_SLICE_SIZE, IndexingService
from semantic_code_mcp.services.progress import ProgressReporter
from semantic_code_mcp.storage.lancedb import FILES_TABLE, LEGACY_ENTRIES, NO_CANDIDATES
from tests.conftest import MODEL_NAME
from tests.fakes import (
    EMBEDDING_DIMENSION,
    FAKE_MODEL_NAME,
    ChangingChunker,
    ChunkCountRecorder,
    EmbedderWatchingRecorder,
    FakeEmbedder,
    FakeIndexStore,
    ProgressRecorder,
    StoreCancelledAfterItsWrite,
    VersionedChunker,
    vector_of,
)
from tests.helpers import (
    chunks_outside_the_full_text_index,
    commit_all,
    git_output,
    indexing_service,
    make_repository,
    registry_of,
)

NO_PROGRESS = ProgressReporter(None)
# An mtime well after any file the test just wrote
LATER_MTIME_SECONDS = 4_000_000_000
# Functions in one file that fill two slices of chunks to embed, and a part of a third
TWO_AND_A_PART_SLICES = 2 * EMBEDDING_SLICE_SIZE + 10
# A vector width other than the fake model's, so a change of model changes the width too
WIDE = 16
# 0.4's record of mtimes, which it kept in a project-local index directory
LEGACY_RECORDS = LEGACY_ENTRIES[1]
# A filter every row of a table matches
MATCH_EVERY_ROW = "true"
# A function added to a module of a project
ADDED_FUNCTION = "\n\ndef added():\n    return 1\n"
# Files whose functions fill one write with the first two and a second with the third
FILES_OF_FUNCTIONS = 3
FUNCTIONS_PER_FILE = EMBEDDING_SLICE_SIZE * 3 // 4
# The names of those files, and the first of them
MODULE_NAME_FORMAT = "module_{file_number}.py"
FIRST_MODULE = MODULE_NAME_FORMAT.format(file_number=0)
# What the fakes and the chunkers in these tests build an index from
IDENTITY = IndexIdentity(
    model_name=FAKE_MODEL_NAME, model_revision=None, chunker_version=CHUNKER_VERSION
)


def touch_later(path: Path) -> None:
    """An edit: the file's mtime moves on."""
    os.utime(path, (LATER_MTIME_SECONDS, LATER_MTIME_SECONDS))


def delete(path: Path) -> None:
    path.unlink(missing_ok=True)


@pytest.fixture
def indexing(test_settings: Settings, sample_project: Path) -> IndexingService:
    return build_registry(test_settings, FakeEmbedder()).services_for(sample_project).indexing


async def test_index_stores_every_chunk_of_every_file(
    indexing: IndexingService, sample_project: Path
):
    report = await indexing.index(sample_project, force=False, progress=NO_PROGRESS)
    status = await indexing.status(sample_project)

    assert (report.files_indexed, report.chunks_indexed) == (2, 6)
    assert (status.is_indexed, status.files_count, status.chunks_count) == (True, 2, 6)


async def test_second_run_without_changes_indexes_nothing(
    indexing: IndexingService, sample_project: Path
):
    await indexing.index(sample_project, force=False, progress=NO_PROGRESS)

    second_run = await indexing.index(sample_project, force=False, progress=NO_PROGRESS)

    assert second_run.files_indexed == 0


async def test_a_changed_file_is_indexed_again_alone(
    indexing: IndexingService, sample_project: Path
):
    await indexing.index(sample_project, force=False, progress=NO_PROGRESS)
    changed = sample_project / "main.py"
    changed.write_text(changed.read_text() + "\n\ndef added():\n    return 1\n")
    os.utime(changed, (LATER_MTIME_SECONDS, LATER_MTIME_SECONDS))

    report = await indexing.index(sample_project, force=False, progress=NO_PROGRESS)
    status = await indexing.status(sample_project)

    assert (report.files_indexed, report.chunks_indexed) == (1, 4)
    assert status.chunks_count == 7


@pytest.mark.parametrize(
    "change",
    [pytest.param(touch_later, id="a file edited"), pytest.param(delete, id="a file deleted")],
)
async def test_a_file_changed_while_it_is_indexed_stays_stale(
    test_settings: Settings,
    sample_project: Path,
    tmp_path: Path,
    change: Callable[[Path], object],
):
    """The change lands after the scan took the files' records, so the next run sees it."""
    indexing = indexing_service(
        open_index_store(tmp_path / "index", IDENTITY),
        FakeEmbedder(),
        test_settings,
        ChangingChunker(build_chunker(), partial(change, sample_project / "utils.py")),
    )

    await indexing.index(sample_project, force=False, progress=NO_PROGRESS)
    status = await indexing.status(sample_project)

    assert status.stale_files_count == 1


async def test_force_indexes_every_file_again(indexing: IndexingService, sample_project: Path):
    await indexing.index(sample_project, force=False, progress=NO_PROGRESS)

    report = await indexing.index(sample_project, force=True, progress=NO_PROGRESS)
    status = await indexing.status(sample_project)

    assert report.files_indexed == 2
    assert status.chunks_count == 6


async def test_a_forced_run_keeps_the_old_index_until_its_new_chunks_are_written(
    test_settings: Settings, sample_project: Path, tmp_path: Path
):
    store = open_index_store(tmp_path / "index", IDENTITY)
    indexing = indexing_service(store, FakeEmbedder(), test_settings, build_chunker())
    await indexing.index(sample_project, force=False, progress=NO_PROGRESS)
    recorder = ChunkCountRecorder(store)

    await indexing.index(sample_project, force=True, progress=ProgressReporter(recorder))

    assert set(recorder.chunks_at_reports) == {6}


async def test_a_file_deleted_after_indexing_makes_the_index_stale(
    indexing: IndexingService, sample_project: Path
):
    await indexing.index(sample_project, force=False, progress=NO_PROGRESS)
    (sample_project / "utils.py").unlink()

    status = await indexing.status(sample_project)

    assert status.stale_files_count == 1


async def test_a_deleted_file_leaves_the_index(indexing: IndexingService, sample_project: Path):
    await indexing.index(sample_project, force=False, progress=NO_PROGRESS)
    (sample_project / "utils.py").unlink()

    report = await indexing.index(sample_project, force=False, progress=NO_PROGRESS)
    status = await indexing.status(sample_project)

    assert report.files_deleted == 1
    assert (status.files_count, status.chunks_count) == (1, 3)


@pytest.mark.parametrize(
    ("second_embedder", "second_chunker"),
    [
        pytest.param(
            FakeEmbedder(model_name="second-model", dimension=WIDE),
            build_chunker(),
            id="another model, of another width",
        ),
        pytest.param(
            FakeEmbedder(revision="another-revision"),
            build_chunker(),
            id="the same model at another revision",
        ),
        pytest.param(
            FakeEmbedder(),
            VersionedChunker(build_chunker(), CHUNKER_VERSION + 1),
            id="another chunker version",
        ),
    ],
)
async def test_sessions_of_two_models_or_chunker_versions_keep_an_index_each(
    test_settings: Settings,
    sample_project: Path,
    second_embedder: FakeEmbedder,
    second_chunker: Chunker,
):
    """Neither builds the other's index again: the first, run again, finds nothing to do."""
    first = registry_of(test_settings, FakeEmbedder(), build_chunker())
    second = registry_of(test_settings, second_embedder, second_chunker)
    first_indexing = first.services_for(sample_project).indexing
    await first_indexing.index(sample_project, force=False, progress=NO_PROGRESS)

    second_run = await second.services_for(sample_project).indexing.index(
        sample_project, force=False, progress=NO_PROGRESS
    )
    first_again = await first_indexing.index(sample_project, force=False, progress=NO_PROGRESS)

    assert (second_run.files_indexed, first_again.files_indexed) == (2, 0)


async def test_two_runs_at_once_store_each_chunk_once(
    indexing: IndexingService, sample_project: Path
):
    """Parallel tool calls on one project must not write its chunks twice."""
    await asyncio.gather(
        indexing.index(sample_project, force=False, progress=NO_PROGRESS),
        indexing.index(sample_project, force=False, progress=NO_PROGRESS),
    )
    status = await indexing.status(sample_project)

    assert status.chunks_count == 6


async def test_a_file_that_is_not_utf8_does_not_stop_the_rest(
    indexing: IndexingService, sample_project: Path
):
    (sample_project / "latin1.py").write_bytes(b"# caf\xe9\ndef order():\n    return 1\n")

    report = await indexing.index(sample_project, force=False, progress=NO_PROGRESS)

    assert report.files_indexed == 3


async def test_tracked_file_deleted_from_disk_is_not_indexed_again(
    indexing: IndexingService, sample_project: Path
):
    make_repository(sample_project)
    commit_all(sample_project)
    (sample_project / "utils.py").unlink()
    await indexing.index(sample_project, force=False, progress=NO_PROGRESS)

    second_run = await indexing.index(sample_project, force=False, progress=NO_PROGRESS)

    assert second_run.files_indexed == 0


async def test_a_local_index_never_shows_in_the_project_s_git_status(
    sample_project: Path, tmp_path: Path
):
    """Not even what 0.4 left in it, untracked, before its removal a week later."""
    make_repository(sample_project)
    commit_all(sample_project)
    (sample_project / LOCAL_INDEX_DIRECTORY).mkdir()
    (sample_project / LOCAL_INDEX_DIRECTORY / LEGACY_RECORDS).write_text("{}")
    settings = Settings(cache_dir=tmp_path / "cache", local_index=True)
    indexing = build_registry(settings, FakeEmbedder()).services_for(sample_project).indexing

    await indexing.index(sample_project, force=False, progress=NO_PROGRESS)

    assert git_output(sample_project, "status", "--porcelain") == ""


async def test_a_project_moved_with_its_local_index_keeps_it(sample_project: Path, tmp_path: Path):
    settings = Settings(cache_dir=tmp_path / "cache", local_index=True)
    first = build_registry(settings, FakeEmbedder()).services_for(sample_project).indexing
    await first.index(sample_project, force=False, progress=NO_PROGRESS)
    moved = await asyncio.to_thread(sample_project.rename, tmp_path / "moved")

    status = (
        await build_registry(settings, FakeEmbedder()).services_for(moved).indexing.status(moved)
    )

    assert (status.chunks_count, status.stale_files_count) == (6, 0)


async def test_progress_only_moves_forward(indexing: IndexingService, sample_project: Path):
    recorder = ProgressRecorder()

    await indexing.index(sample_project, force=False, progress=ProgressReporter(recorder))

    assert recorder.progress_values
    assert recorder.progress_values == sorted(recorder.progress_values)


async def test_status_of_a_project_never_indexed_leaves_no_trace(
    test_settings: Settings, tmp_path: Path
):
    project = tmp_path / "fresh"
    project.mkdir()
    (project / "code.py").write_text("def code():\n    return 1\n")
    indexing = build_registry(test_settings, FakeEmbedder()).services_for(project).indexing

    status = await indexing.status(project)

    assert (status.is_indexed, status.files_count, status.stale_files_count) == (False, 0, 1)
    assert not get_index_path(test_settings, project).exists()


async def test_status_during_a_forced_run_never_fails(
    indexing: IndexingService, sample_project: Path
):
    """A forced run replaces the index while readers ask for status in parallel."""
    await indexing.index(sample_project, force=False, progress=NO_PROGRESS)

    statuses = []
    for _ in range(5):
        _, *round_statuses = await asyncio.gather(
            indexing.index(sample_project, force=True, progress=NO_PROGRESS),
            indexing.status(sample_project),
            indexing.status(sample_project),
        )
        statuses.extend(round_statuses)

    # A reader sees the old index or the new one, or for a moment none, never an error
    assert {status.chunks_count for status in statuses} <= {0, 6}


async def test_a_project_without_indexable_files_indexes_nothing(
    test_settings: Settings, tmp_path: Path
):
    project = tmp_path / "notes"
    project.mkdir()
    (project / "notes.txt").write_text("nothing to index\n")
    indexing = build_registry(test_settings, FakeEmbedder()).services_for(project).indexing

    report = await indexing.index(project, force=False, progress=NO_PROGRESS)
    status = await indexing.status(project)

    assert (report.files_indexed, status.is_indexed) == (0, False)


async def test_status_does_not_load_the_model(test_settings: Settings, sample_project: Path):
    embedder = SentenceTransformerEmbedder(MODEL_NAME)
    indexing = build_registry(test_settings, embedder).services_for(sample_project).indexing

    await indexing.status(sample_project)

    assert embedder.is_loaded is False


async def test_the_model_loads_and_embeds_off_the_event_loop(
    test_settings: Settings, sample_project: Path
):
    embedder = FakeEmbedder(loaded=False)
    indexing = indexing_service(
        FakeIndexStore(NO_CANDIDATES), embedder, test_settings, build_chunker()
    )

    await indexing.index(sample_project, force=False, progress=NO_PROGRESS)

    assert embedder.calling_threads
    assert threading.main_thread() not in embedder.calling_threads


async def test_the_model_load_is_reported_before_it_happens(
    test_settings: Settings, sample_project: Path
):
    embedder = FakeEmbedder(loaded=False)
    recorder = EmbedderWatchingRecorder(embedder)
    indexing = build_registry(test_settings, embedder).services_for(sample_project).indexing

    await indexing.index(sample_project, force=False, progress=ProgressReporter(recorder))

    # Scanning, planning, chunking, loading the model, embedding the one slice
    assert [snapshot.model_loaded for snapshot in recorder.snapshots] == [
        False,
        False,
        False,
        False,
        True,
    ]


@pytest.mark.parametrize(
    ("functions", "embedded_at_reports"),
    [
        pytest.param(
            TWO_AND_A_PART_SLICES,
            [0, 0, 0, 0, EMBEDDING_SLICE_SIZE, 2 * EMBEDDING_SLICE_SIZE],
            id="the last slice in part",
        ),
        pytest.param(
            2 * EMBEDDING_SLICE_SIZE,
            [0, 0, 0, 0, EMBEDDING_SLICE_SIZE],
            id="whole slices",
        ),
    ],
)
async def test_progress_is_reported_before_each_slice_of_chunks_is_embedded(
    test_settings: Settings, tmp_path: Path, functions: int, embedded_at_reports: list[int]
):
    """Reports come at scanning, planning, chunking, and before each slice."""
    project = many_functions_project(tmp_path, functions)
    embedder = FakeEmbedder()
    recorder = EmbedderWatchingRecorder(embedder)
    indexing = build_registry(test_settings, embedder).services_for(project).indexing

    await indexing.index(project, force=False, progress=ProgressReporter(recorder))

    assert [snapshot.texts_embedded for snapshot in recorder.snapshots] == embedded_at_reports


async def test_each_chunk_is_stored_with_its_own_vector_across_slices(
    test_settings: Settings, tmp_path: Path
):
    project = many_functions_project(tmp_path, TWO_AND_A_PART_SLICES)
    store = FakeIndexStore(NO_CANDIDATES)
    indexing = indexing_service(store, FakeEmbedder(), test_settings, build_chunker())

    await indexing.index(project, force=False, progress=NO_PROGRESS)

    chunks = [chunk for chunks in store.chunks_by_file.values() for chunk in chunks]
    stored = numpy.stack(
        [vector for vectors in store.vectors_by_file.values() for vector in vectors]
    )
    expected = numpy.stack([vector_of(chunk.content, EMBEDDING_DIMENSION) for chunk in chunks])
    assert numpy.array_equal(stored, expected)


def many_functions_project(root: Path, functions: int) -> Path:
    """A project of one file holding a number of small functions, one chunk each."""
    project = root / "many"
    project.mkdir()
    (project / "many.py").write_text(
        "".join(
            f"def function_{number}():\n    return {number}\n\n\n" for number in range(functions)
        )
    )
    return project


def test_one_project_has_one_set_of_services(test_settings: Settings, sample_project: Path):
    registry = build_registry(test_settings, FakeEmbedder())

    assert registry.services_for(sample_project) is registry.services_for(sample_project)


async def test_files_whose_records_were_lost_are_indexed_again_without_doubled_chunks(
    test_settings: Settings, sample_project: Path, tmp_path: Path
):
    store = open_index_store(tmp_path / "index", IDENTITY)
    indexing = indexing_service(store, FakeEmbedder(), test_settings, build_chunker())
    await indexing.index(sample_project, force=False, progress=NO_PROGRESS)
    lancedb.connect(str(store.directory)).open_table(FILES_TABLE).delete(MATCH_EVERY_ROW)

    report = await indexing.index(sample_project, force=False, progress=NO_PROGRESS)

    assert (report.files_indexed, store.counts()) == (2, IndexCounts(files=2, chunks=6))


async def test_an_edited_file_embeds_only_its_chunks_whose_text_changed(
    test_settings: Settings, sample_project: Path, tmp_path: Path
):
    store = open_index_store(tmp_path / "index", IDENTITY)
    await indexing_service(store, FakeEmbedder(), test_settings, build_chunker()).index(
        sample_project, force=False, progress=NO_PROGRESS
    )
    edited = sample_project / "main.py"
    edited.write_text(edited.read_text() + ADDED_FUNCTION)
    touch_later(edited)
    embedder = FakeEmbedder()

    await indexing_service(store, embedder, test_settings, build_chunker()).index(
        sample_project, force=False, progress=NO_PROGRESS
    )

    assert embedder.embedded_texts == [ADDED_FUNCTION.strip()]


async def test_a_forced_run_embeds_every_chunk_again(
    test_settings: Settings, sample_project: Path, tmp_path: Path
):
    """A model replaced under a name nothing pins gives other vectors that nothing else
    can see."""
    store = open_index_store(tmp_path / "index", IDENTITY)
    await indexing_service(store, FakeEmbedder(), test_settings, build_chunker()).index(
        sample_project, force=False, progress=NO_PROGRESS
    )
    embedder = FakeEmbedder()

    await indexing_service(store, embedder, test_settings, build_chunker()).index(
        sample_project, force=True, progress=NO_PROGRESS
    )

    assert len(embedder.embedded_texts) == 6


def add_a_function(path: Path) -> None:
    path.write_text(path.read_text() + ADDED_FUNCTION)
    touch_later(path)


async def test_an_edit_after_a_model_of_another_width_took_its_name_is_embedded_anew(
    test_settings: Settings, sample_project: Path, tmp_path: Path
):
    """A name nothing pins, as a local path's: the file's unchanged chunks hold vectors of
    the old width, so none is reused, each chunk is embedded once, and the index starts
    again from the file, the other one unrecorded."""
    store = open_index_store(tmp_path / "index", IDENTITY)
    await indexing_service(store, FakeEmbedder(), test_settings, build_chunker()).index(
        sample_project, force=False, progress=NO_PROGRESS
    )
    add_a_function(sample_project / "main.py")
    wider_embedder = FakeEmbedder(dimension=WIDE)
    wider = indexing_service(store, wider_embedder, test_settings, build_chunker())

    await wider.index(sample_project, force=False, progress=NO_PROGRESS)
    status = await wider.status(sample_project)

    assert (store.counts(), status.stale_files_count, len(wider_embedder.embedded_texts)) == (
        IndexCounts(files=1, chunks=4),
        1,
        4,
    )


async def test_progress_of_a_run_that_cannot_reuse_a_slice_s_vectors_only_moves_forward(
    test_settings: Settings, tmp_path: Path
):
    """The chunks new to a file fill more than one batch, and its old chunks' vectors are
    of another width than the model's now: they are embedded after the new ones."""
    project = functions_in_files(tmp_path, files=1, functions=EMBEDDING_SLICE_SIZE)
    store = open_index_store(tmp_path / "index", IDENTITY)
    await indexing_service(store, FakeEmbedder(), test_settings, build_chunker()).index(
        project, force=False, progress=NO_PROGRESS
    )
    module = next(project.iterdir())
    module.write_text(module.read_text() + "".join(added_functions(TWO_AND_A_PART_SLICES)))
    touch_later(module)
    recorder = ProgressRecorder()
    wider = indexing_service(store, FakeEmbedder(dimension=WIDE), test_settings, build_chunker())

    await wider.index(project, force=False, progress=ProgressReporter(recorder))

    assert recorder.progress_values == sorted(recorder.progress_values)


def added_functions(count: int) -> list[str]:
    """Functions of names no module of `functions_in_files` holds."""
    return [f"def added_{number}():\n    return {number}\n\n\n" for number in range(count)]


async def test_a_cancelled_first_index_keeps_the_files_it_wrote(
    test_settings: Settings, tmp_path: Path
):
    project = functions_in_files(tmp_path, FILES_OF_FUNCTIONS, FUNCTIONS_PER_FILE)
    store = StoreCancelledAfterItsWrite(
        tmp_path / "index", IDENTITY, lock_timeout_seconds=WRITE_LOCK_TIMEOUT_SECONDS
    )
    indexing = indexing_service(store, FakeEmbedder(), test_settings, build_chunker())

    with pytest.raises(asyncio.CancelledError):
        await indexing.index(project, force=False, progress=NO_PROGRESS)

    assert store.counts() == IndexCounts(files=2, chunks=2 * FUNCTIONS_PER_FILE)


async def first_index(indexing: IndexingService, project: Path) -> None:
    """A first index of the project, in several writes."""
    await indexing.index(project, force=False, progress=NO_PROGRESS)


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
    runs: Callable[[IndexingService, Path], Awaitable[None]],
):
    """Rows the full-text index has not folded in score on another scale, so the chunks a
    run wrote, the ones an agent asks about next, would rank apart until upkeep."""
    project = functions_in_files(tmp_path, FILES_OF_FUNCTIONS, FUNCTIONS_PER_FILE)
    store = open_index_store(tmp_path / "index", IDENTITY)
    indexing = indexing_service(store, FakeEmbedder(), test_settings, build_chunker())

    await runs(indexing, project)

    assert chunks_outside_the_full_text_index(store) == 0


async def test_chunks_of_a_file_deleted_before_its_record_was_written_leave_the_index(
    test_settings: Settings, sample_project: Path, tmp_path: Path
):
    """A write that stopped between a new file's chunks and its record left them behind."""
    (sample_project / "gone.py").write_text("def gone():\n    return 0\n")
    store = open_index_store(tmp_path / "index", IDENTITY)
    indexing = indexing_service(store, FakeEmbedder(), test_settings, build_chunker())
    await indexing.index(sample_project, force=False, progress=NO_PROGRESS)
    lancedb.connect(str(store.directory)).open_table(FILES_TABLE).delete("path = 'gone.py'")
    (sample_project / "gone.py").unlink()

    status = await indexing.status(sample_project)
    await indexing.index(sample_project, force=False, progress=NO_PROGRESS)

    assert (status.stale_files_count, store.counts()) == (1, IndexCounts(files=2, chunks=6))


def functions_in_files(root: Path, files: int, functions: int) -> Path:
    """A project of several files, each holding a number of small functions, one chunk each."""
    project = root / "functions"
    project.mkdir()
    for file_number in range(files):
        (project / MODULE_NAME_FORMAT.format(file_number=file_number)).write_text(
            "".join(
                f"def function_{file_number}_{number}():\n    return {number}\n\n\n"
                for number in range(functions)
            )
        )
    return project
