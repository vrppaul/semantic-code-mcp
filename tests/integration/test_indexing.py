"""Tests for indexing a project into a real LanceDB store."""

import asyncio
import json
import os
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import lancedb
import numpy
import pyarrow
import pytest

from semantic_code_mcp.app import build_chunker, build_registry, open_index_store
from semantic_code_mcp.chunkers.composite import CHUNKER_VERSION
from semantic_code_mcp.config import Settings, get_index_path
from semantic_code_mcp.domain.interfaces import Chunker
from semantic_code_mcp.domain.models import (
    IndexIdentity,
    IndexManifest,
    IndexUpdate,
    RebuildReason,
    WriteScope,
)
from semantic_code_mcp.embedding import SentenceTransformerEmbedder
from semantic_code_mcp.services.indexing import (
    EMBEDDING_SLICE_SIZE,
    NO_VECTORS_SHAPE,
    IndexingService,
    IndexPhase,
)
from semantic_code_mcp.services.progress import ProgressReporter
from semantic_code_mcp.storage.lancedb import FILES_TABLE, MATCH_EVERY_ROW, SCHEMA_VERSION
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
    VersionedChunker,
    vector_of,
)
from tests.helpers import commit_all, git_output, indexing_service, make_repository

NO_PROGRESS = ProgressReporter(None)
# An mtime well after any file the test just wrote
LATER_MTIME_SECONDS = 4_000_000_000
# Functions in one file that fill two slices of chunks to embed, and a part of a third
TWO_AND_A_PART_SLICES = 2 * EMBEDDING_SLICE_SIZE + 10
# Vector widths of two models, so a change of model changes the vectors' width too
NARROW = 8
WIDE = 16
# What 0.4 kept in an index directory: its chunks table, and its record of file mtimes
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
        open_index_store(tmp_path / "index"),
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
    store = open_index_store(tmp_path / "index")
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
    ("first_embedder", "second_embedder", "second_chunker", "reason"),
    [
        pytest.param(
            FakeEmbedder(model_name="first-model", dimension=NARROW),
            FakeEmbedder(model_name="second-model", dimension=WIDE),
            build_chunker(),
            RebuildReason.model_changed,
            id="another model, of another width",
        ),
        pytest.param(
            FakeEmbedder(),
            FakeEmbedder(),
            VersionedChunker(build_chunker(), CHUNKER_VERSION + 1),
            RebuildReason.chunker_changed,
            id="another chunker version",
        ),
    ],
)
async def test_an_index_built_from_something_else_is_rebuilt_saying_why(
    test_settings: Settings,
    sample_project: Path,
    tmp_path: Path,
    first_embedder: FakeEmbedder,
    second_embedder: FakeEmbedder,
    second_chunker: Chunker,
    reason: RebuildReason,
):
    store_directory = tmp_path / "index"
    first = indexing_service(
        open_index_store(store_directory), first_embedder, test_settings, build_chunker()
    )
    await first.index(sample_project, force=False, progress=NO_PROGRESS)
    second = indexing_service(
        open_index_store(store_directory), second_embedder, test_settings, second_chunker
    )

    report = await second.index(sample_project, force=False, progress=NO_PROGRESS)
    status = await second.status(sample_project)

    assert (report.rebuild_reason, report.files_indexed) == (reason, 2)
    assert (status.chunks_count, status.stale_files_count, status.rebuild_reason) == (6, 0, None)


async def test_an_index_of_version_0_4_is_rebuilt_as_unversioned(
    test_settings: Settings, sample_project: Path, tmp_path: Path
):
    store_directory = tmp_path / "index"
    lancedb.connect(str(store_directory)).create_table(
        "chunks", data=pyarrow.Table.from_pylist([LEGACY_ROW])
    )
    (store_directory / LEGACY_RECORDS).write_text(json.dumps({"/elsewhere/old.py": 1.0}))
    indexing = indexing_service(
        open_index_store(store_directory), FakeEmbedder(), test_settings, build_chunker()
    )

    report = await indexing.index(sample_project, force=False, progress=NO_PROGRESS)
    status = await indexing.status(sample_project)

    assert report.rebuild_reason == RebuildReason.unversioned
    assert (status.files_count, status.chunks_count) == (2, 6)
    assert not (store_directory / LEGACY_RECORDS).exists()


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
    make_repository(sample_project)
    commit_all(sample_project)
    settings = Settings(cache_dir=tmp_path / "cache", local_index=True)
    indexing = build_registry(settings, FakeEmbedder()).services_for(sample_project).indexing

    await indexing.index(sample_project, force=False, progress=NO_PROGRESS)

    assert git_output(sample_project, "status", "--porcelain") == ""


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
    indexing = indexing_service(FakeIndexStore(hits=[]), embedder, test_settings, build_chunker())

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

    loaded_at = {snapshot.progress: snapshot.model_loaded for snapshot in recorder.snapshots}
    assert loaded_at[IndexPhase.loading_model.value] is False
    assert loaded_at[IndexPhase.embedding.value] is True


@pytest.mark.parametrize(
    ("functions", "embedded_at_reports"),
    [
        pytest.param(
            TWO_AND_A_PART_SLICES,
            [0, 0, 0, 0, EMBEDDING_SLICE_SIZE, 2 * EMBEDDING_SLICE_SIZE, TWO_AND_A_PART_SLICES],
            id="the last slice in part",
        ),
        pytest.param(
            2 * EMBEDDING_SLICE_SIZE,
            [0, 0, 0, 0, EMBEDDING_SLICE_SIZE, 2 * EMBEDDING_SLICE_SIZE],
            id="whole slices",
        ),
    ],
)
async def test_progress_is_reported_before_each_slice_of_chunks_is_embedded(
    test_settings: Settings, tmp_path: Path, functions: int, embedded_at_reports: list[int]
):
    """Reports come at scanning, planning, chunking, before each slice, and at writing."""
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
    store = FakeIndexStore(hits=[])
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


async def test_an_index_whose_file_records_were_lost_is_rebuilt(
    test_settings: Settings, sample_project: Path, tmp_path: Path
):
    """A whole-index write that stopped after removing the records leaves chunks alone."""
    store_directory = tmp_path / "index"
    indexing = indexing_service(
        open_index_store(store_directory), FakeEmbedder(), test_settings, build_chunker()
    )
    await indexing.index(sample_project, force=False, progress=NO_PROGRESS)
    lancedb.connect(str(store_directory)).open_table(FILES_TABLE).delete(MATCH_EVERY_ROW)

    status = await indexing.status(sample_project)
    report = await indexing.index(sample_project, force=False, progress=NO_PROGRESS)

    assert (status.rebuild_reason, report.files_indexed) == (RebuildReason.incomplete, 2)


async def test_a_run_planned_before_another_process_rebuilt_the_index_plans_again(
    test_settings: Settings, sample_project: Path, tmp_path: Path
):
    store_directory = tmp_path / "index"
    indexing = indexing_service(
        open_index_store(store_directory),
        FakeEmbedder(),
        test_settings,
        ChangingChunker(build_chunker(), partial(rebuild_as_another_model, store_directory)),
    )

    report = await indexing.index(sample_project, force=False, progress=NO_PROGRESS)
    status = await indexing.status(sample_project)

    assert report.rebuild_reason == RebuildReason.model_changed
    assert (status.manifest.identity.model_name, status.chunks_count) == (FAKE_MODEL_NAME, 6)


def rebuild_as_another_model(store_directory: Path) -> None:
    """Another process builds the index again from another model, holding nothing yet."""
    another_model = IndexIdentity(
        model_name="another-model", chunker_version=CHUNKER_VERSION, schema_version=SCHEMA_VERSION
    )
    open_index_store(store_directory).write(
        IndexUpdate(
            scope=WriteScope.everything,
            files=[],
            chunks=[],
            removed_paths=[],
            manifest=IndexManifest(identity=another_model, indexed_at=datetime.now(UTC)),
        ),
        numpy.empty(NO_VECTORS_SHAPE, dtype=numpy.float32),
    )
