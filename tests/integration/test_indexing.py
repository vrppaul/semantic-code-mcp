"""Tests for indexing a project into a real LanceDB store."""

import asyncio
import os
from pathlib import Path

import pytest

from semantic_code_mcp.config import Settings, get_index_path
from semantic_code_mcp.embedding import SentenceTransformerEmbedder
from semantic_code_mcp.services.indexing import IndexingService, IndexPhase
from semantic_code_mcp.services.progress import ProgressReporter
from semantic_code_mcp.services.projects import ProjectRegistry, build_chunker
from tests.conftest import MODEL_NAME
from tests.fakes import FakeEmbedder, LoadWatchingRecorder, ProgressRecorder
from tests.helpers import commit_all, make_repository

NO_PROGRESS = ProgressReporter(None)
# An mtime well after any file the test just wrote
LATER_MTIME_SECONDS = 4_000_000_000


@pytest.fixture
async def indexing(test_settings: Settings, sample_project: Path) -> IndexingService:
    registry = ProjectRegistry(
        settings=test_settings, embedder=FakeEmbedder(), chunker=build_chunker()
    )
    return (await registry.services_for(sample_project)).indexing


async def test_index_stores_every_chunk_of_every_file(
    indexing: IndexingService, sample_project: Path
):
    result = await indexing.index(sample_project, force=False, progress=NO_PROGRESS)
    status = await indexing.status(sample_project)

    assert (result.files_indexed, result.chunks_indexed) == (2, 6)
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

    result = await indexing.index(sample_project, force=False, progress=NO_PROGRESS)
    status = await indexing.status(sample_project)

    assert (result.files_indexed, result.chunks_indexed) == (1, 4)
    assert status.chunks_count == 7


async def test_force_indexes_every_file_again(indexing: IndexingService, sample_project: Path):
    await indexing.index(sample_project, force=False, progress=NO_PROGRESS)

    result = await indexing.index(sample_project, force=True, progress=NO_PROGRESS)
    status = await indexing.status(sample_project)

    assert result.files_indexed == 2
    assert status.chunks_count == 6


async def test_a_deleted_file_leaves_the_index(indexing: IndexingService, sample_project: Path):
    await indexing.index(sample_project, force=False, progress=NO_PROGRESS)
    (sample_project / "utils.py").unlink()

    result = await indexing.index(sample_project, force=False, progress=NO_PROGRESS)
    status = await indexing.status(sample_project)

    assert result.files_deleted == 1
    assert (status.files_count, status.chunks_count) == (1, 3)


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

    result = await indexing.index(sample_project, force=False, progress=NO_PROGRESS)

    assert result.files_indexed == 3


async def test_tracked_file_deleted_from_disk_is_not_indexed_again(
    indexing: IndexingService, sample_project: Path
):
    make_repository(sample_project)
    commit_all(sample_project)
    (sample_project / "utils.py").unlink()
    await indexing.index(sample_project, force=False, progress=NO_PROGRESS)

    second_run = await indexing.index(sample_project, force=False, progress=NO_PROGRESS)

    assert second_run.files_indexed == 0


async def test_progress_only_moves_forward(indexing: IndexingService, sample_project: Path):
    recorder = ProgressRecorder()

    await indexing.index(sample_project, force=False, progress=ProgressReporter(recorder))

    assert recorder.progress_values
    assert recorder.progress_values == sorted(recorder.progress_values)


async def test_status_of_a_project_never_indexed(test_settings: Settings, tmp_path: Path):
    project = tmp_path / "fresh"
    project.mkdir()
    (project / "code.py").write_text("def code():\n    return 1\n")
    registry = ProjectRegistry(
        settings=test_settings, embedder=FakeEmbedder(), chunker=build_chunker()
    )

    status = await (await registry.services_for(project)).indexing.status(project)

    assert (status.is_indexed, status.files_count, len(status.stale_files)) == (False, 0, 1)
    assert get_index_path(test_settings, project).is_dir()


async def test_status_during_a_forced_run_never_fails(
    indexing: IndexingService, sample_project: Path
):
    """A forced run empties the store while readers ask for status in parallel."""
    await indexing.index(sample_project, force=False, progress=NO_PROGRESS)

    statuses = []
    for _ in range(5):
        _, *round_statuses = await asyncio.gather(
            indexing.index(sample_project, force=True, progress=NO_PROGRESS),
            indexing.status(sample_project),
            indexing.status(sample_project),
        )
        statuses.extend(round_statuses)

    # A reader sees the index emptied or filled, never half of it, and never an error
    assert {status.chunks_count for status in statuses} <= {0, 6}


async def test_a_project_without_indexable_files_indexes_nothing(
    test_settings: Settings, tmp_path: Path
):
    project = tmp_path / "notes"
    project.mkdir()
    (project / "notes.txt").write_text("nothing to index\n")
    registry = ProjectRegistry(
        settings=test_settings, embedder=FakeEmbedder(), chunker=build_chunker()
    )
    services = await registry.services_for(project)

    result = await services.indexing.index(project, force=False, progress=NO_PROGRESS)
    status = await services.indexing.status(project)

    assert (result.files_indexed, status.is_indexed) == (0, False)


async def test_status_does_not_load_the_model(test_settings: Settings, sample_project: Path):
    embedder = SentenceTransformerEmbedder(MODEL_NAME)
    registry = ProjectRegistry(settings=test_settings, embedder=embedder, chunker=build_chunker())
    services = await registry.services_for(sample_project)

    await services.indexing.status(sample_project)

    assert embedder.is_loaded is False


async def test_the_model_load_is_reported_before_it_happens(
    test_settings: Settings, sample_project: Path
):
    embedder = FakeEmbedder(loaded=False)
    recorder = LoadWatchingRecorder(embedder)
    registry = ProjectRegistry(settings=test_settings, embedder=embedder, chunker=build_chunker())
    services = await registry.services_for(sample_project)

    await services.indexing.index(sample_project, force=False, progress=ProgressReporter(recorder))

    loaded_at = {snapshot.progress: snapshot.model_loaded for snapshot in recorder.snapshots}
    assert loaded_at[IndexPhase.loading_model.value] is False
    assert loaded_at[IndexPhase.embedding.value] is True


async def test_two_first_requests_for_one_project_share_its_services(
    test_settings: Settings, sample_project: Path
):
    registry = ProjectRegistry(
        settings=test_settings, embedder=FakeEmbedder(), chunker=build_chunker()
    )

    first, second = await asyncio.gather(
        registry.services_for(sample_project), registry.services_for(sample_project)
    )

    assert first is second
