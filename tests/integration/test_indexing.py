"""Tests of indexing a project, into each store in turn."""

import asyncio
import re
import sys
import threading
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy
import pytest

from semantic_code_mcp.app import (
    LANCEDB_EXTRA,
    StoreUnavailableError,
    build_chunker,
    build_registry,
)
from semantic_code_mcp.chunkers.composite import CHUNKER_VERSION
from semantic_code_mcp.config import LOCAL_INDEX_DIRECTORY, Settings, get_index_path
from semantic_code_mcp.domain.interfaces import Chunker
from semantic_code_mcp.domain.models import IndexCounts, StoreKind
from semantic_code_mcp.embedding import OnnxEmbedder
from semantic_code_mcp.services.indexing import EMBEDDING_SLICE_SIZE, IndexingService
from semantic_code_mcp.services.progress import ProgressReporter
from semantic_code_mcp.storage.index_directories import LEGACY_ENTRIES
from tests.conftest import MODEL_NAME
from tests.fakes import (
    EMBEDDING_DIMENSION,
    FAKE_EMBEDDER_VERSION,
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
    commit_all,
    git_output,
    indexing_service,
    make_repository,
    registry_of,
)
from tests.indexing import (
    ADDED_FUNCTION,
    FILES_OF_FUNCTIONS,
    FUNCTIONS_PER_FILE,
    IDENTITY,
    NO_PROGRESS,
    first_index,
    functions_in_files,
    touch_later,
)
from tests.stores import NO_CANDIDATES, StoreClass, open_store

# Functions in one file that fill two slices of chunks to embed, and a part of a third
TWO_AND_A_PART_SLICES = 2 * EMBEDDING_SLICE_SIZE + 10
# A vector width other than the fake model's, so a change of model changes the width too
WIDE = 16
# 0.4's record of mtimes, which it kept in a project-local index directory
LEGACY_RECORDS = LEGACY_ENTRIES[1]


def delete(path: Path) -> None:
    path.unlink(missing_ok=True)


@pytest.fixture
def indexing(store_settings: Settings, sample_project: Path) -> IndexingService:
    return build_registry(store_settings, FakeEmbedder()).services_for(sample_project).indexing


async def test_index_stores_every_chunk_of_every_file(
    indexing: IndexingService, sample_project: Path
):
    report = await indexing.index(sample_project, force=False, progress=NO_PROGRESS)
    status = await indexing.status(sample_project)

    assert (report.files_indexed, report.chunks_indexed) == (2, 6)
    assert (status.is_indexed, status.files_count, status.chunks_count) == (True, 2, 6)


async def test_the_index_is_kept_in_the_store_the_settings_pick(
    indexing: IndexingService,
    sample_project: Path,
    store_settings: Settings,
    store_class: StoreClass,
):
    await indexing.index(sample_project, force=False, progress=NO_PROGRESS)

    project_directory = get_index_path(store_settings, sample_project)
    assert open_store(store_class, project_directory, IDENTITY).counts().chunks == 6


def test_a_store_whose_libraries_are_missing_stops_the_server_naming_its_extra(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch
):
    """LanceDB comes with an optional extra; a server set to it without the extra says so
    as it starts, rather than at its first search."""
    monkeypatch.setitem(sys.modules, "lancedb", None)
    monkeypatch.delitem(sys.modules, "semantic_code_mcp.storage.lancedb")
    settings = Settings(cache_dir=test_settings.cache_dir, index_store=StoreKind.lancedb)

    with pytest.raises(StoreUnavailableError, match=re.escape(LANCEDB_EXTRA)):
        build_registry(settings, FakeEmbedder())


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
    changed.write_text(changed.read_text() + ADDED_FUNCTION)
    touch_later(changed)

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
    store_class: StoreClass,
    change: Callable[[Path], object],
):
    """The change lands after the scan took the files' records, so the next run sees it."""
    indexing = indexing_service(
        open_store(store_class, tmp_path / "index", IDENTITY),
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
    test_settings: Settings, sample_project: Path, tmp_path: Path, store_class: StoreClass
):
    store = open_store(store_class, tmp_path / "index", IDENTITY)
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
            FakeEmbedder(version=FAKE_EMBEDDER_VERSION + 1),
            build_chunker(),
            id="the same model read by another embedder version",
        ),
        pytest.param(
            FakeEmbedder(),
            VersionedChunker(build_chunker(), CHUNKER_VERSION + 1),
            id="another chunker version",
        ),
    ],
)
async def test_sessions_of_two_models_or_versions_keep_an_index_each(
    store_settings: Settings,
    sample_project: Path,
    second_embedder: FakeEmbedder,
    second_chunker: Chunker,
):
    """Neither builds the other's index again: the first, run again, finds nothing to do."""
    first = registry_of(store_settings, FakeEmbedder(), build_chunker())
    second = registry_of(store_settings, second_embedder, second_chunker)
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
    sample_project: Path, tmp_path: Path, store_class: StoreClass
):
    """Not even what 0.4 left in it, untracked, before its removal a week later."""
    make_repository(sample_project)
    commit_all(sample_project)
    (sample_project / LOCAL_INDEX_DIRECTORY).mkdir()
    (sample_project / LOCAL_INDEX_DIRECTORY / LEGACY_RECORDS).write_text("{}")
    settings = Settings(
        cache_dir=tmp_path / "cache",
        local_index=True,
        index_store=store_class.LAYOUT.store_kind,
    )
    indexing = build_registry(settings, FakeEmbedder()).services_for(sample_project).indexing

    await indexing.index(sample_project, force=False, progress=NO_PROGRESS)

    assert git_output(sample_project, "status", "--porcelain") == ""


async def test_a_project_moved_with_its_local_index_keeps_it(
    sample_project: Path, tmp_path: Path, store_class: StoreClass
):
    settings = Settings(
        cache_dir=tmp_path / "cache",
        local_index=True,
        index_store=store_class.LAYOUT.store_kind,
    )
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
    store_settings: Settings, tmp_path: Path
):
    project = tmp_path / "fresh"
    project.mkdir()
    (project / "code.py").write_text("def code():\n    return 1\n")
    indexing = build_registry(store_settings, FakeEmbedder()).services_for(project).indexing

    status = await indexing.status(project)

    assert (status.is_indexed, status.files_count, status.stale_files_count) == (False, 0, 1)
    assert not get_index_path(store_settings, project).exists()


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
    store_settings: Settings, tmp_path: Path
):
    project = tmp_path / "notes"
    project.mkdir()
    (project / "notes.txt").write_text("nothing to index\n")
    indexing = build_registry(store_settings, FakeEmbedder()).services_for(project).indexing

    report = await indexing.index(project, force=False, progress=NO_PROGRESS)
    status = await indexing.status(project)

    assert (report.files_indexed, status.is_indexed) == (0, False)


async def test_status_does_not_load_the_model(test_settings: Settings, sample_project: Path):
    embedder = OnnxEmbedder(MODEL_NAME)
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


async def test_an_edited_file_embeds_only_its_chunks_whose_text_changed(
    test_settings: Settings, sample_project: Path, tmp_path: Path, store_class: StoreClass
):
    store = open_store(store_class, tmp_path / "index", IDENTITY)
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
    test_settings: Settings, sample_project: Path, tmp_path: Path, store_class: StoreClass
):
    """A model replaced under a name nothing pins gives other vectors that nothing else
    can see."""
    store = open_store(store_class, tmp_path / "index", IDENTITY)
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
    test_settings: Settings, sample_project: Path, tmp_path: Path, store_class: StoreClass
):
    """A name nothing pins, as a local path's: the file's unchanged chunks hold vectors of
    the old width, so none is reused, each chunk is embedded once, and the index starts
    again from the file, the other one unrecorded."""
    store = open_store(store_class, tmp_path / "index", IDENTITY)
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
    test_settings: Settings, tmp_path: Path, store_class: StoreClass
):
    """The chunks new to a file fill more than one batch, and its old chunks' vectors are
    of another width than the model's now: they are embedded after the new ones."""
    project = functions_in_files(tmp_path, files=1, functions=EMBEDDING_SLICE_SIZE)
    store = open_store(store_class, tmp_path / "index", IDENTITY)
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
    test_settings: Settings, tmp_path: Path, store_class: StoreClass
):
    project = functions_in_files(tmp_path, FILES_OF_FUNCTIONS, FUNCTIONS_PER_FILE)
    store = StoreCancelledAfterItsWrite(open_store(store_class, tmp_path / "index", IDENTITY))
    indexing = indexing_service(store, FakeEmbedder(), test_settings, build_chunker())

    with pytest.raises(asyncio.CancelledError):
        await indexing.index(project, force=False, progress=NO_PROGRESS)

    assert store.counts() == IndexCounts(files=2, chunks=2 * FUNCTIONS_PER_FILE)


# Files of FUNCTIONS_PER_FILE chunks enough for three slices: two of two files, one of one
FILES_OF_THREE_SLICES = 5


async def test_a_run_ends_in_its_last_write_alone(test_settings: Settings, tmp_path: Path):
    """A store does what a run's end needs once, in the last write, not in each slice's."""
    project = functions_in_files(tmp_path, FILES_OF_THREE_SLICES, FUNCTIONS_PER_FILE)
    store = FakeIndexStore(NO_CANDIDATES)

    await first_index(
        indexing_service(store, FakeEmbedder(), test_settings, build_chunker()), project
    )

    assert [update.last_of_run for update in store.updates] == [False, False, True]
