"""Tests of indexing a project's words, into each store in turn."""

import asyncio
import re
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from semantic_code_mcp.app import build_chunker, build_registry, store_opener
from semantic_code_mcp.chunkers.composite import CHUNKER_VERSION
from semantic_code_mcp.cli import build_server
from semantic_code_mcp.config import LOCAL_INDEX_DIRECTORY, Settings, get_index_path
from semantic_code_mcp.domain.models import WHOLE_PROJECT, SearchMode, SearchQuery, StoreKind
from semantic_code_mcp.embedding.embedder import OnnxEmbedder
from semantic_code_mcp.extras import LANCEDB_EXTRA, MEANING_EXTRA, ExtraMissingError, OptionalExtra
from semantic_code_mcp.services import indexing as indexing_module
from semantic_code_mcp.services.indexing import IndexingService
from semantic_code_mcp.services.progress import ProgressReporter
from semantic_code_mcp.storage.index_directories import LEGACY_ENTRIES
from tests.conftest import MODEL_NAME
from tests.fakes import (
    FAKE_EMBEDDER_VERSION,
    FAKE_MODEL,
    ChangingChunker,
    ChunkCountRecorder,
    FakeEmbedder,
    FakeIndexStore,
    ProgressRecorder,
    StoreHeldInItsWrite,
    VersionedChunker,
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
    FUNCTIONS_PER_FILE,
    IDENTITY,
    NO_PROGRESS,
    first_index,
    functions_in_files,
    touch_later,
)
from tests.stores import NO_CANDIDATES, StoreClass, open_store

# A vector width other than the fake model's, so a change of model changes the width too
WIDE = 16
# How often a waiting run tries the index again in these tests, and how long a test waits to
# see it wait
SHORT_WAIT_SECONDS = 0.05
WAIT_SEEN_TIMEOUT_SECONDS = 30.0
# How long a waiting run is watched not to go on: many of its tries
RETRIES_SEEN_SECONDS = 0.5
# 0.4's record of mtimes, which it kept in a project-local index directory
LEGACY_RECORDS = LEGACY_ENTRIES[1]


def delete(path: Path) -> None:
    path.unlink(missing_ok=True)


@pytest.fixture
def indexing(store_settings: Settings, sample_project: Path) -> IndexingService:
    return (
        build_registry(store_settings, FakeEmbedder(), store_opener(store_settings))
        .services_for(sample_project)
        .indexing
    )


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
    assert open_store(store_class, project_directory, IDENTITY).counts(FAKE_MODEL).chunks == 6


class ExtraCase(BaseModel):
    """A setting that needs an optional extra: a library the extra installs, the module of
    the package that imports it, and one of the package's own modules that module imports."""

    model_config = ConfigDict(frozen=True)

    setting: dict[str, StoreKind | SearchMode]
    extra: OptionalExtra
    library: str
    extra_module: str
    package_module: str


EXTRA_CASES = [
    pytest.param(
        ExtraCase(
            setting={"index_store": StoreKind.lancedb},
            library="lancedb",
            extra_module="semantic_code_mcp.storage.lancedb",
            package_module="semantic_code_mcp.storage.globs",
            extra=LANCEDB_EXTRA,
        ),
        id="the LanceDB store",
    ),
    pytest.param(
        ExtraCase(
            setting={"search_mode": SearchMode.words_and_meaning},
            library="onnxruntime",
            extra_module="semantic_code_mcp.embedding.onnx_model",
            package_module="semantic_code_mcp.domain.interfaces",
            extra=MEANING_EXTRA,
        ),
        id="a search by meaning",
    ),
]


@pytest.mark.parametrize("case", EXTRA_CASES)
def test_a_setting_whose_extra_is_missing_stops_the_server_naming_the_extra(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch, case: ExtraCase
):
    """LanceDB and the model's libraries come with optional extras; a server set to need
    one without it says so as it starts, rather than at its first search."""
    monkeypatch.setitem(sys.modules, case.library, None)
    monkeypatch.delitem(sys.modules, case.extra_module, raising=False)
    settings = Settings(cache_dir=test_settings.cache_dir, **case.setting)

    with pytest.raises(ExtraMissingError, match=re.escape(case.extra.requirement)):
        build_server(settings)


@pytest.mark.parametrize("case", EXTRA_CASES)
def test_a_module_of_the_package_missing_is_never_taken_for_the_missing_extra(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch, case: ExtraCase
):
    """A broken install of the package itself says what is missing, not to install the
    extra, which would not help."""
    monkeypatch.setitem(sys.modules, case.package_module, None)
    monkeypatch.delitem(sys.modules, case.extra_module, raising=False)
    settings = Settings(cache_dir=test_settings.cache_dir, **case.setting)

    with pytest.raises(ModuleNotFoundError, match=re.escape(case.package_module)):
        build_server(settings)


async def test_a_server_by_words_starts_and_searches_without_the_model_s_libraries(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch, sample_project: Path
):
    """The extra `semantic-code-mcp[meaning]` is for a search by meaning alone."""
    for library in MEANING_EXTRA.libraries:
        monkeypatch.setitem(sys.modules, library, None)
    monkeypatch.delitem(sys.modules, "semantic_code_mcp.embedding.onnx_model", raising=False)
    settings = Settings(cache_dir=test_settings.cache_dir)
    build_server(settings)
    registry = build_registry(settings, OnnxEmbedder(MODEL_NAME), store_opener(settings))

    report = await registry.services_for(sample_project).search.search(
        SearchQuery(text="greet", limit=1, scope=WHOLE_PROJECT),
        sample_project,
        progress=NO_PROGRESS,
    )

    assert [chunk.name for chunk in report.results] == ["greet"]


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
    indexing = indexing_service(store, test_settings, build_chunker())
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


async def test_sessions_of_two_chunker_versions_keep_an_index_each(
    store_settings: Settings, sample_project: Path
):
    """Neither builds the other's index again: the first, run again, finds nothing to do."""
    first = registry_of(store_settings, FakeEmbedder(), build_chunker())
    second = registry_of(
        store_settings, FakeEmbedder(), VersionedChunker(build_chunker(), CHUNKER_VERSION + 1)
    )
    first_indexing = first.services_for(sample_project).indexing
    await first_indexing.index(sample_project, force=False, progress=NO_PROGRESS)

    second_run = await second.services_for(sample_project).indexing.index(
        sample_project, force=False, progress=NO_PROGRESS
    )
    first_again = await first_indexing.index(sample_project, force=False, progress=NO_PROGRESS)

    assert (second_run.files_indexed, first_again.files_indexed) == (2, 0)


@pytest.mark.parametrize(
    "second_embedder",
    [
        pytest.param(
            FakeEmbedder(model_name="second-model", dimension=WIDE),
            id="another model, of another width",
        ),
        pytest.param(FakeEmbedder(revision="another-revision"), id="another revision"),
        pytest.param(
            FakeEmbedder(version=FAKE_EMBEDDER_VERSION + 1), id="another embedder version"
        ),
    ],
)
async def test_sessions_of_two_models_share_the_index_of_words(
    store_settings: Settings, sample_project: Path, second_embedder: FakeEmbedder
):
    """A change of model embeds again without chunking again (ADR 009, D7)."""
    first = registry_of(store_settings, FakeEmbedder(), build_chunker())
    second = registry_of(store_settings, second_embedder, build_chunker())
    await first.services_for(sample_project).indexing.index(
        sample_project, force=False, progress=NO_PROGRESS
    )

    second_run = await second.services_for(sample_project).indexing.index(
        sample_project, force=False, progress=NO_PROGRESS
    )

    assert second_run.files_indexed == 0


class FirstReportWatcher:
    """A progress callback that marks when the first report comes."""

    def __init__(self) -> None:
        self.first_report = asyncio.Event()

    async def __call__(self, _progress: float, _total: float, _message: str) -> None:
        self.first_report.set()


async def test_a_run_waits_while_another_session_builds_the_index_then_finds_it_built(
    store_settings: Settings,
    sample_project: Path,
    store_class: StoreClass,
    monkeypatch: pytest.MonkeyPatch,
):
    """The other session's run holds the index inside its write: this one tries again and
    again without going on, however long that takes, then finds the work done."""
    monkeypatch.setattr(indexing_module, "RUN_RETRY_SECONDS", SHORT_WAIT_SECONDS)
    project_directory = get_index_path(store_settings, sample_project)
    other_store = StoreHeldInItsWrite(open_store(store_class, project_directory, IDENTITY))
    other_run = asyncio.create_task(
        indexing_service(other_store, store_settings, build_chunker()).index(
            sample_project, force=False, progress=NO_PROGRESS
        )
    )
    await asyncio.to_thread(other_store.writing.wait, WAIT_SEEN_TIMEOUT_SECONDS)
    indexing = (
        build_registry(store_settings, FakeEmbedder(), store_opener(store_settings))
        .services_for(sample_project)
        .indexing
    )
    run = asyncio.create_task(indexing.index(sample_project, force=False, progress=NO_PROGRESS))

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(asyncio.shield(run), RETRIES_SEEN_SECONDS)
    other_store.release.set()
    await other_run
    report = await run

    status = await indexing.status(sample_project)
    assert (report.files_indexed, status.chunks_count) == (0, 6)


async def test_a_run_cancelled_while_it_waits_leaves_the_index_free(
    store_settings: Settings,
    sample_project: Path,
    store_class: StoreClass,
    monkeypatch: pytest.MonkeyPatch,
):
    """A client that hangs up while its search waits for another session's build."""
    monkeypatch.setattr(indexing_module, "RUN_RETRY_SECONDS", SHORT_WAIT_SECONDS)
    project_directory = get_index_path(store_settings, sample_project)
    other_session = open_store(store_class, project_directory, IDENTITY)
    other_session.try_hold_for_run()
    watcher = FirstReportWatcher()
    indexing = (
        build_registry(store_settings, FakeEmbedder(), store_opener(store_settings))
        .services_for(sample_project)
        .indexing
    )
    run = asyncio.create_task(
        indexing.index(sample_project, force=False, progress=ProgressReporter(watcher))
    )
    await asyncio.wait_for(watcher.first_report.wait(), WAIT_SEEN_TIMEOUT_SECONDS)

    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    other_session.release_run()

    assert open_store(store_class, project_directory, IDENTITY).try_hold_for_run()


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
    indexing = (
        build_registry(settings, FakeEmbedder(), store_opener(settings))
        .services_for(sample_project)
        .indexing
    )

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
    first = (
        build_registry(settings, FakeEmbedder(), store_opener(settings))
        .services_for(sample_project)
        .indexing
    )
    await first.index(sample_project, force=False, progress=NO_PROGRESS)
    moved = await asyncio.to_thread(sample_project.rename, tmp_path / "moved")

    status = (
        await build_registry(settings, FakeEmbedder(), store_opener(settings))
        .services_for(moved)
        .indexing.status(moved)
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
    indexing = (
        build_registry(store_settings, FakeEmbedder(), store_opener(store_settings))
        .services_for(project)
        .indexing
    )

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
    indexing = (
        build_registry(store_settings, FakeEmbedder(), store_opener(store_settings))
        .services_for(project)
        .indexing
    )

    report = await indexing.index(project, force=False, progress=NO_PROGRESS)
    status = await indexing.status(project)

    assert (report.files_indexed, status.is_indexed) == (0, False)


async def test_status_does_not_load_the_model(test_settings: Settings, sample_project: Path):
    embedder = OnnxEmbedder(MODEL_NAME)
    indexing = (
        build_registry(test_settings, embedder, store_opener(test_settings))
        .services_for(sample_project)
        .indexing
    )

    await indexing.status(sample_project)

    assert embedder.is_loaded is False


def test_one_project_has_one_set_of_services(test_settings: Settings, sample_project: Path):
    registry = build_registry(test_settings, FakeEmbedder(), store_opener(test_settings))

    assert registry.services_for(sample_project) is registry.services_for(sample_project)


# Files of FUNCTIONS_PER_FILE chunks enough for three slices: two of two files, one of one
FILES_OF_THREE_SLICES = 5


async def test_a_run_ends_in_its_last_write_alone(test_settings: Settings, tmp_path: Path):
    """A store does what a run's end needs once, in the last write, not in each slice's."""
    project = functions_in_files(tmp_path, FILES_OF_THREE_SLICES, FUNCTIONS_PER_FILE)
    store = FakeIndexStore(NO_CANDIDATES)

    await first_index(indexing_service(store, test_settings, build_chunker()), project)

    assert [update.last_of_run for update in store.updates] == [False, False, True]
