"""Tests for search: what it keeps of the store's hits, and the indexing it does first."""

import asyncio
import threading
from pathlib import Path

from semantic_code_mcp.app import build_chunker, open_index_store
from semantic_code_mcp.chunkers.composite import CHUNKER_VERSION
from semantic_code_mcp.config import Settings
from semantic_code_mcp.domain.interfaces import IndexStore
from semantic_code_mcp.domain.models import (
    Chunk,
    ChunkKind,
    IndexIdentity,
    Language,
    SearchResult,
)
from semantic_code_mcp.services.progress import ProgressReporter
from semantic_code_mcp.services.search import MIN_SCORE, SearchService
from tests.fakes import FAKE_MODEL_NAME, FakeEmbedder, FakeIndexStore, ProgressRecorder
from tests.helpers import indexing_service

NO_PROGRESS = ProgressReporter(None)
LIMIT = 10
# A vector width other than the fake model's
WIDE = 16
BELOW_MIN_SCORE = MIN_SCORE / 2
# A file no project of these tests holds, so no recency boost reaches it
OLD_FILE = "nowhere/old.py"


def hit(name: str, score: float, path: str = OLD_FILE) -> SearchResult:
    chunk = Chunk(
        chunk_id=f"{path}#{name}",
        path=Path(path),
        language=Language.python,
        kind=ChunkKind.function,
        name=name,
        parent=None,
        line_start=1,
        line_end=2,
        content=f"def {name}(): pass",
    )
    return SearchResult(chunk=chunk, score=score)


def search_service(store: IndexStore, settings: Settings, embedder: FakeEmbedder) -> SearchService:
    indexing = indexing_service(store, embedder, settings, build_chunker())
    return SearchService(store=store, embedder=embedder, indexing=indexing)


async def test_hits_below_the_minimum_score_are_dropped(
    test_settings: Settings, sample_project: Path
):
    store = FakeIndexStore(hits=[hit("kept", 0.9), hit("dropped", BELOW_MIN_SCORE)])
    service = search_service(store, test_settings, FakeEmbedder())

    report = await service.search("anything", sample_project, limit=LIMIT, progress=NO_PROGRESS)

    assert [result.chunk.name for result in report.results] == ["kept"]
    assert report.filtered_count == 1


async def test_no_more_hits_than_the_limit(test_settings: Settings, sample_project: Path):
    store = FakeIndexStore(hits=[hit(f"function_{rank}", 0.9) for rank in range(LIMIT)])
    service = search_service(store, test_settings, FakeEmbedder())

    report = await service.search("anything", sample_project, limit=3, progress=NO_PROGRESS)

    assert len(report.results) == 3


async def test_hits_of_one_file_stand_together_best_file_first(
    test_settings: Settings, sample_project: Path
):
    store = FakeIndexStore(
        hits=[
            hit("best", 0.9, "nowhere/a.py"),
            hit("second", 0.8, "nowhere/b.py"),
            hit("third", 0.7, "nowhere/a.py"),
        ]
    )
    service = search_service(store, test_settings, FakeEmbedder())

    report = await service.search("anything", sample_project, limit=LIMIT, progress=NO_PROGRESS)

    assert [result.chunk.name for result in report.results] == ["best", "third", "second"]


async def test_a_project_never_indexed_is_indexed_before_the_search(
    test_settings: Settings, sample_project: Path
):
    store = FakeIndexStore(hits=[])
    service = search_service(store, test_settings, FakeEmbedder())

    report = await service.search("anything", sample_project, limit=LIMIT, progress=NO_PROGRESS)

    assert report.index_report is not None
    assert report.index_report.files_indexed == 2


async def test_an_index_up_to_date_is_not_touched(test_settings: Settings, sample_project: Path):
    store = FakeIndexStore(hits=[])
    service = search_service(store, test_settings, FakeEmbedder())
    await service.search("first", sample_project, limit=LIMIT, progress=NO_PROGRESS)

    report = await service.search("second", sample_project, limit=LIMIT, progress=NO_PROGRESS)

    assert report.index_report is None


async def test_progress_of_a_search_that_indexes_only_moves_forward(
    test_settings: Settings, sample_project: Path
):
    recorder = ProgressRecorder()
    service = search_service(FakeIndexStore(hits=[]), test_settings, FakeEmbedder())

    await service.search(
        "anything", sample_project, limit=LIMIT, progress=ProgressReporter(recorder)
    )

    assert recorder.progress_values == sorted(recorder.progress_values)


async def test_a_project_with_nothing_indexed_is_answered_without_the_model(
    test_settings: Settings, tmp_path: Path
):
    empty_project = tmp_path / "empty"
    empty_project.mkdir()
    embedder = FakeEmbedder(loaded=False)
    service = search_service(FakeIndexStore(hits=[]), test_settings, embedder)

    report = await service.search("anything", empty_project, limit=LIMIT, progress=NO_PROGRESS)

    assert (report.results, embedder.is_loaded) == ([], False)


async def test_a_fresh_server_loads_the_model_for_the_query_off_the_event_loop(
    test_settings: Settings, sample_project: Path
):
    """An up-to-date index and a model not yet loaded: the first search after a restart."""
    store = FakeIndexStore(hits=[])
    await indexing_service(store, FakeEmbedder(), test_settings, build_chunker()).index(
        sample_project, force=False, progress=NO_PROGRESS
    )
    fresh_embedder = FakeEmbedder(loaded=False)
    service = search_service(store, test_settings, fresh_embedder)

    await service.search("anything", sample_project, limit=LIMIT, progress=NO_PROGRESS)

    assert fresh_embedder.calling_threads
    assert threading.main_thread() not in fresh_embedder.calling_threads


async def test_two_first_searches_at_once_both_find_the_code(
    test_settings: Settings, sample_project: Path
):
    """The second waits for the first one's index run, finds its work done, and searches."""
    store = FakeIndexStore(hits=[hit("found", 0.9)])
    service = search_service(store, test_settings, FakeEmbedder())

    reports = await asyncio.gather(
        service.search("anything", sample_project, limit=LIMIT, progress=NO_PROGRESS),
        service.search("anything", sample_project, limit=LIMIT, progress=NO_PROGRESS),
    )

    assert [[result.chunk.name for result in report.results] for report in reports] == [
        ["found"],
        ["found"],
    ]


async def test_a_search_after_a_model_of_another_width_took_its_name_builds_the_index_again(
    test_settings: Settings, sample_project: Path, tmp_path: Path
):
    """No file changed, so no run would start: the query's width alone tells."""
    identity = IndexIdentity(model_name=FAKE_MODEL_NAME, chunker_version=CHUNKER_VERSION)
    store = open_index_store(tmp_path / "index", identity)
    await search_service(store, test_settings, FakeEmbedder()).search(
        "first", sample_project, limit=LIMIT, progress=NO_PROGRESS
    )

    report = await search_service(store, test_settings, FakeEmbedder(dimension=WIDE)).search(
        "second", sample_project, limit=LIMIT, progress=NO_PROGRESS
    )

    assert report.index_report.files_indexed == 2
