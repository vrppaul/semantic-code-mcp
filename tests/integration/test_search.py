"""Tests for search: what it keeps of the store's hits, and the indexing it does first."""

import threading
from pathlib import Path

from semantic_code_mcp.config import Settings
from semantic_code_mcp.domain.models import ChunkType, SearchResult
from semantic_code_mcp.services.progress import ProgressReporter
from semantic_code_mcp.services.search import MIN_SCORE, SearchService
from tests.fakes import FakeEmbedder, FakeVectorStore, ProgressRecorder
from tests.helpers import indexing_service

NO_PROGRESS = ProgressReporter(None)
LIMIT = 10
BELOW_MIN_SCORE = MIN_SCORE / 2
# Old enough that no recency boost reaches it
OLD_FILE = "/nowhere/old.py"


def hit(name: str, score: float, file_path: str = OLD_FILE) -> SearchResult:
    return SearchResult(
        file_path=file_path,
        line_start=1,
        line_end=2,
        content=f"def {name}(): pass",
        chunk_type=ChunkType.function,
        name=name,
        score=score,
    )


def search_service(
    store: FakeVectorStore,
    settings: Settings,
    project: Path,
    embedder: FakeEmbedder,
) -> SearchService:
    indexing = indexing_service(store, embedder, settings, project)
    return SearchService(store=store, embedder=embedder, indexing=indexing)


async def test_hits_below_the_minimum_score_are_dropped(
    test_settings: Settings, sample_project: Path
):
    store = FakeVectorStore(hits=[hit("kept", 0.9), hit("dropped", BELOW_MIN_SCORE)])
    service = search_service(store, test_settings, sample_project, FakeEmbedder())

    report = await service.search("anything", sample_project, limit=LIMIT, progress=NO_PROGRESS)

    assert [result.name for result in report.results] == ["kept"]
    assert report.filtered_count == 1


async def test_no_more_hits_than_the_limit(test_settings: Settings, sample_project: Path):
    store = FakeVectorStore(hits=[hit(f"function_{rank}", 0.9) for rank in range(LIMIT)])
    service = search_service(store, test_settings, sample_project, FakeEmbedder())

    report = await service.search("anything", sample_project, limit=3, progress=NO_PROGRESS)

    assert len(report.results) == 3


async def test_hits_of_one_file_stand_together_best_file_first(
    test_settings: Settings, sample_project: Path
):
    store = FakeVectorStore(
        hits=[
            hit("best", 0.9, "/nowhere/a.py"),
            hit("second", 0.8, "/nowhere/b.py"),
            hit("third", 0.7, "/nowhere/a.py"),
        ]
    )
    service = search_service(store, test_settings, sample_project, FakeEmbedder())

    report = await service.search("anything", sample_project, limit=LIMIT, progress=NO_PROGRESS)

    assert [result.name for result in report.results] == ["best", "third", "second"]


async def test_a_project_never_indexed_is_indexed_before_the_search(
    test_settings: Settings, sample_project: Path
):
    store = FakeVectorStore(hits=[])
    service = search_service(store, test_settings, sample_project, FakeEmbedder())

    report = await service.search("anything", sample_project, limit=LIMIT, progress=NO_PROGRESS)

    assert report.index_result is not None
    assert report.index_result.files_indexed == 2


async def test_an_index_up_to_date_is_not_touched(test_settings: Settings, sample_project: Path):
    store = FakeVectorStore(hits=[])
    service = search_service(store, test_settings, sample_project, FakeEmbedder())
    await service.search("first", sample_project, limit=LIMIT, progress=NO_PROGRESS)

    report = await service.search("second", sample_project, limit=LIMIT, progress=NO_PROGRESS)

    assert report.index_result is None


async def test_progress_of_a_search_that_indexes_only_moves_forward(
    test_settings: Settings, sample_project: Path
):
    recorder = ProgressRecorder()
    service = search_service(
        FakeVectorStore(hits=[]), test_settings, sample_project, FakeEmbedder()
    )

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
    service = search_service(FakeVectorStore(hits=[]), test_settings, empty_project, embedder)

    report = await service.search("anything", empty_project, limit=LIMIT, progress=NO_PROGRESS)

    assert (report.results, embedder.is_loaded) == ([], False)


async def test_a_fresh_server_loads_the_model_for_the_query_off_the_event_loop(
    test_settings: Settings, sample_project: Path
):
    """An up-to-date index and a model not yet loaded: the first search after a restart."""
    store = FakeVectorStore(hits=[])
    await indexing_service(store, FakeEmbedder(), test_settings, sample_project).index(
        sample_project, force=False, progress=NO_PROGRESS
    )
    fresh_embedder = FakeEmbedder(loaded=False)
    service = search_service(store, test_settings, sample_project, fresh_embedder)

    await service.search("anything", sample_project, limit=LIMIT, progress=NO_PROGRESS)

    assert fresh_embedder.calling_threads
    assert threading.main_thread() not in fresh_embedder.calling_threads
