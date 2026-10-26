"""Tests for search: what it keeps of the store's hits, and the indexing it does first."""

import asyncio
import threading
from pathlib import Path

from semantic_code_mcp.app import build_chunker
from semantic_code_mcp.config import Settings
from semantic_code_mcp.domain.models import (
    Candidate,
    Chunk,
    ChunkKind,
    Language,
    SearchCandidates,
)
from semantic_code_mcp.services.progress import ProgressReporter
from semantic_code_mcp.services.search import MIN_SIMILARITY, SearchService
from semantic_code_mcp.storage.lancedb import NO_CANDIDATES
from tests.fakes import FakeEmbedder, FakeIndexStore, ProgressRecorder
from tests.helpers import indexing_service

NO_PROGRESS = ProgressReporter(None)
LIMIT = 10
CLOSE = 0.9
BELOW_MIN_SIMILARITY = MIN_SIMILARITY / 2
# A file no project of these tests holds
ELSEWHERE = "nowhere/old.py"


def candidate(name: str, similarity: float, path: str = ELSEWHERE) -> Candidate:
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
    return Candidate(chunk=chunk, similarity=similarity)


def found_by_vector(*candidates: Candidate) -> SearchCandidates:
    return SearchCandidates(by_vector=list(candidates), by_keyword=[])


def search_service(
    store: FakeIndexStore, settings: Settings, embedder: FakeEmbedder
) -> SearchService:
    indexing = indexing_service(store, embedder, settings, build_chunker())
    return SearchService(store=store, embedder=embedder, indexing=indexing)


async def test_a_hit_far_in_meaning_is_dropped_unless_its_words_match(
    test_settings: Settings, sample_project: Path
):
    named = candidate("named", BELOW_MIN_SIMILARITY)
    store = FakeIndexStore(
        SearchCandidates(
            by_vector=[candidate("close", CLOSE), candidate("far", BELOW_MIN_SIMILARITY), named],
            by_keyword=[named],
        )
    )
    service = search_service(store, test_settings, FakeEmbedder())

    report = await service.search("anything", sample_project, limit=LIMIT, progress=NO_PROGRESS)

    assert ([hit.chunk.name for hit in report.results], report.filtered_count) == (
        ["named", "close"],
        1,
    )


async def test_no_more_hits_than_the_limit(test_settings: Settings, sample_project: Path):
    store = FakeIndexStore(
        found_by_vector(*(candidate(f"function_{rank}", CLOSE) for rank in range(LIMIT)))
    )
    service = search_service(store, test_settings, FakeEmbedder())

    report = await service.search("anything", sample_project, limit=3, progress=NO_PROGRESS)

    assert len(report.results) == 3


async def test_hits_come_in_fused_order_whichever_file_they_are_in(
    test_settings: Settings, sample_project: Path
):
    store = FakeIndexStore(
        found_by_vector(
            candidate("best", CLOSE, "nowhere/a.py"),
            candidate("second", CLOSE, "nowhere/b.py"),
            candidate("third", CLOSE, "nowhere/a.py"),
        )
    )
    service = search_service(store, test_settings, FakeEmbedder())

    report = await service.search("anything", sample_project, limit=LIMIT, progress=NO_PROGRESS)

    assert [hit.chunk.name for hit in report.results] == ["best", "second", "third"]


async def test_a_project_never_indexed_is_indexed_before_the_search(
    test_settings: Settings, sample_project: Path
):
    store = FakeIndexStore(NO_CANDIDATES)
    service = search_service(store, test_settings, FakeEmbedder())

    report = await service.search("anything", sample_project, limit=LIMIT, progress=NO_PROGRESS)

    assert report.index_report is not None
    assert report.index_report.files_indexed == 2


async def test_an_index_up_to_date_is_not_touched(test_settings: Settings, sample_project: Path):
    store = FakeIndexStore(NO_CANDIDATES)
    service = search_service(store, test_settings, FakeEmbedder())
    await service.search("first", sample_project, limit=LIMIT, progress=NO_PROGRESS)

    report = await service.search("second", sample_project, limit=LIMIT, progress=NO_PROGRESS)

    assert report.index_report is None


async def test_progress_of_a_search_that_indexes_only_moves_forward(
    test_settings: Settings, sample_project: Path
):
    recorder = ProgressRecorder()
    service = search_service(FakeIndexStore(NO_CANDIDATES), test_settings, FakeEmbedder())

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
    service = search_service(FakeIndexStore(NO_CANDIDATES), test_settings, embedder)

    report = await service.search("anything", empty_project, limit=LIMIT, progress=NO_PROGRESS)

    assert (report.results, embedder.is_loaded) == ([], False)


async def test_a_fresh_server_loads_the_model_for_the_query_off_the_event_loop(
    test_settings: Settings, sample_project: Path
):
    """An up-to-date index and a model not yet loaded: the first search after a restart."""
    store = FakeIndexStore(NO_CANDIDATES)
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
    store = FakeIndexStore(found_by_vector(candidate("found", CLOSE)))
    service = search_service(store, test_settings, FakeEmbedder())

    reports = await asyncio.gather(
        service.search("anything", sample_project, limit=LIMIT, progress=NO_PROGRESS),
        service.search("anything", sample_project, limit=LIMIT, progress=NO_PROGRESS),
    )

    assert [[hit.chunk.name for hit in report.results] for report in reports] == [
        ["found"],
        ["found"],
    ]
