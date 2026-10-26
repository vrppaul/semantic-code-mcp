"""Search: bring the index up to date, then find the chunks closest to a query."""

import asyncio
import time
from collections import defaultdict
from enum import Enum
from pathlib import Path

from semantic_code_mcp.domain.interfaces import Embedder, IndexStore
from semantic_code_mcp.domain.models import (
    HIGHEST_SCORE,
    IndexReport,
    IndexStatus,
    SearchReport,
    SearchResult,
    SearchTimings,
)
from semantic_code_mcp.services.indexing import IndexingService
from semantic_code_mcp.services.progress import ProgressReporter
from semantic_code_mcp.services.timing import elapsed_milliseconds

# Hits below this blended score are noise
MIN_SCORE = 0.3
# Share of the vector score in the blend; the full-text score takes the rest
VECTOR_WEIGHT = 0.5

# Recency: files edited within the last week get a small score boost (up to +5%)
# to surface recently-touched code. 5% is small enough not to override relevance.
ONE_WEEK_SECONDS = 7 * 24 * 60 * 60
MAX_RECENCY_BOOST = 0.05

# Overfetch 2x because score filtering and deduplication typically remove 30-50% of
# raw results; 2x keeps the pipeline simple while enough candidates fill the limit.
SEARCH_OVERFETCH_FACTOR = 2

# The timings of a search that had nothing to look through
NO_TIMINGS = SearchTimings(embedding_ms=0.0, search_ms=0.0, total_ms=0.0)


class SearchPhase(Enum):
    """The steps of a search, each worth how far along the search is when it starts."""

    checking_index = 5.0
    indexing = 10.0
    loading_model = 80.0
    searching = 85.0
    done = 100.0


# An index run inside a search reports within this band of the search's progress
INDEXING_BAND_END = SearchPhase.loading_model.value


class SearchService:
    """Answers searches over one project, indexing what changed first."""

    def __init__(self, store: IndexStore, embedder: Embedder, indexing: IndexingService) -> None:
        self._store = store
        self._embedder = embedder
        self._indexing = indexing

    async def search(
        self, query: str, project_root: Path, *, limit: int, progress: ProgressReporter
    ) -> SearchReport:
        """Index stale files, then return the best hits for a query."""
        await progress.report(SearchPhase.checking_index.value, "Checking index...")
        status = await self._indexing.status(project_root)

        index_report: IndexReport | None = None
        if status.rebuild_reason is not None or status.stale_files_count:
            await progress.report(SearchPhase.indexing.value, indexing_message(status))
            index_report = await self._indexing.index(
                project_root,
                force=False,
                progress=progress.within(SearchPhase.indexing.value, INDEXING_BAND_END),
            )

        # Counted after any index run: one that found its work done by another run indexed
        # nothing, though the index holds chunks
        counts = await asyncio.to_thread(self._store.counts)
        if counts.chunks == 0:
            # No chunk to find: answer at once, without loading the model for the query
            await progress.report(SearchPhase.done.value, "Nothing indexed")
            return SearchReport(
                results=[], filtered_count=0, timings=NO_TIMINGS, index_report=index_report
            )

        if not self._embedder.is_loaded:
            await progress.report(SearchPhase.loading_model.value, "Loading the embedding model...")
            await asyncio.to_thread(self._embedder.load)
        await progress.report(SearchPhase.searching.value, "Searching...")
        report = await asyncio.to_thread(self._run_search, query, project_root, limit, index_report)
        await progress.report(SearchPhase.done.value, f"Found {len(report.results)} results")
        return report

    def _run_search(
        self, query: str, project_root: Path, limit: int, index_report: IndexReport | None
    ) -> SearchReport:
        started = time.perf_counter()

        embedding_started = time.perf_counter()
        query_vector = self._embedder.embed_query(query)
        embedding_ms = elapsed_milliseconds(embedding_started)

        search_started = time.perf_counter()
        raw_results = self._store.search_hybrid(
            query_vector, query, limit * SEARCH_OVERFETCH_FACTOR, VECTOR_WEIGHT
        )
        search_ms = elapsed_milliseconds(search_started)

        kept = [result for result in raw_results if result.score >= MIN_SCORE]
        now = time.time()
        boosted = sorted(
            (with_recency_boost(result, project_root, now) for result in kept),
            key=lambda result: result.score,
            reverse=True,
        )

        return SearchReport(
            results=group_by_file(boosted[:limit]),
            filtered_count=len(raw_results) - len(kept),
            timings=SearchTimings(
                embedding_ms=embedding_ms,
                search_ms=search_ms,
                total_ms=elapsed_milliseconds(started),
            ),
            index_report=index_report,
        )


def indexing_message(status: IndexStatus) -> str:
    """What a search tells the client it is doing before it indexes."""
    if status.rebuild_reason is not None:
        return f"Rebuilding the index ({status.rebuild_reason})..."
    if not status.is_indexed:
        return "Index not found, indexing..."
    return f"Re-indexing {status.stale_files_count} stale files..."


def with_recency_boost(result: SearchResult, project_root: Path, now: float) -> SearchResult:
    """A hit whose file changed within the last week, scored up by as much as MAX_RECENCY_BOOST."""
    try:
        age_seconds = now - (project_root / result.chunk.path).stat().st_mtime
    except OSError:
        return result
    if age_seconds >= ONE_WEEK_SECONDS:
        return result
    boost = MAX_RECENCY_BOOST * (1 - age_seconds / ONE_WEEK_SECONDS)
    return result.with_score(min(HIGHEST_SCORE, result.score + boost))


def group_by_file(results: list[SearchResult]) -> list[SearchResult]:
    """Hits grouped by file, the file with the best hit first, each file's hits in order."""
    by_file: dict[Path, list[SearchResult]] = defaultdict(list)
    for result in results:
        by_file[result.chunk.path].append(result)
    ordered_files = sorted(
        by_file, key=lambda path: max(hit.score for hit in by_file[path]), reverse=True
    )
    return [hit for path in ordered_files for hit in by_file[path]]
