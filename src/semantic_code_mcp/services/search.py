"""Search: bring the index up to date, then find the chunks closest to a query."""

import asyncio
import time
from collections import defaultdict
from pathlib import Path

from semantic_code_mcp.domain.interfaces import Embedder, VectorStore
from semantic_code_mcp.domain.models import IndexResult, SearchReport, SearchResult, SearchTimings
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
HIGHEST_SCORE = 1.0

# Overfetch 2x because score filtering and deduplication typically remove 30-50% of
# raw results; 2x keeps the pipeline simple while enough candidates fill the limit.
SEARCH_OVERFETCH_FACTOR = 2

# How far along a search is at each step, out of the progress total
CHECKING_INDEX_PROGRESS = 5
INDEXING_PROGRESS = 10
LOADING_MODEL_PROGRESS = 80
SEARCHING_PROGRESS = 85
DONE_PROGRESS = 100


class SearchService:
    """Answers searches over one project, indexing what changed first."""

    def __init__(self, store: VectorStore, embedder: Embedder, indexing: IndexingService) -> None:
        self._store = store
        self._embedder = embedder
        self._indexing = indexing

    async def search(
        self, query: str, project_root: Path, *, limit: int, progress: ProgressReporter
    ) -> SearchReport:
        """Index stale files, then return the best hits for a query."""
        await progress.report(CHECKING_INDEX_PROGRESS, "Checking index...")
        status = await self._indexing.status(project_root)

        index_result: IndexResult | None = None
        if not status.is_indexed or status.stale_files:
            reason = (
                f"Re-indexing {len(status.stale_files)} stale files..."
                if status.is_indexed
                else "Index not found, indexing..."
            )
            await progress.report(INDEXING_PROGRESS, reason)
            index_result = await self._indexing.index(project_root, force=False, progress=progress)

        if not self._embedder.is_loaded:
            await progress.report(LOADING_MODEL_PROGRESS, "Loading the embedding model...")
        await progress.report(SEARCHING_PROGRESS, "Searching...")
        report = await asyncio.to_thread(self._run_search, query, limit, index_result)
        await progress.report(DONE_PROGRESS, f"Found {len(report.results)} results")
        return report

    def _run_search(self, query: str, limit: int, index_result: IndexResult | None) -> SearchReport:
        started = time.perf_counter()

        embedding_started = time.perf_counter()
        query_embedding = self._embedder.embed_text(query)
        embedding_ms = elapsed_milliseconds(embedding_started)

        search_started = time.perf_counter()
        raw_results = self._store.search_hybrid(
            query_embedding, query, limit * SEARCH_OVERFETCH_FACTOR, VECTOR_WEIGHT
        )
        search_ms = elapsed_milliseconds(search_started)

        kept = [result for result in raw_results if result.score >= MIN_SCORE]
        now = time.time()
        boosted = sorted(
            (with_recency_boost(result, now) for result in kept),
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
            index_result=index_result,
        )


def with_recency_boost(result: SearchResult, now: float) -> SearchResult:
    """A hit whose file changed within the last week, scored up by as much as MAX_RECENCY_BOOST."""
    try:
        age_seconds = now - Path(result.file_path).stat().st_mtime
    except OSError:
        return result
    if age_seconds >= ONE_WEEK_SECONDS:
        return result
    boost = MAX_RECENCY_BOOST * (1 - age_seconds / ONE_WEEK_SECONDS)
    return result.with_score(min(HIGHEST_SCORE, result.score + boost))


def group_by_file(results: list[SearchResult]) -> list[SearchResult]:
    """Hits grouped by file, the file with the best hit first, each file's hits in order."""
    by_file: dict[str, list[SearchResult]] = defaultdict(list)
    for result in results:
        by_file[result.file_path].append(result)
    ordered_files = sorted(
        by_file, key=lambda file_path: max(hit.score for hit in by_file[file_path]), reverse=True
    )
    return [hit for file_path in ordered_files for hit in by_file[file_path]]
