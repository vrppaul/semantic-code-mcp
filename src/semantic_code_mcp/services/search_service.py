"""Search service - orchestrates search operations."""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from semantic_code_mcp.models import IndexResult, SearchResult
from semantic_code_mcp.protocols import EmbedderProtocol, ProgressCallback, VectorStoreProtocol
from semantic_code_mcp.services.index_service import IndexService

# Recency: files edited within the last week get a small score boost (up to +5%)
# to surface recently-touched code. 5% is small enough not to override relevance.
ONE_WEEK_SECONDS = 7 * 24 * 60 * 60
MAX_RECENCY_BOOST = 0.05

# Overfetch 2x because score filtering (min_score) and deduplication typically
# remove 30-50% of raw results. 2x keeps the pipeline simple while ensuring
# enough candidates survive to fill the requested limit.
SEARCH_OVERFETCH_FACTOR = 2


@dataclass
class SearchOutcome:
    """Domain result from search operation."""

    results: list[SearchResult]
    raw_count: int
    filtered_count: int
    index_result: IndexResult = field(
        default_factory=lambda: IndexResult(
            files_indexed=0, chunks_indexed=0, files_deleted=0, duration_seconds=0.0
        )
    )
    embedding_ms: float = 0.0
    search_ms: float = 0.0
    total_ms: float = 0.0


class SearchService:
    """Orchestrates search operations including indexing."""

    def __init__(
        self,
        store: VectorStoreProtocol,
        embedder: EmbedderProtocol,
        index_service: IndexService,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.index_service = index_service

    async def search(
        self,
        query: str,
        project_path: Path,
        limit: int = 10,
        min_score: float = 0.3,
        vector_weight: float = 0.5,
        on_progress: ProgressCallback | None = None,
    ) -> SearchOutcome:
        """Search with optional auto-indexing and progress callbacks.

        Args:
            query: Natural language search query.
            project_path: Root directory of the project.
            limit: Maximum number of results.
            min_score: Minimum similarity score threshold.
            vector_weight: Weight for vector vs full-text search.
            on_progress: Optional callback matching ctx.report_progress(progress, total, message).

        Returns:
            SearchOutcome with results and timing info.
        """

        async def _progress(percent: float, message: str) -> None:
            if on_progress is not None:
                await on_progress(percent, 100, message)

        await _progress(5, "Checking index...")

        # Check if indexing needed
        status = self.index_service.indexer.get_status(project_path)
        index_result: IndexResult | None = None

        needs_index = not status.is_indexed
        needs_reindex = status.is_indexed and bool(status.stale_files)

        if needs_index or needs_reindex:
            reason = (
                "Index not found, indexing..."
                if needs_index
                else f"Re-indexing {len(status.stale_files)} stale files..."
            )
            await _progress(10, reason)

            index_result = await self.index_service.index(project_path, force=False)

        await _progress(85, "Searching...")

        outcome = await asyncio.to_thread(self._do_search, query, limit, min_score, vector_weight)
        if index_result is not None:
            outcome.index_result = index_result

        await _progress(100, f"Found {len(outcome.results)} results")

        return outcome

    def _do_search(
        self,
        query: str,
        limit: int,
        min_score: float,
        vector_weight: float,
    ) -> SearchOutcome:
        """Internal search implementation."""
        total_start = time.perf_counter()

        # Embed query
        t0 = time.perf_counter()
        query_embedding = self.embedder.embed_text(query)
        embedding_ms = (time.perf_counter() - t0) * 1000

        # Hybrid search
        t0 = time.perf_counter()
        raw_results = self.store.search_hybrid(
            query_embedding,
            query,
            limit * SEARCH_OVERFETCH_FACTOR,
            vector_weight,
        )
        search_ms = (time.perf_counter() - t0) * 1000

        # Filter low-confidence results
        filtered = [r for r in raw_results if r.score >= min_score]

        # Apply recency boost
        boosted = self._apply_recency_boost(filtered)

        # Sort by boosted score, take limit
        boosted.sort(key=lambda x: x[1], reverse=True)
        top_results = [r for r, _ in boosted[:limit]]

        # Group by file
        grouped = self._group_by_file(top_results)

        total_ms = (time.perf_counter() - total_start) * 1000

        return SearchOutcome(
            results=grouped,
            raw_count=len(raw_results),
            filtered_count=len(raw_results) - len(filtered),
            embedding_ms=round(embedding_ms, 1),
            search_ms=round(search_ms, 1),
            total_ms=round(total_ms, 1),
        )

    def _apply_recency_boost(self, results: list[SearchResult]) -> list[tuple[SearchResult, float]]:
        """Apply recency boost to results."""
        boosted = []
        now = time.time()

        for r in results:
            try:
                mtime = Path(r.file_path).stat().st_mtime
            except OSError:
                mtime = None

            recency_boost = 0.0
            if mtime is not None:
                age_seconds = now - mtime
                if age_seconds < ONE_WEEK_SECONDS:
                    recency_boost = MAX_RECENCY_BOOST * (1 - age_seconds / ONE_WEEK_SECONDS)

            boosted_score = min(1.0, r.score + recency_boost)
            boosted.append((r, boosted_score))

        return boosted

    def _group_by_file(self, results: list[SearchResult]) -> list[SearchResult]:
        """Group results by file, ordered by best score."""
        by_file: dict[str, list[SearchResult]] = defaultdict(list)
        for r in results:
            by_file[r.file_path].append(r)

        # Sort files by their best chunk's score
        sorted_files = sorted(
            by_file.keys(),
            key=lambda f: max(r.score for r in by_file[f]),
            reverse=True,
        )

        # Flatten
        grouped = []
        for f in sorted_files:
            grouped.extend(by_file[f])

        return grouped
