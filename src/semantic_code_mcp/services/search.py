"""Search: bring the index up to date, then find the chunks closest to a query.

Each channel of the store brings its candidates, the vector channel by meaning and the
keyword channel by words, and fusion ranks them as one list, which the search returns
as it stands: hits of one file are not grouped. A hit far in meaning from the query is
noise and goes, and one whose words match the query's goes only further away.
"""

import asyncio
import time
from enum import Enum
from pathlib import Path

from semantic_code_mcp.domain.interfaces import Embedder, IndexStore
from semantic_code_mcp.domain.models import (
    IndexReport,
    IndexStatus,
    SearchHit,
    SearchReport,
    SearchTimings,
)
from semantic_code_mcp.services.fusion import fuse
from semantic_code_mcp.services.indexing import IndexingService
from semantic_code_mcp.services.progress import ProgressReporter
from semantic_code_mcp.services.timing import elapsed_milliseconds

# A hit no keyword matched is kept from this cosine similarity to the query up: the
# blended score's floor of 0.3 kept such a hit, which scored (1 + similarity) / 4, from
# a similarity of 0.2
MIN_SIMILARITY = 0.2
# A hit whose words match the query's is kept from this lower similarity up: an
# identifier's own chunk can sit at 0.2, while a word a query shares by chance, as the
# "fresh" of a cake recipe, finds chunks at -0.1 to 0.05 (a review's probes)
MIN_KEYWORD_SIMILARITY = 0.1
# Each channel brings this many candidates per result asked for, so fusion can lift a
# chunk one channel ranks below the limit
CANDIDATES_PER_RESULT = 2

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
        if status.stale_files_count:
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
        report = await asyncio.to_thread(self._run_search, query, limit, index_report)
        await progress.report(SearchPhase.done.value, f"Found {len(report.results)} results")
        return report

    def _run_search(self, query: str, limit: int, index_report: IndexReport | None) -> SearchReport:
        started = time.perf_counter()

        embedding_started = time.perf_counter()
        query_vector = self._embedder.embed_query(query)
        embedding_ms = elapsed_milliseconds(embedding_started)

        search_started = time.perf_counter()
        candidates = self._store.search_candidates(
            query_vector, query, limit * CANDIDATES_PER_RESULT
        )
        hits = fuse(candidates)
        search_ms = elapsed_milliseconds(search_started)

        kept = [hit for hit in hits if is_relevant(hit)]
        return SearchReport(
            results=kept[:limit],
            filtered_count=len(hits) - len(kept),
            timings=SearchTimings(
                embedding_ms=embedding_ms,
                search_ms=search_ms,
                total_ms=elapsed_milliseconds(started),
            ),
            index_report=index_report,
        )


def indexing_message(status: IndexStatus) -> str:
    """What a search tells the client it is doing before it indexes."""
    if not status.is_indexed:
        return "Index not found, indexing..."
    return f"Re-indexing {status.stale_files_count} stale files..."


def is_relevant(hit: SearchHit) -> bool:
    """Whether a hit stays: its meaning near enough the query's, and less near where its
    words match the query's too."""
    floor = MIN_SIMILARITY if hit.keyword_rank is None else MIN_KEYWORD_SIMILARITY
    return hit.similarity >= floor
