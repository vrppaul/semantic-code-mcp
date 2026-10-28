"""Search: bring the index up to date, then find the chunks that best answer a query.

By default a search reads words alone (ADR 009, D1): the chunks named as the query come
first, as an agent that asks for an identifier it knows wants its definition, and a
qualified name (`Stack.push`) is read as a parent and a name; then the chunks whose words
best match the query's, by BM25. The model is not loaded for it.

A search by meaning (D2) embeds the chunks without the model's vectors first, inside the
call (`vectors.py`). Each channel of the store then brings its candidates, the vector channel
by meaning and the keyword channel by words, and fusion ranks them as one list, which the
search returns as it stands: hits of one file are not grouped. A hit far in meaning from the
query is noise and goes, and one whose words match the query's goes only further away; one
named as the query stays, and so does one without a vector yet, which only its words found.
"""

import asyncio
import time
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from semantic_code_mcp.domain.interfaces import Embedder, IndexStore
from semantic_code_mcp.domain.models import (
    PARENT_SEPARATORS,
    Chunk,
    IndexReport,
    IndexStatus,
    SearchHit,
    SearchQuery,
    SearchReport,
    SearchTimings,
    VectorIdentity,
)
from semantic_code_mcp.services.fusion import fuse
from semantic_code_mcp.services.indexing import IndexingService
from semantic_code_mcp.services.progress import ProgressReporter
from semantic_code_mcp.services.timing import elapsed_milliseconds
from semantic_code_mcp.services.vectors import VectorService

# Each channel of a search by meaning brings this many candidates per result asked for, so
# fusion can lift a chunk one channel ranks below the limit
CANDIDATES_PER_RESULT = 2
# What a search by words drops: nothing, as no floor holds back a hit by words (ADR 009, D5)
NONE_DROPPED = 0
# What a search by words spends embedding: nothing, as it loads no model
NO_EMBEDDING_MILLISECONDS = 0.0

# The timings of a search that had nothing to look through
NO_TIMINGS = SearchTimings(embedding_ms=0.0, search_ms=0.0, total_ms=0.0)


class SearchPhase(Enum):
    """The steps of a search, each worth how far along the search is when it starts."""

    checking_index = 5.0
    indexing = 10.0
    embedding = 30.0
    loading_model = 80.0
    searching = 85.0
    done = 100.0


class SimilarityFloors(BaseModel):
    """How close in meaning to the query a hit of a search by meaning must be to stay: a hit
    no keyword matched, and, less close, one whose words match the query's."""

    model_config = ConfigDict(frozen=True)

    # A hit no keyword matched
    meaning_only: float
    # A hit whose words match the query's
    words_match: float


class NameAsked(BaseModel):
    """A query read as the name of a chunk, its case and the blanks around it aside: the
    whole of it, and where it holds a separator of a qualified name, the name after the last
    one and the parent before it."""

    model_config = ConfigDict(frozen=True)

    whole: str
    name: str | None
    parent: str | None


class SearchService:
    """Answers searches over one project, indexing what changed first."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        store: IndexStore,
        embedder: Embedder,
        model: VectorIdentity,
        indexing: IndexingService,
        vectors: VectorService,
        floors: SimilarityFloors,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._model = model
        self._indexing = indexing
        self._vectors = vectors
        self._floors = floors

    async def search(
        self, query: SearchQuery, project_root: Path, *, progress: ProgressReporter
    ) -> SearchReport:
        """Index stale files, then return the best hits for a query within its scope."""
        await progress.report(SearchPhase.checking_index.value, "Checking index...")
        status = await self._indexing.status(project_root)

        index_report: IndexReport | None = None
        if status.stale_files_count:
            await progress.report(SearchPhase.indexing.value, indexing_message(status))
            index_report = await self._indexing.index(
                project_root,
                force=False,
                progress=progress.within(SearchPhase.indexing.value, SearchPhase.embedding.value),
            )

        # Counted after any index run: one that found its work done by another run indexed
        # nothing, though the index holds chunks
        counts = await asyncio.to_thread(self._store.counts, self._model)
        if counts.chunks == 0:
            # No chunk to find: answer at once, without loading the model for the query
            await progress.report(SearchPhase.done.value, "Nothing indexed")
            return SearchReport(
                results=[], filtered_count=0, timings=NO_TIMINGS, index_report=index_report
            )

        if query.by_meaning:
            await self._prepare_meaning(progress)
            await progress.report(SearchPhase.searching.value, "Searching...")
            report = await asyncio.to_thread(self._search_by_meaning, query, index_report)
        else:
            await progress.report(SearchPhase.searching.value, "Searching...")
            report = await asyncio.to_thread(self._search_by_words, query, index_report)
        await progress.report(SearchPhase.done.value, f"Found {len(report.results)} results")
        return report

    async def _prepare_meaning(self, progress: ProgressReporter) -> None:
        """Embed the chunks without the model's vectors, then load the model for the query
        if embedding did not."""
        await self._vectors.embed_missing(
            progress=progress.within(SearchPhase.embedding.value, SearchPhase.loading_model.value)
        )
        if not self._embedder.is_loaded:
            await progress.report(SearchPhase.loading_model.value, "Loading the embedding model...")
            await asyncio.to_thread(self._embedder.load)

    def _search_by_words(
        self, query: SearchQuery, index_report: IndexReport | None
    ) -> SearchReport:
        started = time.perf_counter()
        named = chunks_of_name(self._store, name_asked(query.text), query)
        ranked = self._store.search_words(query, query.limit)
        search_ms = elapsed_milliseconds(started)
        return SearchReport(
            results=named_first(named, ranked)[: query.limit],
            filtered_count=NONE_DROPPED,
            timings=SearchTimings(
                embedding_ms=NO_EMBEDDING_MILLISECONDS, search_ms=search_ms, total_ms=search_ms
            ),
            index_report=index_report,
        )

    def _search_by_meaning(
        self, query: SearchQuery, index_report: IndexReport | None
    ) -> SearchReport:
        started = time.perf_counter()

        embedding_started = time.perf_counter()
        query_vector = self._embedder.embed_query(query.text)
        embedding_ms = elapsed_milliseconds(embedding_started)

        search_started = time.perf_counter()
        candidates = self._store.search(
            self._model, query_vector, query, query.limit * CANDIDATES_PER_RESULT
        )
        hits = fuse(candidates)
        search_ms = elapsed_milliseconds(search_started)

        asked = name_asked(query.text)
        kept = [hit.chunk for hit in hits if is_relevant(hit, self._floors, asked.whole)]
        return SearchReport(
            results=kept[: query.limit],
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


def is_relevant(hit: SearchHit, floors: SimilarityFloors, asked_name: str) -> bool:
    """Whether a hit stays: one named as the query however far in meaning, which a model
    may find far from a bare name, and one without a vector yet, which its words alone
    found; another when its meaning is near enough the query's, and less near where its
    words match the query's too."""
    if hit.chunk.name.casefold() == asked_name or hit.similarity is None:
        return True
    floor = floors.meaning_only if hit.keyword_rank is None else floors.words_match
    return hit.similarity >= floor


def name_asked(query_text: str) -> NameAsked:
    """The query as a name a chunk could have: whole, and split at the last separator of a
    qualified name any language writes, its case and the blanks around it aside."""
    whole = query_text.strip().casefold()
    split_at = max(
        (
            (whole.rfind(separator), separator)
            for separator in PARENT_SEPARATORS.values()
            if separator in whole
        ),
        default=None,
    )
    if split_at is None:
        return NameAsked(whole=whole, name=None, parent=None)
    position, separator = split_at
    return NameAsked(
        whole=whole,
        name=whole[position + len(separator) :].strip(),
        parent=whole[:position].strip(),
    )


def chunks_of_name(store: IndexStore, asked: NameAsked, query: SearchQuery) -> list[Chunk]:
    """The chunks of the query's scope named as the query, whole or as a parent and a name,
    each once."""
    named = store.chunks_named(asked.whole, query.scope)
    if asked.name is not None:
        named += [
            chunk
            for chunk in store.chunks_named(asked.name, query.scope)
            if chunk.parent is not None and chunk.parent.casefold() == asked.parent
        ]
    return list({chunk.chunk_id: chunk for chunk in named}.values())


def named_first(named: list[Chunk], ranked: list[Chunk]) -> list[Chunk]:
    """The chunks named as the query, those the words ranked in that order and the rest
    after them as they came, then the rest of what the words ranked."""
    position_of = {chunk.chunk_id: position for position, chunk in enumerate(ranked)}
    named_ids = {chunk.chunk_id for chunk in named}
    unranked = len(ranked)
    return [
        *sorted(named, key=lambda chunk: position_of.get(chunk.chunk_id, unranked)),
        *(chunk for chunk in ranked if chunk.chunk_id not in named_ids),
    ]
