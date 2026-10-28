"""Vectors: give the chunks of a project's index the model's vectors they lack, inside a
search by meaning.

A run takes the chunks the store lists without the model's vectors, embeds them
EMBEDDING_SLICE_SIZE at a time with a report before each, the model loaded first if it is
not, and adds each slice's vectors as it goes: a long run shows it is moving, and a run
cancelled midway keeps the slices it added, which the next search by meaning does not embed
again. The store keeps a chunk's vectors by its id, which digests its text, so an edit leaves
only the chunks it changed to embed.
"""

import asyncio
import time

import structlog
from pydantic import BaseModel, ConfigDict

from semantic_code_mcp.domain.interfaces import Embedder, IndexStore
from semantic_code_mcp.domain.models import Chunk, VectorIdentity
from semantic_code_mcp.services.progress import PROGRESS_START, ProgressReporter, share_done
from semantic_code_mcp.services.timing import elapsed_milliseconds

log = structlog.get_logger()

# Chunks embedded between two progress reports and added in one write, about ten seconds on
# a laptop CPU
EMBEDDING_SLICE_SIZE = 256


class EmbeddingReport(BaseModel):
    """What one run of embedding did."""

    model_config = ConfigDict(frozen=True)

    chunks_embedded: int


class VectorService:
    """Gives one project's chunks the vectors of the server's model.

    One instance serves one project. Its asyncio lock keeps two runs of that project from
    embedding the same chunks at once: the second waits, then finds them embedded.
    """

    def __init__(self, embedder: Embedder, store: IndexStore, model: VectorIdentity) -> None:
        self._embedder = embedder
        self._store = store
        self._model = model
        self._lock = asyncio.Lock()

    async def embed_missing(self, *, progress: ProgressReporter) -> EmbeddingReport:
        """Embed and add the vectors of every chunk the index holds without the model's."""
        async with self._lock:
            chunks = await asyncio.to_thread(self._store.chunks_without_vectors, self._model)
            if chunks:
                await self._embed(chunks, progress)
            return EmbeddingReport(chunks_embedded=len(chunks))

    async def _embed(self, chunks: list[Chunk], progress: ProgressReporter) -> None:
        """Embed the chunks and add their vectors a slice at a time, a report before each."""
        started = time.perf_counter()
        await self._load_model(progress)
        total = len(chunks)
        for slice_start in range(0, total, EMBEDDING_SLICE_SIZE):
            embedding_slice = chunks[slice_start : slice_start + EMBEDDING_SLICE_SIZE]
            await progress.report(
                share_done(slice_start, total),
                f"Embedding {len(embedding_slice)} chunks, {slice_start} of {total} done...",
            )
            await asyncio.to_thread(self._embed_and_add, embedding_slice)
        log.debug("embedding_completed", chunks=total, duration_ms=elapsed_milliseconds(started))

    def _embed_and_add(self, chunks: list[Chunk]) -> None:
        """Embed a slice of chunks and add their vectors to the index."""
        vectors = self._embedder.embed_documents([chunk.content for chunk in chunks])
        self._store.add_vectors(self._model, [chunk.chunk_id for chunk in chunks], vectors)

    async def _load_model(self, progress: ProgressReporter) -> None:
        """Load the model in a worker thread, reported before it embeds, if it is not."""
        if self._embedder.is_loaded:
            return
        await progress.report(PROGRESS_START, "Loading the embedding model...")
        await asyncio.to_thread(self._embedder.load)
