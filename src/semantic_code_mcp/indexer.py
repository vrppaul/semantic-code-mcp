"""Indexer — embed and store code chunks."""

import asyncio
import time

import structlog

from semantic_code_mcp.models import Chunk, ChunkWithEmbedding, ScanPlan
from semantic_code_mcp.protocols import EmbedderProtocol, VectorStoreProtocol

log = structlog.get_logger()


class Indexer:
    """Embeds and stores code chunks.

    Pure data pipeline: delete stale → embed new → store.
    Cache bookkeeping is handled by the caller (IndexService).
    """

    def __init__(
        self,
        embedder: EmbedderProtocol,
        store: VectorStoreProtocol,
    ) -> None:
        self.embedder = embedder
        self.store = store

    async def embed_and_store(self, plan: ScanPlan, chunks: list[Chunk]) -> None:
        """Delete stale chunks, embed new ones, and store them.

        Args:
            plan: The ScanPlan describing what to delete/index.
            chunks: Chunks extracted from files_to_index.
        """
        # Delete chunks for removed files
        for file_path in plan.files_to_delete:
            self.store.delete_by_file(file_path)

        # Delete old chunks for files being re-indexed
        for file_path in plan.files_to_index:
            self.store.delete_by_file(file_path)

        if chunks:
            await self._embed_and_store(chunks)

    async def _embed_and_store(self, chunks: list[Chunk]) -> None:
        """Generate embeddings and store chunks."""
        contents = [chunk.content for chunk in chunks]
        t0 = time.time()
        embeddings = await asyncio.to_thread(self.embedder.embed_batch, contents)
        log.debug(
            "embedding_completed",
            chunks=len(chunks),
            duration_ms=round((time.time() - t0) * 1000, 1),
        )

        items = [
            ChunkWithEmbedding(chunk=chunk, embedding=embedding)
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]

        t0 = time.time()
        await asyncio.to_thread(self.store.add_chunks, items)
        log.debug(
            "storage_completed",
            chunks=len(items),
            duration_ms=round((time.time() - t0) * 1000, 1),
        )

    def clear_store(self) -> None:
        """Delete all chunks from the store."""
        self.store.clear()

    def get_store_stats(self) -> tuple[list[str], int]:
        """Get indexed files and total chunk count.

        Returns:
            Tuple of (indexed file paths, chunk count).
        """
        return self.store.get_indexed_files(), self.store.count()
