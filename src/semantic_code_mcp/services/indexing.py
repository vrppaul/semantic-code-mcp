"""Indexing: find a project's files, see what changed, then chunk, embed and store them."""

import asyncio
import time
from datetime import UTC, datetime
from enum import StrEnum, auto
from pathlib import Path

import structlog

from semantic_code_mcp.discovery import SourceDiscovery
from semantic_code_mcp.domain.interfaces import Chunker, Embedder, VectorStore
from semantic_code_mcp.domain.models import (
    Chunk,
    ChunkWithEmbedding,
    IndexResult,
    IndexStatus,
    ScanPlan,
)
from semantic_code_mcp.services.progress import ProgressReporter
from semantic_code_mcp.services.timing import elapsed_milliseconds
from semantic_code_mcp.storage.cache import CACHE_FILENAME, FileChangeCache

log = structlog.get_logger()

DURATION_DECIMALS = 3


class IndexPhase(StrEnum):
    """The steps of an index run, each reported to the client as it starts."""

    scanning = auto()
    detecting_changes = auto()
    chunking = auto()
    loading_model = auto()
    embedding = auto()


# How far along an index run is when each phase starts, out of PROGRESS_TOTAL
PHASE_PROGRESS: dict[IndexPhase, float] = {
    IndexPhase.scanning: 5,
    IndexPhase.detecting_changes: 10,
    IndexPhase.chunking: 20,
    IndexPhase.loading_model: 60,
    IndexPhase.embedding: 70,
}


class IndexingService:
    """Keeps one project's index in step with its files.

    One instance serves one project; its lock keeps two index runs of that project
    from writing at once, which would store every changed file's chunks twice.
    """

    def __init__(
        self,
        discovery: SourceDiscovery,
        chunker: Chunker,
        embedder: Embedder,
        store: VectorStore,
        cache_directory: Path,
    ) -> None:
        self._discovery = discovery
        self._chunker = chunker
        self._embedder = embedder
        self._store = store
        self._cache_directory = cache_directory
        self._lock = asyncio.Lock()

    async def index(
        self, project_root: Path, *, force: bool, progress: ProgressReporter
    ) -> IndexResult:
        """Bring the index up to date with the files: all of them when forced, else what changed."""
        async with self._lock:
            return await self._index(project_root, force=force, progress=progress)

    async def _index(
        self, project_root: Path, *, force: bool, progress: ProgressReporter
    ) -> IndexResult:
        started = time.perf_counter()

        await progress.report(PHASE_PROGRESS[IndexPhase.scanning], "Scanning files...")
        files = await asyncio.to_thread(self.scan_files, project_root)

        await progress.report(
            PHASE_PROGRESS[IndexPhase.detecting_changes],
            f"Found {len(files)} files, detecting changes...",
        )
        plan = await asyncio.to_thread(self.detect_changes, files, force=force)

        chunks: list[Chunk] = []
        if plan.has_work:
            await progress.report(
                PHASE_PROGRESS[IndexPhase.chunking], f"Chunking {len(plan.files_to_index)} files..."
            )
            chunks = await asyncio.to_thread(self.chunk_files, plan.files_to_index)
            await self._embed_and_store(plan, chunks, progress)

        return IndexResult(
            files_indexed=len(plan.files_to_index),
            chunks_indexed=len(chunks),
            files_deleted=len(plan.files_to_delete),
            duration_seconds=round(time.perf_counter() - started, DURATION_DECIMALS),
        )

    def scan_files(self, project_root: Path) -> list[str]:
        """List the indexable source files of a project as absolute paths."""
        files = [str(path) for path in self._discovery.discover(project_root)]
        log.debug("scanned_files", project=str(project_root), count=len(files))
        return files

    def detect_changes(self, current_files: list[str], *, force: bool) -> ScanPlan:
        """What an index run has to do: every file when forced, else the changed ones.

        Forcing also empties the store and the cache, so nothing of the old index stays.
        """
        cache = FileChangeCache(self._cache_directory)
        if force:
            cache.clear()
            self._store.clear()
            return ScanPlan(files_to_index=current_files, files_to_delete=[])

        changes = cache.get_changes(current_files)
        log.debug(
            "incremental_index",
            new=len(changes.new),
            modified=len(changes.modified),
            deleted=len(changes.deleted),
        )
        return ScanPlan(files_to_index=changes.stale_files, files_to_delete=changes.deleted)

    def chunk_files(self, files: list[str]) -> list[Chunk]:
        """Chunk files one after another; tree-sitter holds the GIL, so threads gain nothing."""
        started = time.perf_counter()
        chunks = [chunk for file_path in files for chunk in self._chunker.chunk_file(file_path)]
        log.debug(
            "chunking_completed",
            files=len(files),
            chunks=len(chunks),
            duration_ms=elapsed_milliseconds(started),
        )
        return chunks

    async def _embed_and_store(
        self, plan: ScanPlan, chunks: list[Chunk], progress: ProgressReporter
    ) -> None:
        """Replace the chunks of every changed or deleted file, then record the files."""
        await asyncio.to_thread(
            self._store.delete_by_files, [*plan.files_to_delete, *plan.files_to_index]
        )

        if chunks:
            if not self._embedder.is_loaded:
                await progress.report(
                    PHASE_PROGRESS[IndexPhase.loading_model], "Loading the embedding model..."
                )
            await progress.report(
                PHASE_PROGRESS[IndexPhase.embedding], f"Embedding {len(chunks)} chunks..."
            )
            started = time.perf_counter()
            embeddings = await asyncio.to_thread(
                self._embedder.embed_batch, [chunk.content for chunk in chunks]
            )
            log.debug(
                "embedding_completed", chunks=len(chunks), duration_ms=elapsed_milliseconds(started)
            )
            items = [
                ChunkWithEmbedding(chunk=chunk, embedding=embedding)
                for chunk, embedding in zip(chunks, embeddings, strict=True)
            ]
            await asyncio.to_thread(self._store.add_chunks, items)

        cache = FileChangeCache(self._cache_directory)
        cache.remove_files(plan.files_to_delete)
        cache.update_files(plan.files_to_index)

    async def status(self, project_root: Path) -> IndexStatus:
        """The project's index status, worked out off the event loop."""
        return await asyncio.to_thread(self.get_status, project_root)

    def get_status(self, project_root: Path) -> IndexStatus:
        """Whether the project is indexed, how much, and which files changed since."""
        cache = FileChangeCache(self._cache_directory)
        stale_files = cache.get_stale_files(self.scan_files(project_root))
        statistics = self._store.statistics()

        cache_file = self._cache_directory / CACHE_FILENAME
        last_updated = None
        if cache_file.exists():
            last_updated = datetime.fromtimestamp(cache_file.stat().st_mtime, tz=UTC)

        return IndexStatus(
            is_indexed=bool(statistics.indexed_files),
            last_updated=last_updated,
            files_count=len(statistics.indexed_files),
            chunks_count=statistics.chunks_count,
            stale_files=stale_files,
        )
