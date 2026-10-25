"""Indexing: find a project's files, see what changed, then chunk, embed and store them."""

import asyncio
import threading
import time
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

import numpy
import structlog

from semantic_code_mcp.discovery import SourceDiscovery
from semantic_code_mcp.domain.interfaces import Chunker, Embedder, VectorMatrix, VectorStore
from semantic_code_mcp.domain.models import Chunk, IndexResult, IndexStatus, ScanPlan
from semantic_code_mcp.services.progress import PROGRESS_TOTAL, ProgressReporter
from semantic_code_mcp.services.timing import elapsed_milliseconds
from semantic_code_mcp.storage.cache import CACHE_FILENAME, FileChangeCache

log = structlog.get_logger()

DURATION_DECIMALS = 3
# Chunks embedded between two progress reports: about ten seconds on a laptop CPU
EMBEDDING_SLICE_SIZE = 256
# The vectors of no chunks: no rows, and with no vector seen, no columns
NO_VECTORS_SHAPE = (0, 0)


class IndexPhase(Enum):
    """The steps of an index run, each worth how far along the run is when it starts."""

    scanning = 5.0
    detecting_changes = 10.0
    chunking = 20.0
    loading_model = 60.0
    embedding = 70.0
    writing = 95.0


class IndexingService:
    """Keeps one project's index in step with its files.

    One instance serves one project. Its asyncio lock keeps two index runs of that
    project from planning and embedding at once. A cancelled request releases that
    lock while its worker thread may still be writing, so the write itself is one call
    in one thread, under a thread lock: the next run writes only once it has landed.
    Two writes side by side would store every changed file's chunks twice.
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
        self._writing = threading.Lock()

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

        await progress.report(IndexPhase.scanning.value, "Scanning files...")
        files = await asyncio.to_thread(self.scan_files, project_root)

        await progress.report(
            IndexPhase.detecting_changes.value,
            f"Found {len(files)} files, detecting changes...",
        )
        plan = await asyncio.to_thread(self.detect_changes, files, force=force)

        chunks: list[Chunk] = []
        if plan.has_work:
            await progress.report(
                IndexPhase.chunking.value, f"Chunking {len(plan.files_to_index)} files..."
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
        """Embed the new chunks, then replace the chunks of every changed or deleted file
        and record the files; the old chunks stay searchable while the new ones embed."""
        vectors = await self._embed(chunks, progress)
        await progress.report(IndexPhase.writing.value, "Writing the index...")
        await asyncio.to_thread(self._write, plan, chunks, vectors)

    def _write(self, plan: ScanPlan, chunks: list[Chunk], vectors: VectorMatrix) -> None:
        """Replace the chunks of every changed or deleted file and record the files, as one
        unit that a cancelled request's thread finishes before another run's begins."""
        with self._writing:
            self._store.delete_by_files([*plan.files_to_delete, *plan.files_to_index])
            self._store.add_chunks(chunks, vectors)
            self._record_files(plan)

    async def _embed(self, chunks: list[Chunk], progress: ProgressReporter) -> VectorMatrix:
        """The vectors of the chunks, embedded a slice at a time with a report before each,
        so a long run shows it is moving."""
        if not chunks:
            return numpy.empty(NO_VECTORS_SHAPE, dtype=numpy.float32)
        if not self._embedder.is_loaded:
            await progress.report(IndexPhase.loading_model.value, "Loading the embedding model...")
            await asyncio.to_thread(self._embedder.load)

        started = time.perf_counter()
        texts = [chunk.content for chunk in chunks]
        band = progress.within(IndexPhase.embedding.value, IndexPhase.writing.value)
        slices: list[VectorMatrix] = []
        for slice_start in range(0, len(texts), EMBEDDING_SLICE_SIZE):
            texts_slice = texts[slice_start : slice_start + EMBEDDING_SLICE_SIZE]
            slice_end = slice_start + len(texts_slice)
            await band.report(
                slice_start / len(texts) * PROGRESS_TOTAL,
                f"Embedding chunks {slice_start + 1}-{slice_end} of {len(texts)}...",
            )
            slices.append(await asyncio.to_thread(self._embedder.embed_documents, texts_slice))
        log.debug(
            "embedding_completed", chunks=len(chunks), duration_ms=elapsed_milliseconds(started)
        )
        return numpy.concatenate(slices)

    def _record_files(self, plan: ScanPlan) -> None:
        """Record in the cache which files the index now holds, and drop the deleted."""
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
