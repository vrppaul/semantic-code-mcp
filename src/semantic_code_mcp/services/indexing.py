"""Indexing: find a project's files, see what changed, then chunk, embed and store them.

A run plans before it changes anything: the records discovery takes of the files on
disk, compared with the records the index keeps and the files its chunks come from,
give the files to index and the paths gone. It then writes a slice of whole files at a
time, each slice's chunks taking the place of what the index held of those files, so a
reader always finds every file's old chunks or its new ones, and a run stopped midway
keeps the slices it wrote. A chunk whose text the index holds already keeps its stored
vector, so an edit embeds only the chunks it changed; a forced run embeds everything
again. The index is the one of this server's model, embedder version and chunker
version; the store keeps those of others apart.
"""

import asyncio
import time
from enum import Enum
from pathlib import Path

import numpy
import structlog
from pydantic import BaseModel, ConfigDict

from semantic_code_mcp.discovery import SourceDiscovery
from semantic_code_mcp.domain.interfaces import (
    Chunker,
    Embedder,
    IndexStore,
    Vector,
    VectorMatrix,
)
from semantic_code_mcp.domain.models import (
    Chunk,
    IndexDelta,
    IndexedFiles,
    IndexReport,
    IndexStatus,
    IndexUpdate,
    SourceFile,
)
from semantic_code_mcp.services.progress import PROGRESS_TOTAL, ProgressReporter
from semantic_code_mcp.services.timing import elapsed_milliseconds

log = structlog.get_logger()

DURATION_DECIMALS = 3
# Chunks embedded between two progress reports, about ten seconds on a laptop CPU; and
# the fewest a write holds, but the last of a run
EMBEDDING_SLICE_SIZE = 256
# The vectors of no chunks: no rows, and with no vector seen, no columns
NO_VECTORS_SHAPE = (0, 0)


class IndexPhase(Enum):
    """The steps of an index run, each worth how far along the run is when it starts.

    Embedding and writing share the band from `embedding` to the end, a slice at a time;
    the model loads inside it, when the first chunk needs embedding."""

    scanning = 5.0
    planning = 10.0
    chunking = 20.0
    embedding = 30.0


class ChunkedFile(BaseModel):
    """A file indexed in this run, and the chunks it gave."""

    model_config = ConfigDict(frozen=True)

    file: SourceFile
    chunks: list[Chunk]


class WriteSlice(BaseModel):
    """Whole files of a run, written together."""

    model_config = ConfigDict(frozen=True)

    files: list[ChunkedFile]

    @property
    def chunks(self) -> list[Chunk]:
        return [chunk for chunked_file in self.files for chunk in chunked_file.chunks]


class EmbeddingProgress(BaseModel):
    """Where a run's embedding stands: how many of its chunks are behind it."""

    model_config = ConfigDict(frozen=True)

    done: int
    total: int


class IndexingService:
    """Keeps one project's index in step with its files.

    One instance serves one project. Its asyncio lock keeps two index runs of that
    project from planning and embedding at once. A cancelled request releases that lock
    while its worker thread may still be writing; the store's write is a unit no other
    writer interleaves with, so the next run's write waits until that one has landed.
    """

    def __init__(
        self,
        discovery: SourceDiscovery,
        chunker: Chunker,
        embedder: Embedder,
        store: IndexStore,
    ) -> None:
        self._discovery = discovery
        self._chunker = chunker
        self._embedder = embedder
        self._store = store
        self._lock = asyncio.Lock()

    async def index(
        self, project_root: Path, *, force: bool, progress: ProgressReporter
    ) -> IndexReport:
        """Bring the index up to date with the files: all of them when forced, else what
        changed."""
        async with self._lock:
            return await self._index(project_root, force=force, progress=progress)

    async def status(self, project_root: Path) -> IndexStatus:
        """How far the project's index is from its files, worked out off the event loop."""
        return await asyncio.to_thread(self._status, project_root)

    def _status(self, project_root: Path) -> IndexStatus:
        """How far the project's index is from its files: how much it holds, and how many
        files changed since (new, modified or deleted)."""
        delta = self._plan(self._discovery.discover(project_root), force=False)
        counts = self._store.counts()
        return IndexStatus(
            files_count=counts.files,
            chunks_count=counts.chunks,
            stale_files_count=len(delta.to_index) + len(delta.to_delete),
            manifest=self._store.manifest(),
        )

    async def _index(
        self, project_root: Path, *, force: bool, progress: ProgressReporter
    ) -> IndexReport:
        started = time.perf_counter()
        await progress.report(IndexPhase.scanning.value, "Scanning files...")
        files = await asyncio.to_thread(self._discovery.discover, project_root)
        log.debug("scanned_files", project=str(project_root), count=len(files))

        await progress.report(IndexPhase.planning.value, f"Found {len(files)} files, planning...")
        delta = await asyncio.to_thread(self._plan, files, force=force)
        log.debug("index_planned", to_index=len(delta.to_index), to_delete=len(delta.to_delete))

        chunks_indexed = 0
        if delta.has_work:
            await progress.report(
                IndexPhase.chunking.value, f"Chunking {len(delta.to_index)} files..."
            )
            chunked = await asyncio.to_thread(self._chunk, project_root, delta.to_index)
            chunks_indexed = await self._write(
                write_slices(chunked, EMBEDDING_SLICE_SIZE),
                delta.to_delete,
                force=force,
                progress=progress.within(IndexPhase.embedding.value, PROGRESS_TOTAL),
            )

        return IndexReport(
            files_indexed=len(delta.to_index),
            chunks_indexed=chunks_indexed,
            files_deleted=len(delta.to_delete),
            duration_seconds=round(time.perf_counter() - started, DURATION_DECIMALS),
        )

    def _plan(self, files: list[SourceFile], *, force: bool) -> IndexDelta:
        """What a run over these files is to do, from what the store holds."""
        return plan_delta(files, self._store.indexed_files(), every_file=force)

    def _chunk(self, project_root: Path, files: list[SourceFile]) -> list[ChunkedFile]:
        """Chunk files one after another; tree-sitter holds the GIL, so threads gain nothing."""
        started = time.perf_counter()
        chunked = [
            ChunkedFile(file=file, chunks=self._chunker.chunk_file(project_root, file.path))
            for file in files
        ]
        log.debug(
            "chunking_completed",
            files=len(files),
            chunks=sum(len(chunked_file.chunks) for chunked_file in chunked),
            duration_ms=elapsed_milliseconds(started),
        )
        return chunked

    async def _write(
        self,
        slices: list[WriteSlice],
        removed_paths: list[Path],
        *,
        force: bool,
        progress: ProgressReporter,
    ) -> int:
        """Embed and write the slices one after another, the removals with the first and the
        run's end with the last. The number of chunks written."""
        total = sum(len(write_slice.chunks) for write_slice in slices)
        done = 0
        removals = removed_paths
        last_slice = slices[-1]
        for write_slice in slices:
            chunks = write_slice.chunks
            vectors = await self._vectors(
                chunks,
                reuse=not force,
                progress=progress,
                position=EmbeddingProgress(done=done, total=total),
            )
            update = IndexUpdate(
                files=[chunked_file.file for chunked_file in write_slice.files],
                chunks=chunks,
                removed_paths=removals,
                last_of_run=write_slice is last_slice,
            )
            await asyncio.to_thread(self._store.write, update, vectors)
            done += len(chunks)
            removals = []
        return total

    async def _vectors(
        self,
        chunks: list[Chunk],
        *,
        reuse: bool,
        progress: ProgressReporter,
        position: EmbeddingProgress,
    ) -> VectorMatrix:
        """The vectors of a slice's chunks, row by row: the stored one of each chunk the
        index holds already, when reused, and the rest embedded. Stored vectors of another
        width than those embedded now (a model replaced under a name nothing pins, as a
        local path's) are embedded again, after the rest, and the slice's write starts the
        index again."""
        if not chunks:
            return numpy.empty(NO_VECTORS_SHAPE, dtype=numpy.float32)
        stored: dict[str, Vector] = {}
        if reuse:
            chunk_ids = [chunk.chunk_id for chunk in chunks]
            stored = await asyncio.to_thread(self._store.stored_vectors, chunk_ids)
        missing = [chunk for chunk in chunks if chunk.chunk_id not in stored]
        vectors_by_id = await self._embedded_by_id(missing, progress, position)
        if stored and missing and width_of(stored) != width_of(vectors_by_id):
            unusable = [chunk for chunk in chunks if chunk.chunk_id in stored]
            behind = EmbeddingProgress(done=position.done + len(missing), total=position.total)
            stored = await self._embedded_by_id(unusable, progress, behind)
        vectors_by_id.update(stored)
        return numpy.stack([vectors_by_id[chunk.chunk_id] for chunk in chunks])

    async def _embedded_by_id(
        self, chunks: list[Chunk], progress: ProgressReporter, position: EmbeddingProgress
    ) -> dict[str, Vector]:
        """The vectors of the chunks, embedded, by chunk id."""
        embedded = await self._embed(chunks, progress, position)
        return dict(zip([chunk.chunk_id for chunk in chunks], numpy.unstack(embedded), strict=True))

    async def _embed(
        self, chunks: list[Chunk], progress: ProgressReporter, position: EmbeddingProgress
    ) -> VectorMatrix:
        """The vectors of the chunks, embedded EMBEDDING_SLICE_SIZE at a time with a report
        before each, the model loaded first if it is not, so a long run shows it is moving."""
        if not chunks:
            return numpy.empty(NO_VECTORS_SHAPE, dtype=numpy.float32)
        await self._load_model(progress, position)

        started = time.perf_counter()
        texts = [chunk.content for chunk in chunks]
        batches: list[VectorMatrix] = []
        for batch_start in range(0, len(texts), EMBEDDING_SLICE_SIZE):
            batch = texts[batch_start : batch_start + EMBEDDING_SLICE_SIZE]
            behind = position.done + batch_start
            await progress.report(
                behind / position.total * PROGRESS_TOTAL,
                f"Embedding {len(batch)} chunks, {behind} of {position.total} done...",
            )
            batches.append(await asyncio.to_thread(self._embedder.embed_documents, batch))
        log.debug(
            "embedding_completed", chunks=len(chunks), duration_ms=elapsed_milliseconds(started)
        )
        return numpy.concatenate(batches)

    async def _load_model(self, progress: ProgressReporter, position: EmbeddingProgress) -> None:
        """Load the model in a worker thread, reported where the run stands, if it is not."""
        if self._embedder.is_loaded:
            return
        await progress.report(
            position.done / position.total * PROGRESS_TOTAL, "Loading the embedding model..."
        )
        await asyncio.to_thread(self._embedder.load)


def write_slices(chunked: list[ChunkedFile], size: int) -> list[WriteSlice]:
    """The files in order, grouped into slices of at least `size` chunks, the last one
    fewer; one slice, empty, when there are no files, so a run that only removes files
    still writes once."""
    slices: list[WriteSlice] = []
    current: list[ChunkedFile] = []
    current_chunks = 0
    for chunked_file in chunked:
        current.append(chunked_file)
        current_chunks += len(chunked_file.chunks)
        if current_chunks >= size:
            slices.append(WriteSlice(files=current))
            current = []
            current_chunks = 0
    if current or not slices:
        slices.append(WriteSlice(files=current))
    return slices


def width_of(vectors_by_id: dict[str, Vector]) -> int:
    """How wide vectors of one source are: one table's or one model's, all of one width."""
    return len(next(iter(vectors_by_id.values())))


def plan_delta(on_disk: list[SourceFile], indexed: IndexedFiles, *, every_file: bool) -> IndexDelta:
    """The files to index — every file, or those new or changed since their record — and
    the paths no longer on disk that the index holds records or chunks of."""
    recorded = {file.path: file for file in indexed.records}
    paths_on_disk = {source.path for source in on_disk}
    to_index = (
        on_disk
        if every_file
        else [source for source in on_disk if recorded.get(source.path) != source]
    )
    to_delete = sorted((recorded.keys() | set(indexed.unrecorded_paths)) - paths_on_disk)
    return IndexDelta(to_index=to_index, to_delete=to_delete)
