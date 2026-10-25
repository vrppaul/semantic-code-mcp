"""Indexing: find a project's files, see what changed, then chunk, embed and store them.

A run plans before it changes anything: the records discovery takes of the files on
disk, compared with the records the index keeps, give the files to index and the paths
gone; the manifest of what built the index says whether it must be built again whole.
Nothing is removed until the new chunks are embedded, and the store then replaces what
changed in one write, so a reader finds the old index or the new one, or, during a
whole-index write, for a moment an empty one.
"""

import asyncio
import time
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

import numpy
import structlog
from pydantic import BaseModel, ConfigDict

from semantic_code_mcp.discovery import SourceDiscovery
from semantic_code_mcp.domain.errors import IndexRebuiltElsewhereError
from semantic_code_mcp.domain.interfaces import Chunker, Embedder, IndexStore, VectorMatrix
from semantic_code_mcp.domain.models import (
    Chunk,
    IndexCounts,
    IndexDelta,
    IndexIdentity,
    IndexManifest,
    IndexReport,
    IndexStatus,
    IndexUpdate,
    RebuildReason,
    SourceFile,
    WriteScope,
)
from semantic_code_mcp.services.progress import PROGRESS_TOTAL, ProgressReporter
from semantic_code_mcp.services.timing import elapsed_milliseconds

log = structlog.get_logger()

DURATION_DECIMALS = 3
# Chunks embedded between two progress reports: about ten seconds on a laptop CPU
EMBEDDING_SLICE_SIZE = 256
# The vectors of no chunks: no rows, and with no vector seen, no columns
NO_VECTORS_SHAPE = (0, 0)


class IndexPhase(Enum):
    """The steps of an index run, each worth how far along the run is when it starts."""

    scanning = 5.0
    planning = 10.0
    chunking = 20.0
    loading_model = 60.0
    embedding = 70.0
    writing = 95.0


class IndexPlan(BaseModel):
    """What one index run is to do, worked out before it changes anything."""

    model_config = ConfigDict(frozen=True)

    delta: IndexDelta
    scope: WriteScope
    # Set when the index is built again whole because something it was built from changed
    rebuild_reason: RebuildReason | None

    @property
    def needs_write(self) -> bool:
        return self.delta.has_work or self.rebuild_reason is not None


class StoredState(BaseModel):
    """What the store holds before a run, as far as planning needs it."""

    model_config = ConfigDict(frozen=True)

    manifest: IndexManifest | None
    counts: IndexCounts


class ChunkedFile(BaseModel):
    """A file indexed in this run, and the chunks it gave."""

    model_config = ConfigDict(frozen=True)

    file: SourceFile
    chunks: list[Chunk]


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
        """Bring the index up to date with the files: all of them when forced or when the
        index must be built again, else what changed."""
        async with self._lock:
            started = time.perf_counter()
            try:
                return await self._index(project_root, started, force=force, progress=progress)
            except IndexRebuiltElsewhereError:
                # Planned again, the run now finds the identity it differs from and rebuilds;
                # it reports in what is left of the band, so progress never goes back
                log.info("index_rebuilt_elsewhere_planning_again", project=str(project_root))
                remaining = progress.within(IndexPhase.writing.value, PROGRESS_TOTAL)
                return await self._index(project_root, started, force=force, progress=remaining)

    async def status(self, project_root: Path) -> IndexStatus:
        """How far the project's index is from its files, worked out off the event loop."""
        return await asyncio.to_thread(self._status, project_root)

    def _status(self, project_root: Path) -> IndexStatus:
        """How far the project's index is from its files: how much it holds, how many files
        changed since (new, modified or deleted), and whether it must be built again."""
        files = self._discovery.discover(project_root)
        state = self._stored_state()
        plan = self._plan(files, state, force=False)
        return IndexStatus(
            files_count=state.counts.files,
            chunks_count=state.counts.chunks,
            stale_files_count=len(plan.delta.to_index) + len(plan.delta.to_delete),
            manifest=state.manifest,
            rebuild_reason=plan.rebuild_reason,
        )

    async def _index(
        self, project_root: Path, started: float, *, force: bool, progress: ProgressReporter
    ) -> IndexReport:
        """One attempt at a run, timed from `started`, when the run began."""
        await progress.report(IndexPhase.scanning.value, "Scanning files...")
        files = await asyncio.to_thread(self._discovery.discover, project_root)
        log.debug("scanned_files", project=str(project_root), count=len(files))

        await progress.report(IndexPhase.planning.value, f"Found {len(files)} files, planning...")
        state = await asyncio.to_thread(self._stored_state)
        plan = await asyncio.to_thread(self._plan, files, state, force=force)
        log.debug(
            "index_planned",
            to_index=len(plan.delta.to_index),
            to_delete=len(plan.delta.to_delete),
            scope=plan.scope,
            rebuild_reason=plan.rebuild_reason,
        )

        chunks: list[Chunk] = []
        if plan.needs_write:
            await progress.report(
                IndexPhase.chunking.value, f"Chunking {len(plan.delta.to_index)} files..."
            )
            chunked = await asyncio.to_thread(self._chunk, plan.delta.to_index)
            chunks = [chunk for chunked_file in chunked for chunk in chunked_file.chunks]
            vectors = await self._embed(chunks, progress)
            await progress.report(IndexPhase.writing.value, "Writing the index...")
            update = IndexUpdate(
                scope=plan.scope,
                files=[chunked_file.file for chunked_file in chunked],
                chunks=chunks,
                removed_paths=plan.delta.to_delete,
                manifest=IndexManifest(identity=self._identity(), indexed_at=datetime.now(UTC)),
            )
            await asyncio.to_thread(self._store.write, update, vectors)

        return IndexReport(
            files_indexed=len(plan.delta.to_index),
            chunks_indexed=len(chunks),
            files_deleted=len(plan.delta.to_delete),
            duration_seconds=round(time.perf_counter() - started, DURATION_DECIMALS),
            rebuild_reason=plan.rebuild_reason,
        )

    def _stored_state(self) -> StoredState:
        return StoredState(manifest=self._store.manifest(), counts=self._store.counts())

    def _plan(self, files: list[SourceFile], state: StoredState, *, force: bool) -> IndexPlan:
        """What a run over these files is to do, from what the store holds.

        An index that must be built again is not read further: its records may be of
        another layout, and every file is indexed anyway.
        """
        reason = rebuild_reason(state.manifest, counts=state.counts, expected=self._identity())
        if reason is not None:
            return IndexPlan(
                delta=IndexDelta(to_index=files, to_delete=[]),
                scope=WriteScope.everything,
                rebuild_reason=reason,
            )
        delta = plan_delta(files, self._store.indexed_files(), every_file=force)
        scope = WriteScope.everything if force else WriteScope.changed_files
        return IndexPlan(delta=delta, scope=scope, rebuild_reason=None)

    def _identity(self) -> IndexIdentity:
        """What an index built now is built from."""
        return IndexIdentity(
            model_name=self._embedder.model_name,
            chunker_version=self._chunker.version,
            schema_version=self._store.schema_version,
        )

    def _chunk(self, files: list[SourceFile]) -> list[ChunkedFile]:
        """Chunk files one after another; tree-sitter holds the GIL, so threads gain nothing."""
        started = time.perf_counter()
        chunked = [
            ChunkedFile(file=file, chunks=self._chunker.chunk_file(str(file.path)))
            for file in files
        ]
        log.debug(
            "chunking_completed",
            files=len(files),
            chunks=sum(len(chunked_file.chunks) for chunked_file in chunked),
            duration_ms=elapsed_milliseconds(started),
        )
        return chunked

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


def plan_delta(
    on_disk: list[SourceFile], indexed: list[SourceFile], *, every_file: bool
) -> IndexDelta:
    """The files to index — every file, or those new or changed since their record — and
    the recorded paths no longer on disk."""
    recorded = {file.path: file for file in indexed}
    paths_on_disk = {source.path for source in on_disk}
    to_index = (
        on_disk
        if every_file
        else [source for source in on_disk if recorded.get(source.path) != source]
    )
    to_delete = sorted(path for path in recorded if path not in paths_on_disk)
    return IndexDelta(to_index=to_index, to_delete=to_delete)


def rebuild_reason(
    manifest: IndexManifest | None, *, counts: IndexCounts, expected: IndexIdentity
) -> RebuildReason | None:
    """Why the index must be built again whole, or None when it can be brought up to date
    file by file. A store with nothing in it and no manifest is new, not out of date."""
    if manifest is None:
        holds_data = counts.files > 0 or counts.chunks > 0
        return RebuildReason.unversioned if holds_data else None
    built = manifest.identity
    if built.schema_version != expected.schema_version:
        return RebuildReason.schema_changed
    if built.model_name != expected.model_name:
        return RebuildReason.model_changed
    if built.chunker_version != expected.chunker_version:
        return RebuildReason.chunker_changed
    if counts.files == 0 and counts.chunks > 0:
        return RebuildReason.incomplete
    return None
