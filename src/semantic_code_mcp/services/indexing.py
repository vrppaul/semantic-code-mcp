"""Indexing: find a project's files, see what changed, then chunk and store them with their
words.

A run first holds the index against every other run, in this process or another, so two
sessions on one repository never chunk it twice: one that finds it held tries again every
RUN_RETRY_SECONDS, for as long as the other holds it, and says so; it then plans afresh and
finds the other's work done. A run plans before it changes anything: the records discovery
takes of the files on disk, compared with the records the index keeps and the files its
chunks come from, give the files to index and the paths gone. A search's run takes the
files its stale check found rather than scan the project again (ADR 010, O3), and plans
once it holds the index, as another run may have written since. The run then chunks the files
a batch at a time, in a pool of processes where their source pays its start (`chunking.py`),
and writes them as they come, a slice of whole files at a time (ADR 010, O4): a first index
holds what it has not written yet, never every chunk. Each slice's chunks take the place of
what the index held of those files, so a reader always finds every file's old chunks or its
new ones, and a run stopped midway keeps the slices it wrote. A run embeds nothing: a chunk
whose text the index holds already keeps the vectors it has, by its id, and the rest are
embedded by the first search by meaning (`vectors.py`); a forced run drops the vectors of a
model nothing pins, so that search embeds every chunk again. The index is the one of this
server's chunker version; the store keeps those of others apart.
"""

import asyncio
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path

import structlog
from pydantic import BaseModel, ConfigDict

from semantic_code_mcp.domain.interfaces import IndexStore, ProjectFiles
from semantic_code_mcp.domain.models import (
    FileStamps,
    IndexDelta,
    IndexedFiles,
    IndexFreshness,
    IndexManifest,
    IndexReport,
    IndexStatus,
    IndexUpdate,
    SourceFile,
    VectorIdentity,
)
from semantic_code_mcp.services.chunking import ChunkedFile, Chunking
from semantic_code_mcp.services.progress import (
    PROGRESS_START,
    PROGRESS_TOTAL,
    ProgressReporter,
    share_done,
)

log = structlog.get_logger()

DURATION_DECIMALS = 3
# The fewest chunks a write holds, but the last of a run: a report comes before each
WRITE_SLICE_SIZE = 256
# How often a run that finds another holding the index tries again, saying so once:
# the other's build takes seconds, under load a minute and more, and has no limit (ADR 009, D3)
RUN_RETRY_SECONDS = 1.0
# What a run waiting for another session's tells the client
ANOTHER_SESSION_BUILDS = "Another session is building the index, waiting..."


class IndexPhase(Enum):
    """The steps of an index run, each worth how far along the run is when it starts.

    Chunking and writing take the band from `indexing` to the end, a slice at a time."""

    scanning = 5.0
    planning = 10.0
    indexing = 20.0


class StaleCheck(BaseModel):
    """What a search found of its project before it searches: the files on disk as its scan
    took them, and how far the index is from them. The run it starts plans from these files,
    so a search after an edit scans the project once, unless another run wrote since."""

    model_config = ConfigDict(frozen=True)

    project_root: Path
    files: FileStamps
    freshness: IndexFreshness
    # The index's manifest as the check found it, before it read anything else of the index:
    # every write that changes the index records a new one
    manifest: IndexManifest | None


class IndexingService:
    """Keeps one project's index in step with its files.

    One instance serves one project. Its asyncio lock keeps two index runs of that
    project from planning and writing at once, and the store's hold of the index those of
    other processes. A cancelled request releases both while its worker thread may still be
    writing; the store's write is a unit no other writer interleaves with, so the next run's
    write waits until that one has landed.
    """

    def __init__(
        self,
        files: ProjectFiles,
        chunking: Chunking,
        store: IndexStore,
        model: VectorIdentity,
    ) -> None:
        self._files = files
        self._chunking = chunking
        self._store = store
        self._model = model
        self._lock = asyncio.Lock()

    async def index(
        self, project_root: Path, *, force: bool, progress: ProgressReporter
    ) -> IndexReport:
        """Bring the index up to date with the files: all of them when forced, else what
        changed."""
        async with self._lock, self._held_run(progress):
            started = time.perf_counter()
            files = await self._scan(project_root, progress, rescan=force)
            return await self._index(
                project_root, files, force=force, progress=progress, started=started
            )

    async def index_from_check(
        self, check: StaleCheck, *, progress: ProgressReporter
    ) -> IndexReport:
        """Bring the index up to date with the files a stale check found, never forced: an
        edit since the check is the next search's. Where another run wrote since the check,
        the files are scanned again, as what it wrote of a file changed since is newer than
        the check's view of that file; and the run plans once it holds the index, so what
        another run did is not done again."""
        async with self._lock, self._held_run(progress):
            started = time.perf_counter()
            files = check.files
            if await asyncio.to_thread(self._store.manifest) != check.manifest:
                files = await self._scan(check.project_root, progress, rescan=False)
            return await self._index(
                check.project_root, files, force=False, progress=progress, started=started
            )

    async def _scan(
        self, project_root: Path, progress: ProgressReporter, *, rescan: bool
    ) -> FileStamps:
        """The project's files as they are now, found off the event loop; by a full scan where
        asked, what follows them started again."""
        await progress.report(IndexPhase.scanning.value, "Scanning files...")
        read = self._files.rescanned if rescan else self._files.current
        files = await asyncio.to_thread(read, project_root)
        log.debug("scanned_files", project=str(project_root), count=len(files))
        return files

    @asynccontextmanager
    async def _held_run(self, progress: ProgressReporter) -> AsyncGenerator[None]:
        """Hold the index for a run, trying again while another run holds it. Each try is a
        lock taken without waiting, on the event loop's own thread, so a cancelled run never
        leaves a worker thread to take the lock after it."""
        if not self._store.try_hold_for_run():
            # Reported once: a client's progress must only move on
            await progress.report(PROGRESS_START, ANOTHER_SESSION_BUILDS)
            # Another process's file lock sends no event to wait for: it is tried again
            while not self._store.try_hold_for_run():  # noqa: ASYNC110
                await asyncio.sleep(RUN_RETRY_SECONDS)
        try:
            yield
        finally:
            self._store.release_run()

    async def status(self, project_root: Path) -> IndexStatus:
        """How far the project's index is from its files, worked out off the event loop."""
        return await asyncio.to_thread(self._status, project_root)

    async def stale_check(self, project_root: Path) -> StaleCheck:
        """What a search checks before it searches, worked out off the event loop: never
        whether a run holds the index, as that tries the run's lock, which a run's own try
        meeting would take for another session's build."""
        return await asyncio.to_thread(self._stale_check, project_root)

    def _stale_check(self, project_root: Path) -> StaleCheck:
        manifest = self._store.manifest()
        files_count = self._store.counts(self._model).files
        files = self._files.current(project_root)
        return StaleCheck(
            project_root=project_root,
            files=files,
            freshness=IndexFreshness(
                files_count=files_count, stale_files_count=self._stale_files_count(files)
            ),
            manifest=manifest,
        )

    def _status(self, project_root: Path) -> IndexStatus:
        """How far the project's index is from its files: how much it holds, how many of its
        chunks have the model's vectors, how many files changed since, and whether a run
        holds it now."""
        counts = self._store.counts(self._model)
        return IndexStatus(
            files_count=counts.files,
            chunks_count=counts.chunks,
            model=self._model,
            chunks_with_vectors_count=counts.chunks_with_vectors,
            stale_files_count=self._stale_files_count(self._files.current(project_root)),
            manifest=self._store.manifest(),
            being_built=self._store.run_held(),
        )

    def _stale_files_count(self, files: FileStamps) -> int:
        """How many of the files changed since the index was written, new or modified, and
        how many it holds that are gone."""
        delta = self._plan(files, force=False)
        return len(delta.to_index) + len(delta.to_delete)

    async def _index(
        self,
        project_root: Path,
        files: FileStamps,
        *,
        force: bool,
        progress: ProgressReporter,
        started: float,
    ) -> IndexReport:
        """Plan a run over the files found, then chunk and write what it has to do; its report
        timed from the run's start."""
        await progress.report(IndexPhase.planning.value, f"Found {len(files)} files, planning...")
        delta = await asyncio.to_thread(self._plan, files, force=force)
        log.debug("index_planned", to_index=len(delta.to_index), to_delete=len(delta.to_delete))

        if force and self._model.model_revision is None:
            # A model replaced under a name nothing pins gives other vectors nothing else
            # sees; a pinned revision vouches for its vectors, an hour's embedding at 88k chunks
            await asyncio.to_thread(self._store.remove_vectors, self._model)
        chunks_indexed = 0
        if delta.has_work:
            await progress.report(
                IndexPhase.indexing.value, f"Chunking {len(delta.to_index)} files..."
            )
            chunks_indexed = await self._chunk_and_write(
                project_root,
                delta,
                progress=progress.within(IndexPhase.indexing.value, PROGRESS_TOTAL),
            )

        return IndexReport(
            files_indexed=len(delta.to_index),
            chunks_indexed=chunks_indexed,
            files_deleted=len(delta.to_delete),
            duration_seconds=round(time.perf_counter() - started, DURATION_DECIMALS),
        )

    def _plan(self, files: FileStamps, *, force: bool) -> IndexDelta:
        """What a run over these files is to do, from what the store holds."""
        return plan_delta(files, self._store.indexed_files(), every_file=force)

    async def _chunk_and_write(
        self, project_root: Path, delta: IndexDelta, *, progress: ProgressReporter
    ) -> int:
        """Chunk the files to index a batch at a time and write them in slices as they come;
        the number of chunks written."""
        writer = RunWriter(
            self._store, delta.to_delete, files_total=len(delta.to_index), progress=progress
        )
        async with self._chunking.batches(project_root, delta.to_index) as batches:
            async for batch in batches:
                await writer.add(batch)
        return await writer.finish()


class RunWriter:
    """Writes one run's chunked files as they come, in their order, in slices of whole files of
    at least WRITE_SLICE_SIZE chunks: the removals with the first write, a report before each,
    the run's end with the last. A slice is written once a file after it has come, so the slice
    left when the files end is the last write, empty where the run chunks nothing."""

    def __init__(
        self,
        store: IndexStore,
        removed_paths: list[Path],
        *,
        files_total: int,
        progress: ProgressReporter,
    ) -> None:
        self._store = store
        self._removals = removed_paths
        self._files_total = files_total
        self._progress = progress
        self._gathered: list[ChunkedFile] = []
        self._gathered_chunks = 0
        self._files_written = 0
        self._chunks_written = 0

    async def add(self, chunked_files: list[ChunkedFile]) -> None:
        """Gather the next files, writing each slice that a file after it fills."""
        for chunked_file in chunked_files:
            if self._gathered_chunks >= WRITE_SLICE_SIZE:
                await self._write(last_of_run=False)
            self._gathered.append(chunked_file)
            self._gathered_chunks += len(chunked_file.chunks)

    async def finish(self) -> int:
        """Write what is gathered as the run's last write; the number of chunks written."""
        await self._write(last_of_run=True)
        return self._chunks_written

    async def _write(self, *, last_of_run: bool) -> None:
        """Write the files gathered as one slice."""
        chunks = [chunk for chunked_file in self._gathered for chunk in chunked_file.chunks]
        await self._progress.report(
            share_done(self._files_written, self._files_total),
            f"Writing {len(chunks)} chunks, {self._files_written} of {self._files_total}"
            " files done...",
        )
        update = IndexUpdate(
            files=[chunked_file.file for chunked_file in self._gathered],
            chunks=chunks,
            removed_paths=self._removals,
            last_of_run=last_of_run,
        )
        await asyncio.to_thread(self._store.write, update)
        self._removals = []
        self._files_written += len(self._gathered)
        self._chunks_written += len(chunks)
        self._gathered = []
        self._gathered_chunks = 0


def plan_delta(on_disk: FileStamps, indexed: IndexedFiles, *, every_file: bool) -> IndexDelta:
    """The files to index — every file, or those new or changed since their record — and
    the paths no longer on disk that the index holds records or chunks of. Compared by their
    paths' text, a file's record made only for the files to index."""
    recorded = indexed.records
    to_index = [
        SourceFile(path=Path(path), stamp=stamp)
        for path, stamp in on_disk.items()
        if every_file or recorded.get(path) != stamp
    ]
    gone = (recorded.keys() | set(indexed.unrecorded_paths)) - on_disk.keys()
    return IndexDelta(to_index=to_index, to_delete=[Path(path) for path in sorted(gone)])
