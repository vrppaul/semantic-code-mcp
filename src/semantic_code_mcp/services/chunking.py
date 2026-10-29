"""Chunking an index run's files a batch at a time, in their order: in the server, or in a pool
of processes made for a run whose source pays the pool's start (ADR 010, O4).

Tree-sitter's parse holds the GIL, so threads gain nothing (four parsed 4,000 of Sentry's files
in 4.1 s where one took 2.5): only processes cut a first index's chunking; a pool and its
processes' lives are `pools.py`'s. The run's pace is then its writes', one thread of the server
(Sentry's take 3.3 s alone and 4 to 7 s beside a pool, which takes cores and the processor's
clock). A pool takes about 0.2 s to start, so a run of less source than POOL_SOURCE_BYTES is
chunked in the server, in a thread off the event loop; so is every run of a session whose
system could not start a pool, which the log says once.

A task is a batch of files and the chunker itself, which pickles as its classes in a few
hundred bytes, so a worker keeps nothing between tasks and imports the chunkers alone. A run
hands out at most BATCHES_IN_FLIGHT_PER_WORKER batches a worker ahead and takes them back in the
files' order, so it holds those and what it has not written yet, never every chunk, and the
workers chunk on while it writes. The pool lives for the run's block and is shut down as it
ends, the batches not begun cancelled and those begun finished, a cancelled run's too.
"""

import asyncio
import itertools
import time
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

import structlog
from pydantic import BaseModel, ConfigDict

from semantic_code_mcp.domain.interfaces import Chunker
from semantic_code_mcp.domain.models import Chunk, SourceFile
from semantic_code_mcp.pools import WorkerPool
from semantic_code_mcp.services.timing import elapsed_milliseconds

log = structlog.get_logger()

# The files a task chunks: four workers chunked Sentry in 5.0 s in batches of 32 or 128, in
# 5.4 to 6.0 s in batches of 8, 16 or 64
BATCH_FILES = 32
# The least source, in bytes, a run chunks in a pool: of Sentry's first files a pool of two
# indexed 4 MiB in 1.7-1.8 s where the server took 1.6-2.0, and 8 MiB in 4.7-4.9 s against
# 5.6-6.6
POOL_SOURCE_BYTES = 8 * 1024 * 1024
# The batches a run hands out a worker ahead: one it chunks and one waiting, so none idles
# while the run writes
BATCHES_IN_FLIGHT_PER_WORKER = 2
# The workers of a run chunked in the server
IN_PROCESS = 0
# What opening a pool raises where the system cannot run one: no working semaphores, no Unix
# sockets for its forkserver
POOL_START_ERRORS = (OSError, NotImplementedError)

# The files a task chunks together, in their order
type FileBatch = list[SourceFile]
# Opens a pool of the workers given for one run; the registry says how
type PoolOpener = Callable[[int], WorkerPool]


class ChunkedFile(BaseModel):
    """A file indexed in a run, and the chunks it gave."""

    model_config = ConfigDict(frozen=True)

    file: SourceFile
    chunks: list[Chunk]


class Chunking:
    """Chunks the files of each index run with one chunker, in the server or in a pool of the
    workers given, opened for the run."""

    def __init__(self, chunker: Chunker, *, workers: int, open_pool: PoolOpener) -> None:
        self._chunker = chunker
        self._workers = workers
        self._open_pool = open_pool
        # Whether the system failed to start a pool in this session: it is not asked again
        self._pool_refused = False

    @asynccontextmanager
    async def batches(
        self, project_root: Path, files: list[SourceFile]
    ) -> AsyncGenerator[AsyncIterator[list[ChunkedFile]]]:
        """The files' chunks, a batch at a time in the files' order: in a pool where the run's
        source pays its start and the system can run one, else in the server."""
        batches = [list(batch) for batch in itertools.batched(files, BATCH_FILES, strict=False)]
        pool = await self._opened_pool() if self._pays_a_pool(files) else None
        if pool is None:
            yield self._in_process(project_root, batches)
            return
        try:
            yield self._in_pool(pool, project_root, batches)
        finally:
            # Off the event loop: the batches begun end first, milliseconds each
            await asyncio.to_thread(partial(pool.shutdown, cancel_futures=True))

    def _pays_a_pool(self, files: list[SourceFile]) -> bool:
        """Whether a run of these files is chunked in a pool."""
        return (
            self._workers != IN_PROCESS
            and not self._pool_refused
            and source_bytes(files) >= POOL_SOURCE_BYTES
        )

    async def _opened_pool(self) -> WorkerPool | None:
        """A pool for the run, opened off the event loop; None where the system cannot start
        one, which this session then never asks of it again."""
        try:
            return await asyncio.to_thread(self._open_pool, self._workers)
        except POOL_START_ERRORS as error:
            log.warning("chunking_pool_refused", workers=self._workers, error=str(error))
            self._pool_refused = True
            return None

    async def _in_process(
        self, project_root: Path, batches: list[FileBatch]
    ) -> AsyncGenerator[list[ChunkedFile]]:
        """Each batch chunked in a thread off the event loop once the run took the one before."""
        started = time.perf_counter()
        for batch in batches:
            yield await asyncio.to_thread(chunk_files, self._chunker, project_root, batch)
        log_chunked(batches, IN_PROCESS, started)

    async def _in_pool(
        self, pool: WorkerPool, project_root: Path, batches: list[FileBatch]
    ) -> AsyncGenerator[list[ChunkedFile]]:
        """Each batch chunked in the pool, a batch handed out as the run takes one back, in
        their order."""
        started = time.perf_counter()
        waiting = iter(batches)
        handed_out = deque(
            self._hand_out(pool, project_root, batch)
            for batch in itertools.islice(waiting, self._workers * BATCHES_IN_FLIGHT_PER_WORKER)
        )
        while handed_out:
            chunked = await handed_out.popleft()
            following = next(waiting, None)
            if following is not None:
                handed_out.append(self._hand_out(pool, project_root, following))
            yield chunked
        log_chunked(batches, self._workers, started)

    def _hand_out(
        self, pool: WorkerPool, project_root: Path, batch: FileBatch
    ) -> asyncio.Future[list[ChunkedFile]]:
        """A batch handed to the pool, the chunker going with it."""
        return asyncio.get_running_loop().run_in_executor(
            pool, chunk_files, self._chunker, project_root, batch
        )


def chunk_files(chunker: Chunker, project_root: Path, files: list[SourceFile]) -> list[ChunkedFile]:
    """The chunks of each file, in their order."""
    return [
        ChunkedFile(file=file, chunks=chunker.chunk_file(project_root, file.path)) for file in files
    ]


def source_bytes(files: list[SourceFile]) -> int:
    """The size of the files' source as their scan found it."""
    return sum(file.stamp.size_bytes for file in files)


def log_chunked(batches: list[FileBatch], workers: int, started: float) -> None:
    """Log a run's chunking whole, from its first batch to its last, writes between included."""
    log.debug(
        "files_chunked",
        files=sum(len(batch) for batch in batches),
        workers=workers,
        duration_ms=elapsed_milliseconds(started),
    )
