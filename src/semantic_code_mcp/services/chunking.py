"""Chunking an index run's files a batch at a time, in their order: in this process, or in a
pool of processes made for a run whose source pays the pool's start (ADR 010, O4).

Tree-sitter's parse holds the GIL, so threads gain nothing (four parsed 4,000 of Sentry's files
in 4.1 s where one took 2.5): only processes cut a first index's chunking. The run's pace is
then its writes', one thread of this process (Sentry's take 3.4 s alone and 5 to 7 s beside a
pool), so a pool needs only the workers that chunk as fast. A pool takes about 0.2 s to start,
so a run of less source than POOL_SOURCE_BYTES is chunked here, in a thread off the event
loop. The pool is made for its run and shut down after it, so no idle worker holds memory
between runs; each holds about 45 MB while it lives. Python 3.14 starts a
pool's processes from a forkserver on Linux and spawns them on macOS and Windows, never forking
the server and its threads; either way the main module is imported once more, which the
console script's own module keeps light (`__main__.py`).

A task is a batch of files and the chunker itself, which pickles as its classes in a few
hundred bytes, so a worker keeps nothing between tasks and imports the chunkers alone. A
worker's file descriptor 1 is the server's, the MCP session's stdout: each worker points it at
stderr and logs as the server does before it runs a task, as one stray line would break the
session.

A run hands out at most BATCHES_IN_FLIGHT_PER_WORKER batches a worker and takes them back in
the files' order, so it holds those and what it has not written yet, never every chunk, and
the workers chunk on while it writes.
"""

import asyncio
import itertools
import os
import time
from collections import deque
from collections.abc import AsyncGenerator
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import structlog
from pydantic import BaseModel, ConfigDict

from semantic_code_mcp.domain.interfaces import Chunker
from semantic_code_mcp.domain.models import Chunk, SourceFile
from semantic_code_mcp.logging import configure_logging
from semantic_code_mcp.services.timing import elapsed_milliseconds

log = structlog.get_logger()

# The files a task chunks: four workers chunked Sentry in 5.0 s in batches of 32 or 128, in
# 5.4 to 6.0 s in batches of 8, 16 or 64
BATCH_FILES = 32
# The least source, in bytes, a run chunks in a pool: of Sentry's first files a pool indexed
# 4 MiB in 1.7-1.8 s where this process took 1.6-2.0, and 8 MiB in 4.7-4.9 s against 5.6-6.6
POOL_SOURCE_BYTES = 8 * 1024 * 1024
# A pool's workers: two, which chunk as fast as the run writes (Sentry indexed in 7.0-7.2 s
# in two, in 6.6-7.1 in four, which held 85 MB more at the peak), where they are at most half
# the cores, the rest left to the user; one would gain nothing over this process
POOL_WORKERS = 2
CORES_PER_WORKER = 2
# The cores counted where the system cannot say how many this process may use
UNKNOWN_CORES = 1
# The batches a run hands out a worker: one it chunks and one waiting, so none idles while
# the run writes
BATCHES_IN_FLIGHT_PER_WORKER = 2
# A process's standard output and standard error, by their file descriptors
STANDARD_OUTPUT = 1
STANDARD_ERROR = 2
# The workers of a run chunked in this process
IN_PROCESS = 0

# The files a task chunks together, in their order
type FileBatch = list[SourceFile]


class ChunkedFile(BaseModel):
    """A file indexed in a run, and the chunks it gave."""

    model_config = ConfigDict(frozen=True)

    file: SourceFile
    chunks: list[Chunk]


class Chunking:
    """Chunks the files of each index run with one chunker, in this process or in a pool of
    the workers given."""

    def __init__(self, chunker: Chunker, *, workers: int, debug_logging: bool) -> None:
        self._chunker = chunker
        self._workers = workers
        self._debug_logging = debug_logging

    def batches(
        self, project_root: Path, files: list[SourceFile]
    ) -> AsyncGenerator[list[ChunkedFile]]:
        """The files' chunks, a batch at a time in the files' order, in a pool where the run's
        source pays its start. Closed before its end, as a cancelled run closes it, it shuts
        its pool down, the batches not begun cancelled."""
        batches = [list(batch) for batch in itertools.batched(files, BATCH_FILES, strict=False)]
        if self._workers != IN_PROCESS and source_bytes(files) >= POOL_SOURCE_BYTES:
            return self._in_pool(project_root, batches)
        return self._in_process(project_root, batches)

    async def _in_process(
        self, project_root: Path, batches: list[FileBatch]
    ) -> AsyncGenerator[list[ChunkedFile]]:
        """Each batch chunked in a thread off the event loop once the run took the one before."""
        started = time.perf_counter()
        for batch in batches:
            yield await asyncio.to_thread(chunk_files, self._chunker, project_root, batch)
        log_chunked(batches, IN_PROCESS, started)

    async def _in_pool(
        self, project_root: Path, batches: list[FileBatch]
    ) -> AsyncGenerator[list[ChunkedFile]]:
        """Each batch chunked in a pool made for the run, a batch handed out as the run takes
        one back, in their order; the pool shut down at the end."""
        started = time.perf_counter()
        pool = ProcessPoolExecutor(
            self._workers,
            initializer=partial(prepare_worker, debug_logging=self._debug_logging),
        )
        try:
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
        finally:
            # Off the event loop: the batches begun end first, milliseconds each
            await asyncio.to_thread(partial(pool.shutdown, cancel_futures=True))
        log_chunked(batches, self._workers, started)

    def _hand_out(
        self, pool: ProcessPoolExecutor, project_root: Path, batch: FileBatch
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


def prepare_worker(*, debug_logging: bool) -> None:
    """Point a pool's process's standard output at its standard error and log as the server
    does, before it runs a task."""
    os.dup2(STANDARD_ERROR, STANDARD_OUTPUT)
    configure_logging(debug=debug_logging)


def pool_workers() -> int:
    """The workers a pool has on this machine: POOL_WORKERS where they are at most half the
    cores this process may use, else none."""
    cores = os.process_cpu_count() or UNKNOWN_CORES
    return POOL_WORKERS if cores >= POOL_WORKERS * CORES_PER_WORKER else IN_PROCESS


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
