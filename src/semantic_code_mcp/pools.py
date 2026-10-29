"""Pools of processes made for one run of work beside the server: a first index's chunking
(ADR 010, O4).

Python 3.14 starts a pool's processes from a forkserver on Linux and spawns them on macOS and
Windows, never forking the server and its threads; either way the main module is imported once
more, which the console script's own module keeps light (`__main__.py`). A pool is started as it
is opened, its first process with it, so a system that cannot run one (no working semaphores, no
Unix sockets for the forkserver) says so there.

Each worker, before it runs a task:
- points its file descriptor 1 at its stderr. While it serves, mcp's stdio server (2.3) points
  the server's fd 0 at the null device and fd 1 at stderr and keeps the protocol on a descriptor
  of its own, which children do not inherit, so a worker inherits no part of the session; it
  holds to stderr all the same, whatever the server's fd 1 is, as one stray line on the
  protocol's stream would break the session;
- logs as the server does: structlog's default prints to stdout, a line for every file at the
  debug level;
- ends when the server does. A worker waits on its call queue, whose ends each worker holds too,
  so a server killed mid-run (SIGKILL, the OOM killer, a SIGTERM it does not handle) left its
  workers waiting for good, and the forkserver and the resource tracker beside them; a thread
  of the worker's own waits on the server's sentinel and ends the worker with it, and the
  forkserver and the tracker end once the workers have.

A pool has half the cores this process may use, at most MOST_WORKERS: on Sentry a first index
took 5.3 s at the median of four turns in four workers, 5.5 in three and 6.9 in two (11.7-12.2
in the server alone), each worker holding about 45 MB while it lives; one would gain nothing
over the server's own thread.
"""

import multiprocessing
import os
import threading
from concurrent.futures import Executor, ProcessPoolExecutor
from functools import partial
from multiprocessing.connection import wait

from semantic_code_mcp.logging import configure_logging

# A pool's workers: half the cores at most, and at most four; fewer than two gain nothing
MOST_WORKERS = 4
CORES_PER_WORKER = 2
FEWEST_WORKERS = 2
# The workers where a pool would have fewer than FEWEST_WORKERS: none, every run in the server
NO_WORKERS = 0
# The cores counted where the system cannot say how many this process may use
UNKNOWN_CORES = 1
# A process's standard output and standard error, by their file descriptors
STANDARD_OUTPUT = 1
STANDARD_ERROR = 2
# What a worker ends with when its server has gone
SERVER_GONE_EXIT_CODE = 1
# The name of the thread a worker waits for its server's end in
SERVER_WATCH_THREAD = "server-watch"

# A pool of processes as the services hand it work, which they need not know is one
type WorkerPool = Executor


def open_chunking_pool(workers: int, *, debug_logging: bool) -> WorkerPool:
    """A pool of the workers given, its forkserver and first process started now, each worker
    held to stderr, logging as the server does and ending with it."""
    pool = ProcessPoolExecutor(
        workers, initializer=partial(prepare_worker, debug_logging=debug_logging)
    )
    try:
        # A pool starts its first process at its first task: one that does nothing, now
        pool.submit(os.getpid)
    except OSError:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    return pool


def prepare_worker(*, debug_logging: bool) -> None:
    """What a pool's process does before its first task: its stdout pointed at its stderr,
    logging set as the server's, and a thread that ends it with the server."""
    os.dup2(STANDARD_ERROR, STANDARD_OUTPUT)
    configure_logging(debug=debug_logging)
    server = multiprocessing.parent_process()
    # A pool's process always has its server for parent; this narrows the type alone
    if server is not None:
        threading.Thread(
            target=end_with_the_server,
            args=(server.sentinel,),
            name=SERVER_WATCH_THREAD,
            daemon=True,
        ).start()


def end_with_the_server(server_sentinel: int) -> None:
    """Wait until the server has gone, then end this process at once: its tasks are the
    server's, and nobody is left to take their results."""
    wait([server_sentinel])
    os._exit(SERVER_GONE_EXIT_CODE)


def pool_workers(cores: int | None) -> int:
    """The workers a pool has on a machine whose process may use the cores given, None where
    the system cannot say: half of them, at most MOST_WORKERS, or none where that is fewer
    than FEWEST_WORKERS."""
    workers = min(MOST_WORKERS, (cores or UNKNOWN_CORES) // CORES_PER_WORKER)
    return workers if workers >= FEWEST_WORKERS else NO_WORKERS
