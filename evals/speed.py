"""Speed: what an agent waits for, and what the server costs the machine.

Measured per corpus, in server processes of its own: a full index from scratch, warm
search latency, how much the index grows with each single-file edit, the server's peak
memory, and the first answer of a freshly started server on the built index — the
process start and model load an agent waits for. Peak memory is read from `/proc`, so
it is reported on Linux only.
"""

import os
import platform
import statistics
import time
from pathlib import Path
from typing import TextIO

from pydantic import BaseModel, ConfigDict

from evals.corpus import Corpus, restore
from evals.quality import RESULTS_PER_QUERY
from evals.server import ServerConnection, connect

MILLISECONDS_PER_SECOND = 1000
BYTES_PER_MEGABYTE = 1024 * 1024
KILOBYTES_PER_MEGABYTE = 1024
# statistics.quantiles with this many cuts gives percentiles; two of them are reported
PERCENTILE_CUTS = 100
MEDIAN_CUT = 49
NINETY_FIFTH_CUT = 94
REPORTED_DECIMALS = 2
# What an edit appends to a file: enough to change its mtime and its content
EDIT_SUFFIX = "\n"
PROC_ROOT = Path("/proc")
PEAK_RESIDENT_FIELD = "VmHWM:"
SERVER_COMMAND_MARK = b"semantic_code_mcp"


class Machine(BaseModel):
    """What a speed was measured on; speeds compare only on the same machine."""

    model_config = ConfigDict(frozen=True)

    processor: str
    logical_cores: int
    python: str


class CorpusSpeed(BaseModel):
    """The speeds of one corpus."""

    model_config = ConfigDict(frozen=True)

    corpus: str
    files_indexed: int
    chunks_indexed: int
    full_index_seconds: float
    files_per_second: float
    chunks_per_second: float
    warm_search_p50_milliseconds: float
    warm_search_p95_milliseconds: float
    index_growth_megabytes_per_edit: float
    peak_memory_megabytes: float | None
    first_search_seconds: float


def current_machine() -> Machine:
    """The machine this process runs on."""
    return Machine(
        processor=platform.processor() or platform.machine(),
        logical_cores=os.cpu_count() or 0,
        python=platform.python_version(),
    )


async def measure_speed(
    corpus: Corpus, working_tree: Path, index_directory: Path, edits: int, server_log: TextIO
) -> CorpusSpeed:
    """Every speed of one corpus, starting from an empty index directory."""
    queries = [query.text for query in corpus.queries]
    edited_file = working_tree / corpus.queries[0].answers[0].path

    async with connect(index_directory, server_log) as connection:
        report = await connection.index(working_tree)
        latencies = [
            await timed_search_seconds(connection, working_tree, query) for query in queries
        ]
        growth_bytes = await index_growth_bytes(
            connection, working_tree, edited_file, queries[0], index_directory, edits
        )
        peak_memory = server_peak_memory_megabytes()

    started = time.perf_counter()
    async with connect(index_directory, server_log) as connection:
        await connection.search(working_tree, queries[0], RESULTS_PER_QUERY)
        first_search_seconds = time.perf_counter() - started

    return CorpusSpeed(
        corpus=corpus.corpus.name,
        files_indexed=report.files_indexed,
        chunks_indexed=report.chunks_indexed,
        full_index_seconds=report.duration_seconds,
        files_per_second=rounded(report.files_indexed / report.duration_seconds),
        chunks_per_second=rounded(report.chunks_indexed / report.duration_seconds),
        warm_search_p50_milliseconds=percentile_milliseconds(latencies, MEDIAN_CUT),
        warm_search_p95_milliseconds=percentile_milliseconds(latencies, NINETY_FIFTH_CUT),
        index_growth_megabytes_per_edit=rounded(growth_bytes / BYTES_PER_MEGABYTE / edits),
        peak_memory_megabytes=peak_memory,
        first_search_seconds=rounded(first_search_seconds),
    )


async def timed_search_seconds(
    connection: ServerConnection, working_tree: Path, query: str
) -> float:
    """How long one search takes, seen from the client."""
    started = time.perf_counter()
    await connection.search(working_tree, query, RESULTS_PER_QUERY)
    return time.perf_counter() - started


async def index_growth_bytes(
    connection: ServerConnection,
    working_tree: Path,
    edited_file: Path,
    query: str,
    index_directory: Path,
    edits: int,
) -> int:
    """How many bytes the index grows over a number of single-file edits.

    Each edit appends to one file and searches, which re-indexes that file. The
    checkout is restored afterwards, whatever happens.
    """
    size_before = directory_bytes(index_directory)
    try:
        for _ in range(edits):
            with edited_file.open("a") as edited:
                edited.write(EDIT_SUFFIX)
            await connection.search(working_tree, query, RESULTS_PER_QUERY)
    finally:
        restore(working_tree)
    return directory_bytes(index_directory) - size_before


def percentile_milliseconds(seconds: list[float], cut: int) -> float:
    """One percentile of a list of durations, in milliseconds."""
    cut_points = statistics.quantiles(seconds, n=PERCENTILE_CUTS)
    return rounded(cut_points[cut] * MILLISECONDS_PER_SECOND)


def directory_bytes(directory: Path) -> int:
    """The total size of the files under a directory."""
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def server_peak_memory_megabytes() -> float | None:
    """Peak resident memory of the server process this harness started; None off Linux."""
    status_paths = [PROC_ROOT / str(pid) / "status" for pid in server_process_ids()]
    peaks = [peak_resident_kilobytes(path) for path in status_paths]
    if not peaks:
        return None
    return rounded(max(peaks) / KILOBYTES_PER_MEGABYTE)


def server_process_ids() -> list[int]:
    """Child processes of this one whose command line runs the server."""
    children_files = (PROC_ROOT / str(os.getpid()) / "task").glob("*/children")
    child_ids = [int(pid) for children in children_files for pid in children.read_text().split()]
    return [
        pid
        for pid in child_ids
        if SERVER_COMMAND_MARK in (PROC_ROOT / str(pid) / "cmdline").read_bytes()
    ]


def peak_resident_kilobytes(status_path: Path) -> int:
    """The peak resident set size a /proc status file reports, in kilobytes."""
    field = next(
        line
        for line in status_path.read_text().splitlines()
        if line.startswith(PEAK_RESIDENT_FIELD)
    )
    return int(field.split()[1])


def rounded(value: float) -> float:
    return round(value, REPORTED_DECIMALS)
