"""Speed: what an agent waits for, and what the server costs the machine.

Measured per corpus, in server processes of its own: the model load (a search of a
one-file project), a full index from scratch with the model already loaded, warm search
latency, an `index_status` call (each search makes one first), how much the index grows
with each edit of a file other than the last (the copies upkeep keeps for a few seconds
included) and how long the search after it takes, at the median and at the slowest (it
re-indexes the edited file first; one in twenty rebuilds the full-text index), the
server's peak memory, and the first answer of a freshly started server on an up-to-date
index — the process start and model load an agent waits for.
Peak memory is read from `/proc`, so it is reported on Linux only. With 15 to 20
questions per corpus, p95 is close to the slowest search: read it as that.
"""

import os
import platform
import statistics
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from pydantic import BaseModel, ConfigDict

from evals.corpus import Corpus, Query, restore
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
# What the numbered edit of a file appends, by its extension: a definition of its own, so
# the edit adds one chunk and changes no other, as a small edit does
EDIT_BY_EXTENSION = {
    ".py": "\n\ndef edited_{number}():\n    return {number}\n",
    ".rs": "\n\nfn edited_{number}() -> u32 {{\n    {number}\n}}\n",
    ".md": "\n\n## Edited {number}\n\nEdit {number}.\n",
}
PROC_ROOT = Path("/proc")
PEAK_RESIDENT_FIELD = "VmHWM:"
SERVER_COMMAND_MARK = b"semantic_code_mcp"
# The one-file project the model-load search runs on, and what it asks
WARM_UP_FILE_NAME = "warm_up.py"
WARM_UP_SOURCE = "def warm_up():\n    return None\n"
WARM_UP_QUERY = "warm up"
# Status calls timed on the indexed corpus; each search makes the same check first
STATUS_CALLS = 10
# Where Linux names the processor, and the field that holds its model
CPU_INFO = Path("/proc/cpuinfo")
CPU_MODEL_FIELD = "model name"
CPU_INFO_SEPARATOR = ":"


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
    status_p50_milliseconds: float
    index_growth_megabytes_per_edit: float
    search_after_edit_p50_milliseconds: float
    search_after_edit_slowest_milliseconds: float
    peak_memory_megabytes: float | None
    model_load_seconds: float
    first_search_seconds: float


def current_machine() -> Machine:
    """The machine this process runs on."""
    return Machine(
        processor=processor_model(),
        logical_cores=os.cpu_count() or 0,
        python=platform.python_version(),
    )


def processor_model() -> str:
    """The processor's model, as Linux names it; elsewhere what `platform` knows.

    `platform.processor()` gives only `x86_64` on Linux, which every such machine shares.
    """
    try:
        lines = CPU_INFO.read_text().splitlines()
    except OSError:
        return platform.processor() or platform.machine()
    models = [
        line.partition(CPU_INFO_SEPARATOR)[2].strip()
        for line in lines
        if line.startswith(CPU_MODEL_FIELD)
    ]
    return models[0] if models else platform.machine()


async def measure_speed(
    corpus: Corpus, working_tree: Path, index_directory: Path, edits: int, server_log: TextIO
) -> CorpusSpeed:
    """Every speed of one corpus, starting from an empty index directory."""
    queries = [query.text for query in corpus.queries]

    async with connect(index_directory, server_log) as connection:
        model_load_seconds = await timed_model_load_seconds(connection)
        report = await connection.index(working_tree)
        latencies = [
            await timed_search_seconds(connection, working_tree, query) for query in queries
        ]
        status_seconds = [
            await timed_status_seconds(connection, working_tree) for _ in range(STATUS_CALLS)
        ]
        edits_cost = await edit_cost(connection, corpus, working_tree, index_directory, edits)
        # Restoring the checkout touched the edited files: index them now, so the fresh
        # server below answers its first search without re-indexing anything
        await connection.index(working_tree)
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
        status_p50_milliseconds=median_milliseconds(status_seconds),
        index_growth_megabytes_per_edit=rounded(
            edits_cost.growth_bytes / BYTES_PER_MEGABYTE / edits
        ),
        search_after_edit_p50_milliseconds=median_milliseconds(edits_cost.search_seconds),
        search_after_edit_slowest_milliseconds=milliseconds(max(edits_cost.search_seconds)),
        peak_memory_megabytes=peak_memory,
        model_load_seconds=rounded(model_load_seconds),
        first_search_seconds=rounded(first_search_seconds),
    )


async def timed_model_load_seconds(connection: ServerConnection) -> float:
    """How long the server takes to load its model: a search of a project of one tiny file.

    The file makes the search embed something, so the model loads; an empty project
    would be answered without it.
    """
    with tempfile.TemporaryDirectory() as warm_up_project:
        (Path(warm_up_project) / WARM_UP_FILE_NAME).write_text(WARM_UP_SOURCE)
        started = time.perf_counter()
        await connection.search(Path(warm_up_project), WARM_UP_QUERY, RESULTS_PER_QUERY)
        return time.perf_counter() - started


async def timed_status_seconds(connection: ServerConnection, working_tree: Path) -> float:
    """How long one `index_status` call takes on an up-to-date index, seen from the client."""
    started = time.perf_counter()
    await connection.status(working_tree)
    return time.perf_counter() - started


async def timed_search_seconds(
    connection: ServerConnection, working_tree: Path, query: str
) -> float:
    """How long one search takes, seen from the client."""
    started = time.perf_counter()
    await connection.search(working_tree, query, RESULTS_PER_QUERY)
    return time.perf_counter() - started


class Edit(BaseModel):
    """One edit of the speed run: the file it appends to, from the checkout's root, and the
    question asked after it, which that file answers."""

    model_config = ConfigDict(frozen=True)

    path: Path
    question: str


class EditCost(BaseModel):
    """What a run of single-file edits cost: how much the index grew, and how long each
    search took that re-indexed the edited file before it answered."""

    model_config = ConfigDict(frozen=True)

    growth_bytes: int
    search_seconds: tuple[float, ...]


async def edit_cost(
    connection: ServerConnection,
    corpus: Corpus,
    working_tree: Path,
    index_directory: Path,
    edits: int,
) -> EditCost:
    """What a number of single-file edits cost the index and the searches after them.

    Each edit appends a definition to a file, then asks a question that file answers, which
    re-indexes the file. The checkout is restored afterwards, whatever happens.
    """
    size_before = directory_bytes(index_directory)
    search_seconds: list[float] = []
    try:
        for number, edit in enumerate(edits_in_turn(corpus.queries, edits)):
            edited_file = working_tree / edit.path
            with edited_file.open("a") as edited:
                edited.write(edit_text(edited_file, number))
            search_seconds.append(
                await timed_search_seconds(connection, working_tree, edit.question)
            )
    finally:
        restore(working_tree)
    return EditCost(
        growth_bytes=directory_bytes(index_directory) - size_before,
        search_seconds=tuple(search_seconds),
    )


def edits_in_turn(queries: Sequence[Query], count: int) -> list[Edit]:
    """`count` edits of the files that answer the questions, one file after another in turn,
    each followed by the first question its file answers. An edit of a file other than the
    last is what makes upkeep compact what the last edit left, which edits of one file never
    show."""
    first_question_of_file: dict[str, str] = {}
    for query in queries:
        first_question_of_file.setdefault(query.answers[0].path, query.text)
    turns = [
        Edit(path=Path(path), question=question)
        for path, question in first_question_of_file.items()
    ]
    return [turns[number % len(turns)] for number in range(count)]


def edit_text(path: Path, number: int) -> str:
    """What the numbered edit of a file appends: a definition in the file's language."""
    return EDIT_BY_EXTENSION[path.suffix].format(number=number)


def median_milliseconds(seconds: Sequence[float]) -> float:
    """The median of durations, in milliseconds; one duration is its own median."""
    return milliseconds(statistics.median(seconds))


def milliseconds(seconds: float) -> float:
    """A duration in milliseconds, rounded as the report gives it."""
    return rounded(seconds * MILLISECONDS_PER_SECOND)


def percentile_milliseconds(seconds: list[float], cut: int) -> float:
    """One percentile of a list of durations, in milliseconds."""
    cut_points = statistics.quantiles(seconds, n=PERCENTILE_CUTS)
    return milliseconds(cut_points[cut])


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
    return [pid for pid in child_ids if runs_the_server(pid)]


def runs_the_server(process_id: int) -> bool:
    """Whether a process runs the server; False for one that has already exited."""
    try:
        return SERVER_COMMAND_MARK in (PROC_ROOT / str(process_id) / "cmdline").read_bytes()
    except OSError:
        return False


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
