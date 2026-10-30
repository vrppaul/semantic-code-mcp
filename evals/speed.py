"""Speed: what an agent waits for, and what the server costs the machine.

Measured per corpus, in server processes of its own, every search by words, the default, but the
model's: a full index from scratch, warm search latency, plain and narrowed to functions and methods
outside every tests directory (a narrowing every corpus takes, which both filters read), and of one
long question, an `index_status` call (each search makes one first), how much the index grows with
each edit of a file other than the last (the copies upkeep keeps for a few seconds included) and how
long the search after it takes, at the median and at the slowest (it re-indexes the edited file
first; one in twenty compacts a LanceDB index), the search after a burst of files where the index
never looks, as a build writes under `target/` (a watcher of a whole tree hears each one), the peak
memory of the server and the processes it starts, a first index's pool among them, and what they
still hold once its work is done, as a session's idle server keeps it, with no model loaded; in a
server of its own set to search by meaning too, the model load, a search of a one-file project, then
the corpus's first search, which embeds every chunk, the warm searches after it, and that server's
memory, left out of a run asked to measure words alone, as that first search takes an hour at
Sentry's size; the first answer of a freshly started server on an up-to-date index — the process
start an agent waits for — and, last, the first answer on a repository never indexed: a server
started on an empty index directory, until its first search answers, the whole index built inside
it. Memory is each process's proportional share of what it maps, summed, read from `/proc` every
MEMORY_SAMPLE_SECONDS in a thread of the harness's own while a server indexes or loads its model,
where a pool lives and the peak is, never while a search is timed, which the reads slowed by up to
30 ms at the median on Sentry; it is reported on Linux only. Pages shared with processes outside the
server's, the harness among them, are split with those, so the figure moves a little with what else
runs; a spike shorter than the reads' interval, as a model's load can be, may pass unseen; and it
reads 5 to 20% below the processes' resident sizes. With 15 to 20 questions per corpus, p95 is close
to the slowest search: read it as that.
"""

import os
import platform
import shutil
import statistics
import tempfile
import threading
import time
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from evals.corpus import Corpus, Query, restore
from evals.quality import RESULTS_PER_QUERY
from evals.server import SearchNarrowing, ServerConnection, ServerStarter
from semantic_code_mcp.domain.models import SearchMode

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
# Where a process's proportional share of the memory it maps is read: its own pages and its
# share of those it maps with others, so a pool's processes, forked from one, are not counted
# once each as their resident sizes would be
MEMORY_ROLLUP_FILE = "smaps_rollup"
PROPORTIONAL_FIELD = "Pss:"
# Where a process's children are listed, a file for each of its threads
CHILDREN_FILES = "task/*/children"
# What a process holds that ended as it was read
NOTHING_HELD = 0
SERVER_COMMAND_MARK = b"semantic_code_mcp"
# How often the memory of a server's processes is read while it works: a pool's processes
# live for the seconds of a first index
MEMORY_SAMPLE_SECONDS = 0.05
# The one-file project the model-load search runs on, and what it asks
WARM_UP_FILE_NAME = "warm_up.py"
WARM_UP_SOURCE = "def warm_up():\n    return None\n"
WARM_UP_QUERY = "warm up"
# A question as long as an agent writes when it pastes a sentence or a list of terms: this
# many words of the corpus's questions run together, each one more term the full-text index
# scores every chunk by; timed this many times
LONG_QUESTION_WORDS = 40
LONG_QUESTION_SEARCHES = 5
# Status calls timed on the indexed corpus; each search makes the same check first
STATUS_CALLS = 10
# Where a burst of files is written, as a build writes them: under a directory every default
# ignores, so no burst file is indexed; a directory of its own there, removed after; so many
# files a directory, and a name with no extension a chunker reads, as an object file's
BURST_DIRECTORY = Path("target") / "semantic-code-mcp-burst"
BURST_FILES_PER_DIRECTORY = 100
BURST_FILE = "unit_{number:05}.o"
BURST_CONTENT = b"\0"
# A narrowing every corpus can take, which both filters read: two kinds, and a glob
# left out, the regex a search tests every path against
NARROWING = SearchNarrowing(chunk_kinds=("function", "method"), exclude_glob="tests")
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


class SpeedRun(BaseModel):
    """How a run measures each corpus: the edits it makes, the files of the burst it writes,
    and whether a server set to search by meaning is measured too."""

    model_config = ConfigDict(frozen=True)

    edits: int
    burst_files: int
    by_meaning: bool


class MeaningSpeed(BaseModel):
    """The speeds of a server set to search by meaning too, and its memory."""

    model_config = ConfigDict(frozen=True)

    model_load_seconds: float
    # The first search by meaning of the corpus, which embeds every chunk inside the call
    first_search_seconds: float
    warm_search_p50_milliseconds: float
    peak_memory_megabytes: float | None
    resident_memory_megabytes: float | None


class ServerMemory:
    """The memory of the servers this harness started and of every process they started in
    turn, a first index's pool among them: each one's proportional share, summed. Its peak is
    the highest sum read, by `sample` or every MEMORY_SAMPLE_SECONDS by `sampled_while`; None
    where no server runs, as off Linux, which has no /proc."""

    def __init__(self) -> None:
        self._peak_kilobytes: int | None = None
        # The watching thread and the harness's own both read, and keep the peak
        self._keeping_peak = threading.Lock()

    def sample(self) -> int | None:
        """What the servers' processes hold now, in kilobytes, kept as the peak where it is
        the highest yet."""
        held = server_tree_kilobytes()
        with self._keeping_peak:
            if held is not None and (self._peak_kilobytes is None or held > self._peak_kilobytes):
                self._peak_kilobytes = held
        return held

    def peak_megabytes(self) -> float | None:
        """The most the servers' processes held at once, of what was read."""
        self.sample()
        return None if self._peak_kilobytes is None else megabytes_of(self._peak_kilobytes)

    def resident_megabytes(self) -> float | None:
        """What the servers' processes hold now."""
        held = self.sample()
        return None if held is None else megabytes_of(held)

    def watch(self, stop: threading.Event) -> None:
        """Sample every MEMORY_SAMPLE_SECONDS until told to stop."""
        while not stop.wait(MEMORY_SAMPLE_SECONDS):
            self.sample()


@asynccontextmanager
async def sampled_while(memory: ServerMemory) -> AsyncGenerator[None]:
    """The servers' memory sampled by a thread of its own while the block runs."""
    stop = threading.Event()
    watcher = threading.Thread(target=memory.watch, args=(stop,), daemon=True)
    watcher.start()
    try:
        yield
    finally:
        stop.set()
        watcher.join()


class CorpusSpeed(BaseModel):
    """The speeds of one corpus."""

    model_config = ConfigDict(frozen=True)

    corpus: str
    cold_first_answer_seconds: float
    files_indexed: int
    chunks_indexed: int
    full_index_seconds: float
    files_per_second: float
    chunks_per_second: float
    warm_search_p50_milliseconds: float
    warm_search_p95_milliseconds: float
    narrowed_search_p50_milliseconds: float
    long_question_p50_milliseconds: float
    status_p50_milliseconds: float
    index_growth_megabytes_per_edit: float
    search_after_edit_p50_milliseconds: float
    search_after_edit_slowest_milliseconds: float
    search_after_burst_milliseconds: float
    peak_memory_megabytes: float | None
    resident_memory_megabytes: float | None
    # None for a run that measures words alone
    meaning: MeaningSpeed | None
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
    corpus: Corpus,
    working_tree: Path,
    index_directory: Path,
    run: SpeedRun,
    servers: ServerStarter,
) -> CorpusSpeed:
    """Every speed of one corpus, starting from an empty index directory."""
    queries = [query.text for query in corpus.queries]

    memory = ServerMemory()
    async with servers.connect(index_directory) as connection:
        async with sampled_while(memory):
            report = await connection.index(working_tree)
        latencies = [
            await timed_search_seconds(connection, working_tree, query, None) for query in queries
        ]
        narrowed_latencies = [
            await timed_search_seconds(connection, working_tree, query, NARROWING)
            for query in queries
        ]
        long_question_seconds = [
            await timed_search_seconds(connection, working_tree, long_question(queries), None)
            for _ in range(LONG_QUESTION_SEARCHES)
        ]
        status_seconds = [
            await timed_status_seconds(connection, working_tree) for _ in range(STATUS_CALLS)
        ]
        edits_cost = await edit_cost(connection, corpus, working_tree, index_directory, run.edits)
        # Restoring the checkout touched the edited files: index them now, so the fresh
        # server below answers its first search without re-indexing anything
        await connection.index(working_tree)
        peak_memory = memory.peak_megabytes()
        resident_memory = memory.resident_megabytes()

    meaning = (
        await measure_meaning(working_tree, index_directory, queries, servers)
        if run.by_meaning
        else None
    )

    started = time.perf_counter()
    async with servers.connect(index_directory) as connection:
        await connection.search(working_tree, queries[0], RESULTS_PER_QUERY)
        first_search_seconds = time.perf_counter() - started

    # Last: a cold index keeps a core busy for seconds, 15 at Sentry's size, and the machine
    # it leaves warm would slow what is measured after it
    cold_first_answer_seconds = await timed_cold_first_answer_seconds(
        working_tree, queries[0], servers
    )
    # After the cold first answer too: a watcher of a whole tree hears the burst's every file
    # made and removed, and a system's scanner of new files reads them for a while
    search_after_burst_seconds = await timed_search_after_burst(
        working_tree, index_directory, queries[0], run.burst_files, servers
    )

    return CorpusSpeed(
        corpus=corpus.corpus.name,
        cold_first_answer_seconds=rounded(cold_first_answer_seconds),
        files_indexed=report.files_indexed,
        chunks_indexed=report.chunks_indexed,
        full_index_seconds=report.duration_seconds,
        files_per_second=rounded(report.files_indexed / report.duration_seconds),
        chunks_per_second=rounded(report.chunks_indexed / report.duration_seconds),
        warm_search_p50_milliseconds=percentile_milliseconds(latencies, MEDIAN_CUT),
        warm_search_p95_milliseconds=percentile_milliseconds(latencies, NINETY_FIFTH_CUT),
        narrowed_search_p50_milliseconds=percentile_milliseconds(narrowed_latencies, MEDIAN_CUT),
        long_question_p50_milliseconds=median_milliseconds(long_question_seconds),
        status_p50_milliseconds=median_milliseconds(status_seconds),
        index_growth_megabytes_per_edit=rounded(
            edits_cost.growth_bytes / BYTES_PER_MEGABYTE / run.edits
        ),
        search_after_edit_p50_milliseconds=median_milliseconds(edits_cost.search_seconds),
        search_after_edit_slowest_milliseconds=milliseconds(max(edits_cost.search_seconds)),
        search_after_burst_milliseconds=milliseconds(search_after_burst_seconds),
        peak_memory_megabytes=peak_memory,
        resident_memory_megabytes=resident_memory,
        meaning=meaning,
        first_search_seconds=rounded(first_search_seconds),
    )


async def measure_meaning(
    working_tree: Path, index_directory: Path, queries: Sequence[str], servers: ServerStarter
) -> MeaningSpeed:
    """A server set to search by meaning too, on the corpus's index of words: its model's
    load, its first search, which embeds every chunk, its warm searches, and its memory."""
    memory = ServerMemory()
    async with servers.searching(SearchMode.words_and_meaning).connect(
        index_directory
    ) as connection:
        async with sampled_while(memory):
            model_load_seconds = await timed_model_load_seconds(connection)
            first_search_seconds = await timed_search_seconds(
                connection, working_tree, queries[0], None
            )
        latencies = [
            await timed_search_seconds(connection, working_tree, query, None) for query in queries
        ]
        return MeaningSpeed(
            model_load_seconds=rounded(model_load_seconds),
            first_search_seconds=rounded(first_search_seconds),
            warm_search_p50_milliseconds=percentile_milliseconds(latencies, MEDIAN_CUT),
            peak_memory_megabytes=memory.peak_megabytes(),
            resident_memory_megabytes=memory.resident_megabytes(),
        )


async def timed_cold_first_answer_seconds(
    working_tree: Path, query: str, servers: ServerStarter
) -> float:
    """What an agent waits for on first contact with a repository: a server started fresh on
    an index directory of its own, empty, until its first search answers."""
    with tempfile.TemporaryDirectory() as empty_index_directory:
        started = time.perf_counter()
        async with servers.connect(Path(empty_index_directory)) as connection:
            await connection.search(working_tree, query, RESULTS_PER_QUERY)
            return time.perf_counter() - started


async def timed_model_load_seconds(connection: ServerConnection) -> float:
    """How long a server set to search by meaning too takes to load its model: a search of a
    project of one tiny file.

    The file makes the search embed something, so the model loads; an empty project
    would be answered without it, and a search by words never loads it.
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
    connection: ServerConnection,
    working_tree: Path,
    query: str,
    narrowing: SearchNarrowing | None,
) -> float:
    """How long one search takes, seen from the client, narrowed where a narrowing is given."""
    started = time.perf_counter()
    await connection.search(working_tree, query, RESULTS_PER_QUERY, narrowing)
    return time.perf_counter() - started


async def timed_search_after_burst(
    working_tree: Path, index_directory: Path, query: str, files: int, servers: ServerStarter
) -> float:
    """How long a search takes right after a burst of files is written where the index never
    looks, in a server of its own on the index as it is, its first search made before; what the
    burst made is removed after, the directory it made it in too, whatever came of the search."""
    burst = working_tree / BURST_DIRECTORY
    # What to remove after: the burst's own directory, and `target/` where the burst made it
    made = burst if burst.parent.exists() else burst.parent
    async with servers.connect(index_directory) as connection:
        await connection.search(working_tree, query, RESULTS_PER_QUERY)
        try:
            for number in range(files):
                directory = burst / str(number // BURST_FILES_PER_DIRECTORY)
                directory.mkdir(parents=True, exist_ok=True)
                (directory / BURST_FILE.format(number=number)).write_bytes(BURST_CONTENT)
            return await timed_search_seconds(connection, working_tree, query, None)
        finally:
            # A burst cut short, or a file a system's scanner of new files still holds, never
            # hides the search's own failure
            shutil.rmtree(made, ignore_errors=True)


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
                await timed_search_seconds(connection, working_tree, edit.question, None)
            )
    finally:
        restore(working_tree)
    return EditCost(
        growth_bytes=directory_bytes(index_directory) - size_before,
        search_seconds=tuple(search_seconds),
    )


def edits_in_turn(queries: Sequence[Query], count: int) -> list[Edit]:
    """`count` edits of the files that answer the questions, one file after another in turn,
    each followed by the first question its file answers: edits of one file would replace
    and fold the same rows over and over, where an agent's edits spread over files."""
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


def long_question(questions: Sequence[str]) -> str:
    """The first LONG_QUESTION_WORDS words of the questions, run together into one."""
    return " ".join(" ".join(questions).split()[:LONG_QUESTION_WORDS])


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


def server_tree_kilobytes() -> int | None:
    """The proportional share of the memory the servers this harness started and every process
    they started in turn hold, summed, in kilobytes; None where no server runs."""
    servers = server_process_ids()
    if not servers:
        return None
    return sum(
        proportional_kilobytes(process_id)
        for server in servers
        for process_id in [server, *descendant_ids(server)]
    )


def server_process_ids() -> list[int]:
    """Child processes of this one whose command line runs the server."""
    return [pid for pid in child_ids(os.getpid()) if runs_the_server(pid)]


def descendant_ids(process_id: int) -> list[int]:
    """The processes a process started, and those they started in turn, that still run."""
    children = child_ids(process_id)
    return [*children, *(descendant for child in children for descendant in descendant_ids(child))]


def child_ids(process_id: int) -> list[int]:
    """The processes a process started that still run; none for one that ended as it was
    read."""
    try:
        return [
            int(child)
            for children in (PROC_ROOT / str(process_id)).glob(CHILDREN_FILES)
            for child in children.read_text().split()
        ]
    except OSError:
        return []


def runs_the_server(process_id: int) -> bool:
    """Whether a process runs the server; False for one that has already exited."""
    try:
        return SERVER_COMMAND_MARK in (PROC_ROOT / str(process_id) / "cmdline").read_bytes()
    except OSError:
        return False


def proportional_kilobytes(process_id: int) -> int:
    """A process's proportional share of the memory it maps, in kilobytes; nothing for one that
    ended as it was read."""
    try:
        rollup = (PROC_ROOT / str(process_id) / MEMORY_ROLLUP_FILE).read_text()
    except OSError:
        return NOTHING_HELD
    line = next(line for line in rollup.splitlines() if line.startswith(PROPORTIONAL_FIELD))
    return int(line.split()[1])


def megabytes_of(kilobytes: int) -> float:
    return rounded(kilobytes / KILOBYTES_PER_MEGABYTE)


def rounded(value: float) -> float:
    return round(value, REPORTED_DECIMALS)
