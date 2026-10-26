"""Measure search quality or speed on the pinned corpora, and compare with the baseline.

    uv run python -m evals quality [--corpus NAME ...] [--save-baseline]
    uv run python -m evals speed [--corpus NAME ...] [--edits N] [--save-baseline]

Each run indexes into a fresh temporary directory; checkouts of the corpora are kept
under ~/.cache/semantic-code-mcp-evals and reused. Prints one line per corpus next to the
saved baseline, writes the whole report as JSON to `evals/results/`, and with
--save-baseline makes it the new baseline. Exits 0, or 1 when a tool call fails or
times out, a clone fails, or a corpus names an answer its checkout lacks.
"""

import argparse
import asyncio
import subprocess  # nosec B404
import sys
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from mcp.shared.exceptions import MCPError
from pydantic import BaseModel, ConfigDict, ValidationError

from evals.corpus import (
    Corpus,
    available_corpus_names,
    checkout,
    load_corpora,
    load_off_topic_questions,
)
from evals.quality import CorpusMistakeError, CorpusQuality, RankScores, measure_quality
from evals.server import ToolCallFailedError, connect
from evals.speed import CorpusSpeed, Machine, current_machine, measure_speed
from semantic_code_mcp.logging import configure_logging

CHECKOUTS_ROOT = Path.home() / ".cache" / "semantic-code-mcp-evals"
EVALS_DIRECTORY = Path(__file__).parent
BASELINE_DIRECTORY = EVALS_DIRECTORY / "baseline"
RESULTS_DIRECTORY = EVALS_DIRECTORY / "results"
QUALITY_FILE = "quality.json"
SPEED_FILE = "speed.json"
SERVER_LOG_FILE = "server.log"
# Enough single-file edits for the index's upkeep to run, so its growth shows settled
DEFAULT_EDITS = 20
JSON_INDENT = 2
EXIT_OK = 0
EXIT_FAILED = 1


class QualityReport(BaseModel):
    """Search quality over every corpus measured."""

    model_config = ConfigDict(frozen=True)

    corpora: tuple[CorpusQuality, ...]


class SpeedReport(BaseModel):
    """Speeds over every corpus measured, and the machine they were measured on."""

    model_config = ConfigDict(frozen=True)

    machine: Machine
    corpora: tuple[CorpusSpeed, ...]


def main() -> int:
    arguments = parse_arguments(sys.argv[1:])
    # The answer check runs the chunker in this process; its debug lines are noise here
    configure_logging(debug=False)
    RESULTS_DIRECTORY.mkdir(exist_ok=True)
    BASELINE_DIRECTORY.mkdir(exist_ok=True)
    corpora = load_corpora(arguments.corpus)
    try:
        match arguments.command:
            case "quality":
                run_quality(corpora, save_baseline=arguments.save_baseline)
            case "speed":
                run_speed(corpora, edits=arguments.edits, save_baseline=arguments.save_baseline)
    except (
        ToolCallFailedError,
        CorpusMistakeError,
        MCPError,
        subprocess.CalledProcessError,
    ) as error:
        print(error, file=sys.stderr)
        return EXIT_FAILED
    return EXIT_OK


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m evals", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    quality = commands.add_parser("quality", help="search quality: where the answers land")
    speed = commands.add_parser("speed", help="indexing, search, memory and index growth")
    for command in (quality, speed):
        command.add_argument(
            "--corpus", action="append", default=[], choices=available_corpus_names()
        )
        command.add_argument("--save-baseline", action="store_true")
    speed.add_argument("--edits", type=positive_integer, default=DEFAULT_EDITS)
    return parser.parse_args(argv)


def positive_integer(text: str) -> int:
    """An argument that must be a whole number above zero."""
    value = int(text)
    if value < 1:
        message = f"must be at least 1, not {value}"
        raise argparse.ArgumentTypeError(message)
    return value


def run_quality(corpora: list[Corpus], *, save_baseline: bool) -> None:
    report = QualityReport(corpora=tuple(asyncio.run(collect_quality(corpora))))
    baseline = read_report(BASELINE_DIRECTORY / QUALITY_FILE, QualityReport)
    baseline_by_corpus: dict[str, CorpusQuality] = {}
    if baseline is not None:
        baseline_by_corpus = {entry.corpus: entry for entry in baseline.corpora}
    for entry in report.corpora:
        print(describe_quality(entry, baseline_by_corpus.get(entry.corpus)))
    write_report(report, QUALITY_FILE, save_baseline=save_baseline)


def run_speed(corpora: list[Corpus], *, edits: int, save_baseline: bool) -> None:
    report = SpeedReport(
        machine=current_machine(), corpora=tuple(asyncio.run(collect_speed(corpora, edits)))
    )
    baseline = read_report(BASELINE_DIRECTORY / SPEED_FILE, SpeedReport)
    # Speeds compare only on the machine they were measured on
    baseline_by_corpus: dict[str, CorpusSpeed] = {}
    if baseline is not None and baseline.machine == report.machine:
        baseline_by_corpus = {entry.corpus: entry for entry in baseline.corpora}
    for entry in report.corpora:
        print(describe_speed(entry, baseline_by_corpus.get(entry.corpus)))
    write_report(report, SPEED_FILE, save_baseline=save_baseline)


async def collect_quality(corpora: list[Corpus]) -> list[CorpusQuality]:
    measured: list[CorpusQuality] = []
    off_topic = load_off_topic_questions()
    for corpus in corpora:
        working_tree = checkout(corpus.corpus, CHECKOUTS_ROOT)
        with (
            tempfile.TemporaryDirectory() as index_directory,
            (RESULTS_DIRECTORY / SERVER_LOG_FILE).open("a") as server_log,
        ):
            server_log.write(log_header("quality", corpus))
            async with connect(Path(index_directory), server_log) as connection:
                measured.append(await measure_quality(corpus, working_tree, off_topic, connection))
    return measured


async def collect_speed(corpora: list[Corpus], edits: int) -> list[CorpusSpeed]:
    measured: list[CorpusSpeed] = []
    for corpus in corpora:
        working_tree = checkout(corpus.corpus, CHECKOUTS_ROOT)
        with (
            tempfile.TemporaryDirectory() as index_directory,
            (RESULTS_DIRECTORY / SERVER_LOG_FILE).open("a") as server_log,
        ):
            server_log.write(log_header("speed", corpus))
            measured.append(
                await measure_speed(corpus, working_tree, Path(index_directory), edits, server_log)
            )
    return measured


def log_header(command: str, corpus: Corpus) -> str:
    """The line that opens one corpus's server log in the shared log file."""
    return f"\n===== {command} {corpus.corpus.name} {datetime.now(tz=UTC).isoformat()} =====\n"


def describe_quality(entry: CorpusQuality, baseline: CorpusQuality | None) -> str:
    """A line of file and chunk scores, and the baseline's under it."""
    line = f"{entry.corpus}: {quality_summary(entry)}"
    if baseline is None:
        return line
    return f"{line}\n  baseline: {quality_summary(baseline)}"


def quality_summary(entry: CorpusQuality) -> str:
    return (
        f"file {rank_summary(entry.file, entry.queries)} | "
        f"chunk {rank_summary(entry.chunk, entry.queries)} | "
        f"off-topic {entry.off_topic_results} results for {entry.off_topic_questions} questions"
    )


def rank_summary(scores: RankScores, queries: int) -> str:
    return (
        f"hits@1 {scores.hits_at_1}/{queries} hits@3 {scores.hits_at_3}/{queries} "
        f"MRR {scores.mean_reciprocal_rank}"
    )


def describe_speed(entry: CorpusSpeed, baseline: CorpusSpeed | None) -> str:
    """A line of speeds, and the baseline's under it when it was measured on this machine."""
    line = f"{entry.corpus}: {speed_summary(entry)}"
    if baseline is None:
        return line
    return f"{line}\n  baseline: {speed_summary(baseline)}"


def speed_summary(entry: CorpusSpeed) -> str:
    return (
        f"{entry.files_indexed} files, {entry.chunks_indexed} chunks "
        f"in {entry.full_index_seconds}s ({entry.files_per_second} files/s, "
        f"{entry.chunks_per_second} chunks/s); search p50 "
        f"{entry.warm_search_p50_milliseconds}ms p95 {entry.warm_search_p95_milliseconds}ms; "
        f"status {entry.status_p50_milliseconds}ms; "
        f"model load {entry.model_load_seconds}s; first search {entry.first_search_seconds}s; "
        f"peak {entry.peak_memory_megabytes}MB; "
        f"index growth {entry.index_growth_megabytes_per_edit}MB per edit; "
        f"search after an edit {entry.search_after_edit_p50_milliseconds}ms"
    )


def read_report[Report: BaseModel](path: Path, report_type: type[Report]) -> Report | None:
    """A saved report, or None when there is none or it was saved in an older shape."""
    if not path.is_file():
        return None
    try:
        return report_type.model_validate_json(path.read_text())
    except ValidationError:
        print(f"{path} was saved in an older shape; compare after --save-baseline", file=sys.stderr)
        return None


def write_report(report: BaseModel, file_name: str, *, save_baseline: bool) -> None:
    serialized = report.model_dump_json(indent=JSON_INDENT) + "\n"
    (RESULTS_DIRECTORY / file_name).write_text(serialized)
    if save_baseline:
        (BASELINE_DIRECTORY / file_name).write_text(serialized)


if __name__ == "__main__":
    sys.exit(main())
