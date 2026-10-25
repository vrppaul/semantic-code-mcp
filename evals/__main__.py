"""Measure search quality or speed on the pinned corpora, and compare with the baseline.

    uv run python -m evals quality [--corpus NAME ...] [--save-baseline]
    uv run python -m evals speed [--corpus NAME ...] [--edits N] [--save-baseline]

Each run indexes into a fresh temporary directory; checkouts of the corpora are kept
under ~/.cache/semantic-code-mcp-evals and reused. Prints one line per corpus next to the
saved baseline, writes the whole report as JSON to `evals/results/`, and with
--save-baseline makes it the new baseline. Exits 0, or 1 when a tool call fails or a
corpus names an answer file its checkout lacks.
"""

import argparse
import asyncio
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from evals.corpus import Corpus, available_corpus_names, checkout, load_corpora
from evals.quality import CorpusMistakeError, CorpusQuality, RankScores, measure_quality
from evals.server import ToolCallFailedError, connect
from evals.speed import CorpusSpeed, Machine, current_machine, measure_speed

CHECKOUTS_ROOT = Path.home() / ".cache" / "semantic-code-mcp-evals"
EVALS_DIRECTORY = Path(__file__).parent
BASELINE_DIRECTORY = EVALS_DIRECTORY / "baseline"
RESULTS_DIRECTORY = EVALS_DIRECTORY / "results"
QUALITY_FILE = "quality.json"
SPEED_FILE = "speed.json"
SERVER_LOG_FILE = "server.log"
DEFAULT_EDITS = 5
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
    RESULTS_DIRECTORY.mkdir(exist_ok=True)
    BASELINE_DIRECTORY.mkdir(exist_ok=True)
    corpora = load_corpora(arguments.corpus)
    try:
        match arguments.command:
            case "quality":
                run_quality(corpora, save_baseline=arguments.save_baseline)
            case "speed":
                run_speed(corpora, edits=arguments.edits, save_baseline=arguments.save_baseline)
    except (ToolCallFailedError, CorpusMistakeError) as error:
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
    speed.add_argument("--edits", type=int, default=DEFAULT_EDITS)
    return parser.parse_args(argv)


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
    for corpus in corpora:
        working_tree = checkout(corpus.corpus, CHECKOUTS_ROOT)
        with (
            tempfile.TemporaryDirectory() as index_directory,
            (RESULTS_DIRECTORY / SERVER_LOG_FILE).open("a") as server_log,
        ):
            async with connect(Path(index_directory), server_log) as connection:
                measured.append(await measure_quality(corpus, working_tree, connection))
    return measured


async def collect_speed(corpora: list[Corpus], edits: int) -> list[CorpusSpeed]:
    measured: list[CorpusSpeed] = []
    for corpus in corpora:
        working_tree = checkout(corpus.corpus, CHECKOUTS_ROOT)
        with (
            tempfile.TemporaryDirectory() as index_directory,
            (RESULTS_DIRECTORY / SERVER_LOG_FILE).open("a") as server_log,
        ):
            measured.append(
                await measure_speed(corpus, working_tree, Path(index_directory), edits, server_log)
            )
    return measured


def describe_quality(entry: CorpusQuality, baseline: CorpusQuality | None) -> str:
    """A line of file and chunk scores, and the baseline's under it."""
    line = f"{entry.corpus}: {quality_summary(entry)}"
    if baseline is None:
        return line
    return f"{line}\n  baseline: {quality_summary(baseline)}"


def quality_summary(entry: CorpusQuality) -> str:
    return (
        f"file {rank_summary(entry.file, entry.queries)} | "
        f"chunk {rank_summary(entry.chunk, entry.queries)}"
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
        f"first search {entry.first_search_seconds}s; peak {entry.peak_memory_megabytes}MB; "
        f"index growth {entry.index_growth_megabytes_per_edit}MB per edit"
    )


def read_report[Report: BaseModel](path: Path, report_type: type[Report]) -> Report | None:
    if not path.is_file():
        return None
    return report_type.model_validate_json(path.read_text())


def write_report(report: BaseModel, file_name: str, *, save_baseline: bool) -> None:
    serialized = report.model_dump_json(indent=JSON_INDENT) + "\n"
    (RESULTS_DIRECTORY / file_name).write_text(serialized)
    if save_baseline:
        (BASELINE_DIRECTORY / file_name).write_text(serialized)


if __name__ == "__main__":
    sys.exit(main())
