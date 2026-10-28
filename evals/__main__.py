"""Measure search quality, speed, or search as agents use it, on the pinned corpora.

    uv run python -m evals quality [--corpus NAME ...] [--save-baseline] [--held-out]
    uv run python -m evals speed [--corpus NAME ...] [--corpus-file PATH ...] [--edits N]
        [--save-baseline]
    uv run python -m evals agents [--corpus NAME ...] [--corpus-file PATH ...] [--arm ARM ...]
        [--repeats N] [--model MODEL]

Each run indexes into a fresh temporary directory; checkouts of the corpora are kept
under ~/.cache/semantic-code-mcp-evals and reused. Prints one line per corpus next to the
saved baseline, writes the whole report as JSON to `evals/results/`, and with
--save-baseline makes it the new baseline; a corpus kept outside the repository
(`--corpus-file`, as a large or a private one is) is never saved to it. `agents` asks
Claude Code headless each question in each arm (`evals/agents.py`), spending the owner's
subscription; it prints each session as it ends and each arm's sums, and keeps no baseline,
as its sessions hold the questions of private corpora; its indexes are kept between runs.

Exits 0, or 1 when a tool call fails or times out, a clone fails, a corpus names an answer its
checkout lacks, or Claude Code does not start.
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

from evals.agents import (
    DEFAULT_MODEL,
    AgentArm,
    AgentSessionError,
    AgentsRun,
    ArmSummary,
    CorpusAgents,
    SessionRecord,
    measure_agents,
)
from evals.corpus import (
    COMMIT_PREFIX_LENGTH,
    Corpus,
    available_corpus_names,
    checkout,
    load_corpora,
    load_corpus_file,
    load_off_topic_questions,
)
from evals.quality import (
    CorpusMistakeError,
    CorpusQuality,
    KindScores,
    RankScores,
    WordingScores,
    measure_quality,
)
from evals.server import ToolCallFailedError, connect
from evals.speed import CorpusSpeed, Machine, current_machine, measure_speed
from semantic_code_mcp.logging import configure_logging

CHECKOUTS_ROOT = Path.home() / ".cache" / "semantic-code-mcp-evals"
EVALS_DIRECTORY = Path(__file__).parent
BASELINE_DIRECTORY = EVALS_DIRECTORY / "baseline"
RESULTS_DIRECTORY = EVALS_DIRECTORY / "results"
QUALITY_FILE = "quality.json"
SPEED_FILE = "speed.json"
AGENTS_FILE = "agents.json"
# Where `agents` keeps each corpus's index between runs, as one of a large corpus takes an
# hour, and its own config directory and MCP configs
INDEXES_ROOT = CHECKOUTS_ROOT / "indexes"
AGENTS_RUN_DIRECTORY = CHECKOUTS_ROOT / "agents"
TOKENS_PER_THOUSAND = 1000
# How much of a question a session's line shows
QUESTION_SHOWN_LENGTH = 60
SERVER_LOG_FILE = "server.log"
# Edits enough for one to pay a store's periodic upkeep, which the slowest search after an
# edit shows: LanceDB compacts and rebuilds its full-text index at its first upkeep, the full
# index's, and once in twenty after, so at the last edit's
DEFAULT_EDITS = 20
JSON_INDENT = 2
EXIT_OK = 0
EXIT_FAILED = 1


class QualityReport(BaseModel):
    """Search quality over every corpus measured."""

    model_config = ConfigDict(frozen=True)

    corpora: tuple[CorpusQuality, ...]


class AgentsReport(BaseModel):
    """Every agent session over every corpus asked, and each arm's sums."""

    model_config = ConfigDict(frozen=True)

    corpora: tuple[CorpusAgents, ...]


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
    try:
        match arguments.command:
            case "quality":
                run_quality(
                    load_corpora(arguments.corpus),
                    save_baseline=arguments.save_baseline,
                    held_out=arguments.held_out,
                )
            case "speed":
                run_speed(
                    chosen_corpora(arguments.corpus, arguments.corpus_file),
                    edits=arguments.edits,
                    save_baseline=arguments.save_baseline,
                )
            case "agents":
                run_agents(
                    chosen_corpora(arguments.corpus, arguments.corpus_file),
                    arms=tuple(AgentArm(arm) for arm in arguments.arm or list(AgentArm)),
                    repeats=arguments.repeats,
                    model=arguments.model,
                )
    except (
        ToolCallFailedError,
        CorpusMistakeError,
        MCPError,
        AgentSessionError,
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
    quality.add_argument("--save-baseline", action="store_true")
    # A corpus kept outside the repository, never saved to the baseline the repository keeps
    speed_corpora = speed.add_mutually_exclusive_group()
    speed_corpora.add_argument("--corpus-file", action="append", default=[], type=Path)
    speed_corpora.add_argument("--save-baseline", action="store_true")
    # The held-out questions are asked on every run but shown only once a choice is made
    quality.add_argument("--held-out", action="store_true")
    speed.add_argument("--edits", type=positive_integer, default=DEFAULT_EDITS)
    agents = commands.add_parser("agents", help="search judged by agents: Claude Code headless")
    agents.add_argument("--corpus", action="append", default=[], choices=available_corpus_names())
    # A corpus kept outside the repository, as a private one's questions are
    agents.add_argument("--corpus-file", action="append", default=[], type=Path)
    agents.add_argument("--arm", action="append", default=[], choices=list(AgentArm))
    agents.add_argument("--repeats", type=positive_integer, default=1)
    agents.add_argument("--model", default=DEFAULT_MODEL)
    return parser.parse_args(argv)


def positive_integer(text: str) -> int:
    """An argument that must be a whole number above zero."""
    value = int(text)
    if value < 1:
        message = f"must be at least 1, not {value}"
        raise argparse.ArgumentTypeError(message)
    return value


def run_quality(corpora: list[Corpus], *, save_baseline: bool, held_out: bool) -> None:
    report = QualityReport(corpora=tuple(asyncio.run(collect_quality(corpora))))
    baseline = read_report(BASELINE_DIRECTORY / QUALITY_FILE, QualityReport)
    baseline_by_corpus: dict[str, CorpusQuality] = {}
    if baseline is not None:
        baseline_by_corpus = {entry.corpus: entry for entry in baseline.corpora}
    for entry in report.corpora:
        print(describe_quality(entry, baseline_by_corpus.get(entry.corpus), held_out=held_out))
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


def chosen_corpora(names: list[str], files: list[Path]) -> list[Corpus]:
    """The corpora a run measures: those named and those read from files; every corpus of
    `evals/corpora/` when neither is given."""
    named = load_corpora(names) if names or not files else []
    return [*named, *(load_corpus_file(path) for path in files)]


def run_agents(
    corpora: list[Corpus], *, arms: tuple[AgentArm, ...], repeats: int, model: str
) -> None:
    report = AgentsReport(corpora=tuple(asyncio.run(collect_agents(corpora, arms, repeats, model))))
    for entry in report.corpora:
        print(f"{entry.corpus} on {entry.model}:")
        for summary in entry.arms:
            print(describe_arm(summary))
    write_report(report, AGENTS_FILE, save_baseline=False)


async def collect_agents(
    corpora: list[Corpus], arms: tuple[AgentArm, ...], repeats: int, model: str
) -> list[CorpusAgents]:
    measured: list[CorpusAgents] = []
    for corpus in corpora:
        working_tree = checkout(corpus.corpus, CHECKOUTS_ROOT)
        source = corpus.corpus
        run = AgentsRun(
            arms=arms,
            repeats=repeats,
            model=model,
            index_directory=INDEXES_ROOT / f"{source.name}-{source.commit[:COMMIT_PREFIX_LENGTH]}",
            run_directory=AGENTS_RUN_DIRECTORY,
        )
        with (RESULTS_DIRECTORY / SERVER_LOG_FILE).open("a") as server_log:
            server_log.write(log_header("agents", corpus))
            measured.append(
                await measure_agents(corpus, working_tree, run, server_log, print_session)
            )
    return measured


def print_session(session: SessionRecord) -> None:
    """A line of one session, printed as it ends, as a run of many takes a while."""
    print(describe_session(session), flush=True)


def describe_session(session: SessionRecord) -> str:
    answer = session.answer
    answered = "no answer" if answer is None else f"{answer.path}:{answer.symbol}"
    verdict = "right" if session.answer_right else "file only" if session.file_right else "wrong"
    read = session.usage.cache_read_tokens // TOKENS_PER_THOUSAND
    return (
        f"  {session.arm} {verdict}: {session.turns} turns, {read}k read, "
        f"tools {session.tool_calls}, {session.duration_seconds}s; "
        f"{session.question[:QUESTION_SHOWN_LENGTH]} -> {answered}"
    )


def describe_arm(summary: ArmSummary) -> str:
    """A line of one arm's sums over its sessions."""
    return (
        f"  {summary.arm}: right {summary.right_answers}/{summary.sessions} "
        f"(file {summary.right_files}), turns {summary.mean_turns}, "
        f"context read {summary.mean_context_read_tokens // TOKENS_PER_THOUSAND}k, "
        f"output {summary.mean_output_tokens}, searched {summary.sessions_that_searched}, "
        f"reads {summary.read_calls}, {summary.mean_duration_seconds}s, "
        f"cost ${summary.cost_usd} (within this run)"
    )


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


def describe_quality(
    entry: CorpusQuality, baseline: CorpusQuality | None, *, held_out: bool
) -> str:
    """A line of file and chunk scores, and the baseline's under it."""
    line = f"{entry.corpus}: {quality_summary(entry, held_out=held_out)}"
    if baseline is None:
        return line
    return f"{line}\n  baseline: {quality_summary(baseline, held_out=held_out)}"


def quality_summary(entry: CorpusQuality, *, held_out: bool) -> str:
    """The scores of each kind of question and of its wordings, the names, the off-topic
    questions and, when asked for, the questions held out."""
    kinds = " | ".join(kind_summary(scores) for scores in entry.kinds)
    wordings = ", ".join(wording_summary(scores) for scores in entry.wordings)
    off_topic = entry.off_topic
    summary = (
        f"{kinds}\n    by wording: {wordings}\n    names {entry.names.names}: "
        f"chunk {rank_summary(entry.names.chunk, entry.names.names)} | "
        f"off-topic {off_topic.far_results} results for {off_topic.far_questions} far, "
        f"{off_topic.near_results} for {off_topic.near_questions} near questions"
    )
    if not held_out:
        return summary
    held = " | ".join(kind_summary(scores) for scores in entry.held_out)
    return f"{summary}\n    held out: {held}"


def wording_summary(scores: WordingScores) -> str:
    holding = "holding an answer's words" if scores.shares_answer_words else "without them"
    return (
        f"{scores.kind} {holding} {scores.queries}: chunk MRR {scores.chunk.mean_reciprocal_rank}"
    )


def kind_summary(scores: KindScores) -> str:
    return (
        f"{scores.kind} {scores.queries}: file {rank_summary(scores.file, scores.queries)}, "
        f"chunk {rank_summary(scores.chunk, scores.queries)}"
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
        f"cold first answer {entry.cold_first_answer_seconds}s; "
        f"{entry.files_indexed} files, {entry.chunks_indexed} chunks "
        f"in {entry.full_index_seconds}s ({entry.files_per_second} files/s, "
        f"{entry.chunks_per_second} chunks/s); search p50 "
        f"{entry.warm_search_p50_milliseconds}ms p95 {entry.warm_search_p95_milliseconds}ms, "
        f"narrowed p50 {entry.narrowed_search_p50_milliseconds}ms; "
        f"status {entry.status_p50_milliseconds}ms; "
        f"model load {entry.model_load_seconds}s; first search {entry.first_search_seconds}s; "
        f"peak {entry.peak_memory_megabytes}MB, held after {entry.resident_memory_megabytes}MB; "
        f"index growth {entry.index_growth_megabytes_per_edit}MB per edit; "
        f"search after an edit {entry.search_after_edit_p50_milliseconds}ms, "
        f"slowest {entry.search_after_edit_slowest_milliseconds}ms"
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
