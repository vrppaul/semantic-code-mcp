"""Measure search quality, speed, or search as agents use it, on the pinned corpora.

    uv run python -m evals quality [--corpus NAME ...] [--store STORE] [--search-mode MODE]
        [--save-baseline] [--held-out]
    uv run python -m evals speed [--corpus NAME ...] [--corpus-file PATH ...] [--store STORE]
        [--edits N] [--words-only] [--save-baseline]
    uv run python -m evals agents (--corpus NAME | --corpus-file PATH) ... [--arm ARM ...]
        [--repeats N] [--model MODEL]
    uv run python -m evals rescore RUN

Each run indexes into a fresh temporary directory; checkouts of the corpora are kept
under ~/.cache/semantic-code-mcp-evals and reused. Prints one line per corpus next to the
saved baseline, writes the whole report as JSON to `evals/results/`, and with
--save-baseline makes it the new baseline; a corpus kept outside the repository
(`--corpus-file`, as a large or a private one is) is never saved to it, nor a run in a store
other than the server's default (`--store`, as when two stores are compared in turns), nor a
quality run in another search mode than the server's default (`--search-mode`, which sets
the servers' setting to it, `words_and_meaning` searching by meaning too), nor a speed run
without its server set to search by meaning (`--words-only`, as that server's first search
embeds every chunk, an hour at Sentry's size). `agents` asks
Claude Code headless each question of the corpora it names in each arm (`evals/agents.py`),
spending the owner's subscription; it prints each session as it ends, then each arm's sums
and every two arms compared, and keeps no baseline, as its sessions hold the questions of
private corpora. Its indexes are kept between runs, and each run under its own name outside
the repository: what it asked, its sessions and their streams, its report. `rescore` scores
a kept run again by its corpora's keys as they are now, as once a judge has added an answer.

Exits 0, or 1 when a tool call fails or times out, a clone fails, a corpus names an answer its
checkout lacks, Claude Code does not start, a session stops an agents run, or a kept run's
corpus no longer asks one of its questions.
"""

import argparse
import asyncio
import subprocess  # nosec B404
import sys
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum, auto
from pathlib import Path

from mcp.shared.exceptions import MCPError
from pydantic import BaseModel, ConfigDict, ValidationError

from evals.agent_summary import (
    AgentsReport,
    AnswerToJudge,
    ArmComparison,
    ArmSummary,
    CorpusAgents,
    PairedDifference,
    agents_report,
    rescore_corpus,
    summarize_corpus,
)
from evals.agents import (
    AGENT_ARMS,
    DEFAULT_ARMS,
    DEFAULT_MODEL,
    AgentRunStoppedError,
    AgentSessionError,
    AgentsRun,
    CorpusCheckout,
    CorpusNameSharedError,
    QuestionNotAskedError,
    RunDirectory,
    RunNotKeptError,
    ScoredSession,
    measure_agents,
    refuse_shared_names,
)
from evals.corpus import (
    COMMIT_PREFIX_LENGTH,
    Corpus,
    available_corpus_names,
    checkout,
    corpus_file_of,
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
from evals.server import DEFAULT_OPTIONS, ServerOptions, ServerStarter, ToolCallFailedError
from evals.speed import (
    CorpusSpeed,
    Machine,
    MeaningSpeed,
    SpeedRun,
    current_machine,
    measure_speed,
)
from evals.targets import TargetCrossed, crossed_targets
from semantic_code_mcp.domain.models import SearchMode, StoreKind
from semantic_code_mcp.logging import configure_logging

CHECKOUTS_ROOT = Path.home() / ".cache" / "semantic-code-mcp-evals"
EVALS_DIRECTORY = Path(__file__).parent
BASELINE_DIRECTORY = EVALS_DIRECTORY / "baseline"
RESULTS_DIRECTORY = EVALS_DIRECTORY / "results"
QUALITY_FILE = "quality.json"
SPEED_FILE = "speed.json"
# Where `agents` keeps each corpus's index between runs, as one of a large corpus takes an
# hour; each run's own directory, by its name, outside the repository, as its streams hold
# a private corpus's code; and the home every run's sessions share
INDEXES_ROOT = CHECKOUTS_ROOT / "indexes"
AGENTS_RUNS_ROOT = CHECKOUTS_ROOT / "agents" / "runs"
AGENTS_SESSIONS_HOME = CHECKOUTS_ROOT / "agents" / "home"
# A run's name: the moment it started, in UTC
RUN_NAME_FORMAT = "%Y%m%dT%H%M%SZ"
TOKENS_PER_THOUSAND = 1000
# How much of a question a session's line shows
QUESTION_SHOWN_LENGTH = 60
SERVER_LOG_FILE = "server.log"
# Edits enough for the slowest search after an edit to show a store's periodic upkeep, the
# costliest an edit can meet: LanceDB compacts and rebuilds its full-text index at its first
# upkeep, the full index's, and once in its UPKEEPS_PER_COMPACTION after, so at the last
# edit's; a unit test holds the two together, as evals import no store
DEFAULT_EDITS = 20
JSON_INDENT = 2


class EvalsCommand(StrEnum):
    """What a run of the harness measures."""

    quality = auto()
    speed = auto()
    agents = auto()
    rescore = auto()


# The commands that keep a baseline, which holds the default store's figures alone
BASELINE_COMMANDS = frozenset({EvalsCommand.quality, EvalsCommand.speed})
# How a report that named no store names the one it ran in
DEFAULT_STORE_NAME = "the server's default"
# How a report that named no search mode names the one its servers searched in
DEFAULT_SEARCH_NAME = "the server's default"
EXIT_OK = 0
EXIT_FAILED = 1


class QualityReport(BaseModel):
    """Search quality over every corpus measured, in the store and the search mode a run
    named."""

    model_config = ConfigDict(frozen=True)

    # None for the server's default store at the commit measured
    index_store: StoreKind | None
    # None for the server's default search at the commit measured
    search_mode: SearchMode | None
    corpora: tuple[CorpusQuality, ...]


class SpeedReport(BaseModel):
    """Speeds over every corpus measured, the machine they were measured on, and the store
    a run named."""

    model_config = ConfigDict(frozen=True)

    machine: Machine
    # None for the server's default store at the commit measured
    index_store: StoreKind | None
    corpora: tuple[CorpusSpeed, ...]


def main() -> int:
    arguments = parse_arguments(sys.argv[1:])
    # The answer check runs the chunker in this process; its debug lines are noise here
    configure_logging(debug=False)
    RESULTS_DIRECTORY.mkdir(exist_ok=True)
    BASELINE_DIRECTORY.mkdir(exist_ok=True)
    try:
        match arguments.command:
            case EvalsCommand.quality:
                run_quality(
                    load_corpora(arguments.corpus),
                    ServerOptions(index_store=arguments.store, search_mode=arguments.search_mode),
                    save_baseline=arguments.save_baseline,
                    held_out=arguments.held_out,
                )
            case EvalsCommand.speed:
                run_speed(
                    chosen_corpora(arguments.corpus, arguments.corpus_file),
                    index_store=arguments.store,
                    run=SpeedRun(edits=arguments.edits, by_meaning=not arguments.words_only),
                    save_baseline=arguments.save_baseline,
                )
            case EvalsCommand.agents:
                run_agents(
                    AgentsRun(
                        name=datetime.now(tz=UTC).strftime(RUN_NAME_FORMAT),
                        corpus_files=corpus_files(arguments.corpus, arguments.corpus_file),
                        arms=tuple(AGENT_ARMS[name] for name in arguments.arm) or DEFAULT_ARMS,
                        repeats=arguments.repeats,
                        model=arguments.model,
                    )
                )
            case EvalsCommand.rescore:
                rescore_agents(arguments.run)
    except (
        ToolCallFailedError,
        CorpusMistakeError,
        MCPError,
        AgentSessionError,
        AgentRunStoppedError,
        QuestionNotAskedError,
        CorpusNameSharedError,
        RunNotKeptError,
        subprocess.CalledProcessError,
    ) as error:
        print(error, file=sys.stderr)
        return EXIT_FAILED
    return EXIT_OK


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m evals", description=__doc__)
    # A command that names no search mode searches as the server's default does
    parser.set_defaults(search_mode=None)
    commands = parser.add_subparsers(dest="command", required=True)
    quality = commands.add_parser(
        EvalsCommand.quality, help="search quality: where the answers land"
    )
    speed = commands.add_parser(
        EvalsCommand.speed, help="indexing, search, memory and index growth"
    )
    for command in (quality, speed):
        command.add_argument(
            "--corpus", action="append", default=[], choices=available_corpus_names()
        )
        # A store other than the server's default, never saved to the baseline
        command.add_argument("--store", type=StoreKind, choices=list(StoreKind))
    quality.add_argument("--save-baseline", action="store_true")
    # Another search mode than the server's default, never saved to the baseline
    quality.add_argument("--search-mode", type=SearchMode, choices=list(SearchMode))
    # A corpus kept outside the repository, never saved to the baseline the repository keeps
    speed_corpora = speed.add_mutually_exclusive_group()
    speed_corpora.add_argument("--corpus-file", action="append", default=[], type=Path)
    speed_corpora.add_argument("--save-baseline", action="store_true")
    # The held-out questions are asked on every run but shown only once a choice is made
    quality.add_argument("--held-out", action="store_true")
    speed.add_argument("--edits", type=positive_integer, default=DEFAULT_EDITS)
    # Words alone, never saved to the baseline: the server set to search by meaning embeds
    # every chunk on its first search, an hour at Sentry's size
    speed.add_argument("--words-only", action="store_true")
    agents = commands.add_parser(
        EvalsCommand.agents, help="search judged by agents: Claude Code headless"
    )
    agents.add_argument("--corpus", action="append", default=[], choices=available_corpus_names())
    # A corpus kept outside the repository, as a private one's questions are
    agents.add_argument("--corpus-file", action="append", default=[], type=Path)
    agents.add_argument("--arm", action="append", default=[], choices=list(AGENT_ARMS))
    agents.add_argument("--repeats", type=positive_integer, default=1)
    agents.add_argument("--model", default=DEFAULT_MODEL)
    rescore = commands.add_parser(
        EvalsCommand.rescore,
        help="a kept agents run scored again by its corpora's keys as they are now",
    )
    rescore.add_argument("run", help="the run's name, as `agents` printed it")
    arguments = parser.parse_args(argv)
    if arguments.command == EvalsCommand.agents and not (arguments.corpus or arguments.corpus_file):
        # The pinned corpora are small enough to grep: about 130 sessions that show nothing
        parser.error("agents asks only the corpora it names: give --corpus or --corpus-file")
    keeps_a_baseline = arguments.command in BASELINE_COMMANDS
    if keeps_a_baseline and arguments.save_baseline and arguments.store is not None:
        parser.error("--save-baseline keeps the default store's figures: leave --store out")
    if keeps_a_baseline and arguments.save_baseline and arguments.search_mode is not None:
        parser.error("--save-baseline keeps the default search's figures: leave --search-mode out")
    if arguments.command == EvalsCommand.speed and arguments.save_baseline and arguments.words_only:
        parser.error("--save-baseline keeps the speeds by meaning too: leave --words-only out")
    return arguments


def positive_integer(text: str) -> int:
    """An argument that must be a whole number above zero."""
    value = int(text)
    if value < 1:
        message = f"must be at least 1, not {value}"
        raise argparse.ArgumentTypeError(message)
    return value


def run_quality(
    corpora: list[Corpus], options: ServerOptions, *, save_baseline: bool, held_out: bool
) -> None:
    report = QualityReport(
        index_store=options.index_store,
        search_mode=options.search_mode,
        corpora=tuple(asyncio.run(collect_quality(corpora, options))),
    )
    baseline = read_report(BASELINE_DIRECTORY / QUALITY_FILE, QualityReport)
    baseline_by_corpus: dict[str, CorpusQuality] = {}
    if baseline is not None:
        baseline_by_corpus = {entry.corpus: entry for entry in baseline.corpora}
        print(describe_stores(report.index_store, baseline.index_store))
        print(describe_searches(report.search_mode, baseline.search_mode))
    for entry in report.corpora:
        print(describe_quality(entry, baseline_by_corpus.get(entry.corpus), held_out=held_out))
    write_report(report, QUALITY_FILE, save_baseline=save_baseline)


def run_speed(
    corpora: list[Corpus], *, index_store: StoreKind | None, run: SpeedRun, save_baseline: bool
) -> None:
    report = SpeedReport(
        machine=current_machine(),
        index_store=index_store,
        corpora=tuple(
            asyncio.run(
                collect_speed(
                    corpora, ServerOptions(index_store=index_store, search_mode=None), run
                )
            )
        ),
    )
    baseline = read_report(BASELINE_DIRECTORY / SPEED_FILE, SpeedReport)
    # Speeds compare only on the machine they were measured on
    baseline_by_corpus: dict[str, CorpusSpeed] = {}
    if baseline is not None and baseline.machine == report.machine:
        baseline_by_corpus = {entry.corpus: entry for entry in baseline.corpora}
        print(describe_stores(report.index_store, baseline.index_store))
    for entry in report.corpora:
        print(describe_speed(entry, baseline_by_corpus.get(entry.corpus)))
        for crossing in crossed_targets(entry):
            print(describe_crossing(crossing))
    write_report(report, SPEED_FILE, save_baseline=save_baseline)


def describe_crossing(crossing: TargetCrossed) -> str:
    """A line of a target a corpus's speeds crossed, which its change must explain."""
    target = crossing.target
    return (
        f"  crosses a target: {target.figure} {crossing.measured}, "
        f"at most {target.most} for {target.size.name}"
    )


def describe_stores(measured: StoreKind | None, baseline: StoreKind | None) -> str:
    """The line naming the store a run measured and the one its baseline did."""
    return f"store: {store_name(measured)}; the baseline's: {store_name(baseline)}"


def store_name(index_store: StoreKind | None) -> str:
    return DEFAULT_STORE_NAME if index_store is None else index_store


def describe_searches(measured: SearchMode | None, baseline: SearchMode | None) -> str:
    """The line naming the search mode a run measured and the one its baseline did."""
    return f"search: {search_name(measured)}; the baseline's: {search_name(baseline)}"


def search_name(search_mode: SearchMode | None) -> str:
    return DEFAULT_SEARCH_NAME if search_mode is None else search_mode


def chosen_corpora(names: list[str], files: list[Path]) -> list[Corpus]:
    """The corpora a run measures: those named and those read from files; every corpus of
    `evals/corpora/` when neither is given."""
    named = load_corpora(names) if names or not files else []
    return [*named, *(load_corpus_file(path) for path in files)]


def corpus_files(names: list[str], files: list[Path]) -> tuple[Path, ...]:
    """The files of the corpora a run names, by name and by path."""
    return (*(corpus_file_of(name) for name in names), *(path.resolve() for path in files))


def run_directory(name: str) -> RunDirectory:
    return RunDirectory(AGENTS_RUNS_ROOT / name, AGENTS_SESSIONS_HOME)


def corpus_checkout(corpus_file: Path) -> CorpusCheckout:
    """A corpus a run asks, checked out, with the index kept for it."""
    corpus = load_corpus_file(corpus_file)
    source = corpus.corpus
    return CorpusCheckout(
        corpus=corpus,
        working_tree=checkout(source, CHECKOUTS_ROOT),
        index_directory=INDEXES_ROOT / f"{source.name}-{source.commit[:COMMIT_PREFIX_LENGTH]}",
    )


def run_agents(run: AgentsRun) -> None:
    refuse_shared_names([load_corpus_file(corpus_file) for corpus_file in run.corpus_files])
    directory = run_directory(run.name)
    directory.start(run)
    print(f"agents run {run.name}, kept in {directory.root}", flush=True)
    finish_agents(run, directory, asyncio.run(collect_agents(run, directory)))


async def collect_agents(run: AgentsRun, directory: RunDirectory) -> list[CorpusAgents]:
    measured: list[CorpusAgents] = []
    arms = [arm.name for arm in run.arms]
    for corpus_file in run.corpus_files:
        corpus = corpus_checkout(corpus_file)
        with (RESULTS_DIRECTORY / SERVER_LOG_FILE).open("a") as server_log:
            server_log.write(log_header(EvalsCommand.agents, corpus.corpus))
            # The agents' own servers take the default store, so this one does too
            servers = ServerStarter(server_log, DEFAULT_OPTIONS)
            sessions = await measure_agents(corpus, run, directory, servers, print_session)
        measured.append(summarize_corpus(corpus.name, arms, sessions))
    return measured


def rescore_agents(name: str) -> None:
    """A kept run scored again by its corpora's keys as they are now, a run stopped midway
    by the sessions it kept."""
    directory = run_directory(name)
    run = directory.run()
    kept = directory.sessions()
    measured = [
        rescore_corpus(corpus_checkout(corpus_file), run, kept) for corpus_file in run.corpus_files
    ]
    finish_agents(run, directory, measured)


def finish_agents(run: AgentsRun, directory: RunDirectory, corpora: list[CorpusAgents]) -> None:
    report = agents_report(run, corpora)
    print(describe_agents(report))
    print(f"report: {directory.save_report(report)}")


def print_session(scored: ScoredSession) -> None:
    """A line of one session, printed as it ends, as a run of many takes a while."""
    print(describe_session(scored), flush=True)


def describe_session(scored: ScoredSession) -> str:
    session = scored.session
    answer = session.answer
    answered = "no answer" if answer is None else f"{answer.path}:{answer.symbol}"
    verdict = session.outcome if scored.score is None else scored.score.verdict
    prompt = session.usage.prompt_tokens // TOKENS_PER_THOUSAND
    return (
        f"  {session.arm} {verdict}: {session.turns} turns, prompt {prompt}k, "
        f"tools {session.tool_calls}, {session.duration_seconds}s; "
        f"{session.question[:QUESTION_SHOWN_LENGTH]} -> {answered}"
    )


def describe_agents(report: AgentsReport) -> str:
    """The run's lines: what its sessions ran on, then each corpus's arms, comparisons and
    answers to judge."""
    ran_on = (
        f"run {report.run}: {report.sessions} sessions on {', '.join(report.models)} "
        f"(asked as {report.model_asked}), Claude Code {', '.join(report.claude_code_versions)}; "
        f"reported cost ${report.reported_cost_usd}, never compared"
    )
    lines = [ran_on]
    for corpus in report.corpora:
        lines.append(f"{corpus.corpus}:")
        lines += [describe_arm(summary) for summary in corpus.arms]
        lines += [describe_comparison(comparison) for comparison in corpus.comparisons]
        lines += [describe_to_judge(answer) for answer in corpus.to_judge]
    return "\n".join(lines)


def describe_arm(summary: ArmSummary) -> str:
    """A line of one arm's sums over its sessions."""
    if not summary.sessions:
        return f"  {summary.arm}: no sessions"
    outcomes = ", ".join(f"{outcome} {count}" for outcome, count in summary.outcomes.items())
    return (
        f"  {summary.arm}: right {summary.right_answers}/{summary.sessions} "
        f"({summary.answers_as_members} as members, {summary.answers_judged_right} judged; "
        f"file {summary.right_files}), "
        f"to judge {summary.answers_to_judge}; {outcomes}; turns {summary.mean_turns}, "
        f"prompt {summary.mean_prompt_tokens}, output {summary.mean_output_tokens}, "
        f"{summary.mean_duration_seconds}s; searched {summary.sessions_that_searched}, "
        f"reads {summary.read_calls}"
    )


def describe_comparison(comparison: ArmComparison) -> str:
    return (
        f"  {comparison.arm} against {comparison.against} over {comparison.questions} "
        f"questions, {comparison.sessions_left_out} sessions left out: "
        f"turns {describe_difference(comparison.turns)}; "
        f"prompt tokens {describe_difference(comparison.prompt_tokens)}"
    )


def describe_difference(difference: PairedDifference) -> str:
    return (
        f"{difference.mean:+} [{difference.low:+}, {difference.high:+}] "
        f"(fewer {difference.fewer}, equal {difference.equal}, more {difference.more})"
    )


def describe_to_judge(to_judge: AnswerToJudge) -> str:
    given = to_judge.answer
    file_verdict = "its file keyed" if given.file_right else "its file not keyed"
    return (
        f"  to judge, {given.verdict}, {file_verdict}: "
        f"{given.question[:QUESTION_SHOWN_LENGTH]} -> {given.path}:{given.symbol} "
        f"({', '.join(to_judge.arms)})"
    )


async def collect_quality(corpora: list[Corpus], options: ServerOptions) -> list[CorpusQuality]:
    measured: list[CorpusQuality] = []
    off_topic = load_off_topic_questions()
    for corpus in corpora:
        working_tree = checkout(corpus.corpus, CHECKOUTS_ROOT)
        with (
            tempfile.TemporaryDirectory() as index_directory,
            (RESULTS_DIRECTORY / SERVER_LOG_FILE).open("a") as server_log,
        ):
            server_log.write(log_header(EvalsCommand.quality, corpus))
            servers = ServerStarter(server_log, options)
            async with servers.connect(Path(index_directory)) as connection:
                measured.append(await measure_quality(corpus, working_tree, off_topic, connection))
    return measured


async def collect_speed(
    corpora: list[Corpus], options: ServerOptions, run: SpeedRun
) -> list[CorpusSpeed]:
    measured: list[CorpusSpeed] = []
    for corpus in corpora:
        working_tree = checkout(corpus.corpus, CHECKOUTS_ROOT)
        with (
            tempfile.TemporaryDirectory() as index_directory,
            (RESULTS_DIRECTORY / SERVER_LOG_FILE).open("a") as server_log,
        ):
            server_log.write(log_header(EvalsCommand.speed, corpus))
            servers = ServerStarter(server_log, options)
            measured.append(
                await measure_speed(corpus, working_tree, Path(index_directory), run, servers)
            )
    return measured


def log_header(command: EvalsCommand, corpus: Corpus) -> str:
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
        f"narrowed p50 {entry.narrowed_search_p50_milliseconds}ms, "
        f"a long question p50 {entry.long_question_p50_milliseconds}ms; "
        f"status {entry.status_p50_milliseconds}ms; "
        f"{meaning_summary(entry.meaning)}; "
        f"first search {entry.first_search_seconds}s; "
        f"peak {entry.peak_memory_megabytes}MB, held after {entry.resident_memory_megabytes}MB; "
        f"index growth {entry.index_growth_megabytes_per_edit}MB per edit; "
        f"search after an edit {entry.search_after_edit_p50_milliseconds}ms, "
        f"slowest {entry.search_after_edit_slowest_milliseconds}ms"
    )


def meaning_summary(meaning: MeaningSpeed | None) -> str:
    """The speeds of the server set to search by meaning too, or that the run left it out."""
    if meaning is None:
        return "meaning left out"
    return (
        f"by meaning: model load {meaning.model_load_seconds}s, first search "
        f"{meaning.first_search_seconds}s, warm p50 {meaning.warm_search_p50_milliseconds}ms, "
        f"peak {meaning.peak_memory_megabytes}MB, held after "
        f"{meaning.resident_memory_megabytes}MB"
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
