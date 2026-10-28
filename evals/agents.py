"""Search judged by agents (ADR 008): each question of a corpus asked of Claude Code headless,
in arms that differ only in what the agent is given, each session scored by its answer, its
turns, the context it read and its tool calls.

A session runs in a config directory of its own, given only a link to the owner's login, so
none of the owner's own instructions, memory, hooks or MCP servers reach it, as his
instructions may ask an agent to wait for his word; Claude Code still brings the account's
synced skills and plugins into it. The arm's MCP config is its only one;
the server is this environment's, its index kept between runs and brought up to date before
the first session. Arms take turns question by question, as a session's cost depends on
what the sessions before it left in the prompt cache: costs compare within a run, turns and
the context read across runs.

Every session spends the owner's subscription, about $0.05 to $0.12 each on Sonnet as Claude
Code reports it.
"""

import asyncio
import json
import os
import subprocess  # nosec B404
import sys
import time
from collections import Counter
from collections.abc import Callable
from enum import StrEnum, auto
from pathlib import Path
from statistics import mean

from pydantic import BaseModel, ConfigDict, ValidationError

from evals.corpus import Answer, Corpus, Query, QuestionKind
from evals.server import (
    CACHE_DIR_VARIABLE,
    SERVER_MODULE,
    ServerStarter,
    environment_without_settings,
)

CLAUDE_COMMAND = "claude"
# The model most sessions of Claude Code run on
DEFAULT_MODEL = "sonnet"
# A session that runs this long has lost its way; it counts as answering nothing
SESSION_TIMEOUT_SECONDS = 600
# The server's name in a session's MCP config, which names its tools
SERVER_NAME = "semantic-code"
SEARCH_TOOL = f"mcp__{SERVER_NAME}__search_code"
STATUS_TOOL = f"mcp__{SERVER_NAME}__index_status"
READ_TOOL = "Read"
# What every arm may use: Claude Code's own tools for finding and reading, its shell for
# reading alone
OWN_TOOLS = (
    "Read",
    "Grep",
    "Glob",
    "Bash(rg:*)",
    "Bash(grep:*)",
    "Bash(find:*)",
    "Bash(ls:*)",
    "Bash(cat:*)",
    "Bash(head:*)",
    "Bash(tail:*)",
    "Bash(sed -n:*)",
    "Bash(wc:*)",
    "Bash(git grep:*)",
    "Bash(git log:*)",
)
# No arm edits, fetches from the web or hands the task to a subagent, so a session's cost is
# its own
FORBIDDEN_TOOLS = ("Edit", "Write", "NotebookEdit", "WebFetch", "WebSearch", "Agent", "Task")
PROMPTS = {
    QuestionKind.code: "Find the code in this repository that does this: {question}",
    QuestionKind.documentation: (
        "Find where this repository's documentation answers this: {question}"
    ),
}
ANSWER_FORMAT = (
    "\n\nWhen you have found it, end your reply with one line holding only JSON: "
    '{"path": "<the file, from the repository root>", "symbol": "<the function, method, '
    'class or documentation section that does it>"}'
)
# Where Claude Code keeps its login on Linux, and the variable that moves its config; on
# macOS the login is in the keychain, which every config directory reads
CONFIG_DIRECTORY_VARIABLE = "CLAUDE_CONFIG_DIR"
DEFAULT_CONFIG_DIRECTORY = Path.home() / ".claude"
LOGIN_FILE_NAME = ".credentials.json"
# What parts a qualified name, `ProgressBar.generator` or `outer::Stack`
QUALIFIER_SEPARATORS = ("::", ".")
JSON_OBJECT_START = "{"
CODE_FENCE = "`"
# What prefixes a path written from the repository's root
CURRENT_DIRECTORY_PREFIX = "./"
ASSISTANT_EVENT = "assistant"
RESULT_EVENT = "result"
TOOL_USE_BLOCK = "tool_use"
# Where a run keeps its config directory of Claude Code's and each arm's MCP config
CONFIG_DIRECTORY_NAME = "claude-config"
MCP_CONFIG_NAME = "mcp-{arm}.json"
COST_DECIMALS = 4
MEAN_DECIMALS = 1
DURATION_DECIMALS = 1


class AgentArm(StrEnum):
    """What an agent is given besides the question."""

    # Claude Code's own tools alone
    own_tools = auto()
    # The same, and this server's search
    with_search = auto()


class SessionAnswer(BaseModel):
    """What a session answered: a file and the symbol in it."""

    model_config = ConfigDict(frozen=True)

    path: str
    symbol: str


class SessionUsage(BaseModel):
    """The tokens a session spent, and its cost as Claude Code reports it."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    output_tokens: int
    cost_usd: float


class TranscriptReading(BaseModel):
    """What a session's transcript says it did."""

    model_config = ConfigDict(frozen=True)

    answer: SessionAnswer | None
    turns: int
    tool_calls: dict[str, int]
    usage: SessionUsage


class SessionRecord(BaseModel):
    """One session: the question, the arm, the answer and how it was scored, and what it
    spent."""

    model_config = ConfigDict(frozen=True)

    corpus: str
    question: str
    arm: AgentArm
    repeat: int
    answer: SessionAnswer | None
    file_right: bool
    answer_right: bool
    turns: int
    tool_calls: dict[str, int]
    usage: SessionUsage
    duration_seconds: float


class ArmSummary(BaseModel):
    """The sessions of one arm: how many were right, and what they took on average."""

    model_config = ConfigDict(frozen=True)

    arm: AgentArm
    sessions: int
    right_answers: int
    right_files: int
    mean_turns: float
    mean_context_read_tokens: int
    mean_output_tokens: int
    sessions_that_searched: int
    read_calls: int
    # Compares only with the other arms of the same run
    cost_usd: float
    mean_duration_seconds: float


class CorpusAgents(BaseModel):
    """Every session asked of one corpus, and each arm's summary."""

    model_config = ConfigDict(frozen=True)

    corpus: str
    model: str
    arms: tuple[ArmSummary, ...]
    sessions: tuple[SessionRecord, ...]


class AgentsRun(BaseModel):
    """How a run asks its sessions: in which arms, how often each question, on which model,
    and where it keeps the corpus's index and its own files."""

    model_config = ConfigDict(frozen=True)

    arms: tuple[AgentArm, ...]
    repeats: int
    model: str
    index_directory: Path
    run_directory: Path


class SessionSetting(BaseModel):
    """Where and how one arm's sessions of a corpus run."""

    model_config = ConfigDict(frozen=True)

    corpus: str
    arm: AgentArm
    model: str
    working_tree: Path
    config_directory: Path
    mcp_config: Path


class StreamBlock(BaseModel):
    """A block of a message in Claude Code's stream: text, a tool's use or its result."""

    model_config = ConfigDict(frozen=True)

    type: str
    name: str | None = None


class StreamMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    content: list[StreamBlock] | str = []


class StreamUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    output_tokens: int = 0


class StreamEvent(BaseModel):
    """A line of Claude Code's `stream-json` output, as far as the harness reads it."""

    model_config = ConfigDict(frozen=True)

    type: str
    message: StreamMessage | None = None
    result: str | None = None
    num_turns: int | None = None
    usage: StreamUsage | None = None
    total_cost_usd: float | None = None


class AgentSessionError(RuntimeError):
    """Claude Code could not be started for a session."""


def answer_of(reply: str) -> SessionAnswer | None:
    """The answer a reply ends with: its last line that holds a JSON object, a fence's
    backticks aside; None when there is none, or it is not an answer."""
    lines = [line.strip().strip(CODE_FENCE) for line in reply.strip().splitlines()]
    objects = [line for line in lines if line.startswith(JSON_OBJECT_START)]
    if not objects:
        return None
    try:
        return SessionAnswer.model_validate_json(objects[-1])
    except ValidationError:
        return None


def answered_path(answer: SessionAnswer, working_tree: Path) -> str:
    """The answer's file from the repository's root, as the keys name files."""
    path = Path(answer.path)
    if path.is_absolute() and path.is_relative_to(working_tree):
        return path.relative_to(working_tree).as_posix()
    return answer.path.removeprefix(CURRENT_DIRECTORY_PREFIX)


def unqualified(symbol: str) -> str:
    """A qualified name's last part: `generator` of `ProgressBar.generator`."""
    for separator in QUALIFIER_SEPARATORS:
        symbol = symbol.rsplit(separator, maxsplit=1)[-1]
    return symbol.strip()


def symbol_matches(answered: str, keyed: str) -> bool:
    """Whether an answered symbol is the keyed one, its case aside: whole, as a heading with
    dots is; by its last part, as a qualified name is; or as a member of the keyed class."""
    answered_name = answered.strip().casefold()
    keyed_name = keyed.casefold()
    members_prefixes = [keyed_name + separator for separator in QUALIFIER_SEPARATORS]
    return (
        answered_name == keyed_name
        or unqualified(answered_name) == keyed_name
        or any(answered_name.startswith(prefix) for prefix in members_prefixes)
    )


def is_right_file(answer: SessionAnswer, keys: tuple[Answer, ...], working_tree: Path) -> bool:
    """Whether a key names the answer's file."""
    path = answered_path(answer, working_tree)
    return any(key.path == path for key in keys)


def is_right_answer(answer: SessionAnswer, keys: tuple[Answer, ...], working_tree: Path) -> bool:
    """Whether a key names the answer's file and its symbol."""
    path = answered_path(answer, working_tree)
    return any(key.path == path and symbol_matches(answer.symbol, key.symbol) for key in keys)


def read_transcript(stream: str) -> TranscriptReading:
    """What a session did, from its `stream-json` output: the tools it called, by name, and
    from its closing result its answer, turns and tokens. A session that never closed
    answered nothing."""
    tool_calls: Counter[str] = Counter()
    closing: StreamEvent | None = None
    for line in stream.splitlines():
        try:
            event = StreamEvent.model_validate_json(line)
        except ValidationError:
            continue
        if event.type == ASSISTANT_EVENT and event.message is not None:
            tool_calls.update(tool_names(event.message))
        if event.type == RESULT_EVENT:
            closing = event
    if closing is None:
        return TranscriptReading(
            answer=None, turns=0, tool_calls=dict(tool_calls), usage=usage_of(None, None)
        )
    return TranscriptReading(
        answer=answer_of(closing.result or ""),
        turns=closing.num_turns or 0,
        tool_calls=dict(tool_calls),
        usage=usage_of(closing.usage, closing.total_cost_usd),
    )


def tool_names(message: StreamMessage) -> list[str]:
    """The tools a message of the agent called."""
    if isinstance(message.content, str):
        return []
    return [
        block.name
        for block in message.content
        if block.type == TOOL_USE_BLOCK and block.name is not None
    ]


def usage_of(usage: StreamUsage | None, cost_usd: float | None) -> SessionUsage:
    """A session's tokens and cost, none where it never closed."""
    counted = usage or StreamUsage()
    return SessionUsage(
        input_tokens=counted.input_tokens,
        cache_read_tokens=counted.cache_read_input_tokens,
        cache_creation_tokens=counted.cache_creation_input_tokens,
        output_tokens=counted.output_tokens,
        cost_usd=cost_usd or 0.0,
    )


def summarize_arm(arm: AgentArm, sessions: list[SessionRecord]) -> ArmSummary:
    """One arm's sessions: what they got right, and what they took on average."""
    return ArmSummary(
        arm=arm,
        sessions=len(sessions),
        right_answers=sum(session.answer_right for session in sessions),
        right_files=sum(session.file_right for session in sessions),
        mean_turns=round(mean(session.turns for session in sessions), MEAN_DECIMALS),
        mean_context_read_tokens=round(
            mean(session.usage.cache_read_tokens for session in sessions)
        ),
        mean_output_tokens=round(mean(session.usage.output_tokens for session in sessions)),
        sessions_that_searched=sum(SEARCH_TOOL in session.tool_calls for session in sessions),
        read_calls=sum(session.tool_calls.get(READ_TOOL, 0) for session in sessions),
        cost_usd=round(sum(session.usage.cost_usd for session in sessions), COST_DECIMALS),
        mean_duration_seconds=round(
            mean(session.duration_seconds for session in sessions), DURATION_DECIMALS
        ),
    )


def session_prompt(query: Query) -> str:
    """What a session is asked: the question, by its kind, and the form of its answer."""
    return PROMPTS[query.kind].format(question=query.text) + ANSWER_FORMAT


def prepare_config_directory(directory: Path) -> Path:
    """A config directory of Claude Code's that holds only a link to the owner's login,
    where a file holds it."""
    directory.mkdir(parents=True, exist_ok=True)
    owner_directory = Path(os.environ.get(CONFIG_DIRECTORY_VARIABLE, DEFAULT_CONFIG_DIRECTORY))
    login = owner_directory / LOGIN_FILE_NAME
    link = directory / LOGIN_FILE_NAME
    if login.is_file() and not link.is_symlink():
        link.symlink_to(login)
    return directory


def write_mcp_config(arm: AgentArm, path: Path, index_directory: Path) -> Path:
    """An arm's MCP config: no server, or this environment's server over the kept index."""
    servers = {}
    if arm == AgentArm.with_search:
        servers[SERVER_NAME] = {
            "command": sys.executable,
            "args": ["-m", SERVER_MODULE],
            "env": {CACHE_DIR_VARIABLE: str(index_directory)},
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mcpServers": servers}))
    return path


def allowed_tools(arm: AgentArm) -> tuple[str, ...]:
    match arm:
        case AgentArm.own_tools:
            return OWN_TOOLS
        case AgentArm.with_search:
            return (*OWN_TOOLS, SEARCH_TOOL, STATUS_TOOL)


def session_environment(config_directory: Path) -> dict[str, str]:
    """The environment a session runs in: the caller's without the server's settings, so the
    servers its sessions start take the defaults the harness's own server takes, and the
    agents' config directory."""
    return {**environment_without_settings(), CONFIG_DIRECTORY_VARIABLE: str(config_directory)}


def run_session(query: Query, setting: SessionSetting) -> TranscriptReading:
    """Ask one question of Claude Code headless in the corpus's checkout, as an arm has it."""
    command = [
        CLAUDE_COMMAND,
        "--print",
        session_prompt(query),
        "--model",
        setting.model,
        "--output-format",
        "stream-json",
        "--verbose",
        "--no-session-persistence",
        "--strict-mcp-config",
        "--mcp-config",
        str(setting.mcp_config),
        "--allowedTools",
        *allowed_tools(setting.arm),
        "--disallowedTools",
        *FORBIDDEN_TOOLS,
    ]
    environment = session_environment(setting.config_directory)
    try:
        # Claude Code from the PATH, given the harness's own question and flags
        completed = subprocess.run(  # nosec B603, B607
            command,
            cwd=setting.working_tree,
            env=environment,
            capture_output=True,
            text=True,
            timeout=SESSION_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return read_transcript("")
    except OSError as error:
        message = f"{CLAUDE_COMMAND} did not start: {error}"
        raise AgentSessionError(message) from error
    return read_transcript(completed.stdout)


async def measure_agents(
    corpus: Corpus,
    working_tree: Path,
    run: AgentsRun,
    servers: ServerStarter,
    report_session: Callable[[SessionRecord], None],
) -> CorpusAgents:
    """Ask every chosen question of a corpus in each arm, the arms in turn, as often as the
    run says, once the kept index is up to date; each session is reported as it ends."""
    async with servers.connect(run.index_directory) as connection:
        await connection.index(working_tree)
    config_directory = prepare_config_directory(run.run_directory / CONFIG_DIRECTORY_NAME)
    settings = [
        SessionSetting(
            corpus=corpus.corpus.name,
            arm=arm,
            model=run.model,
            working_tree=working_tree,
            config_directory=config_directory,
            mcp_config=write_mcp_config(
                arm, run.run_directory / MCP_CONFIG_NAME.format(arm=arm), run.index_directory
            ),
        )
        for arm in run.arms
    ]
    sessions: list[SessionRecord] = []
    for repeat in range(run.repeats):
        for query in corpus.queries:
            for setting in settings:
                started = time.perf_counter()
                reading = await asyncio.to_thread(run_session, query, setting)
                duration = round(time.perf_counter() - started, DURATION_DECIMALS)
                session = scored_session(query, setting, repeat, reading, duration)
                report_session(session)
                sessions.append(session)
    return CorpusAgents(
        corpus=corpus.corpus.name,
        model=run.model,
        arms=tuple(
            summarize_arm(arm, [session for session in sessions if session.arm == arm])
            for arm in run.arms
        ),
        sessions=tuple(sessions),
    )


def scored_session(
    query: Query,
    setting: SessionSetting,
    repeat: int,
    reading: TranscriptReading,
    duration_seconds: float,
) -> SessionRecord:
    """A session's record, its answer scored against the question's keys."""
    answer = reading.answer
    keys = query.answers
    return SessionRecord(
        corpus=setting.corpus,
        question=query.text,
        arm=setting.arm,
        repeat=repeat,
        answer=answer,
        file_right=answer is not None and is_right_file(answer, keys, setting.working_tree),
        answer_right=answer is not None and is_right_answer(answer, keys, setting.working_tree),
        turns=reading.turns,
        tool_calls=reading.tool_calls,
        usage=reading.usage,
        duration_seconds=duration_seconds,
    )
