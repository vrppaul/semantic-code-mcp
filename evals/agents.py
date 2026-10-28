"""Search judged by agents (ADR 008): each question of a corpus asked of Claude Code headless,
in arms named by what the agent is given, its own tools alone or with them this server's
search in one of its modes; each session kept, and scored by its answer, its turns, the
prompt it sent and its tool calls.

A session runs in a config directory of its own, given only a link to the owner's login, so
none of the owner's own instructions, memory, hooks or MCP servers reach it, as his
instructions may ask an agent to wait for his word; Claude Code still brings the account's
synced skills and plugins into it. The corpus's own settings, rules and skills are left out
too (`--setting-sources user`), and of the caller's environment only what Claude Code needs
to run: a headless session bills an API key wherever one is set, and the caller's effort or
thinking budget would change how the agent works. The arm's MCP config is its only one; its
server is this environment's, over an index kept between runs and brought up to date, by
meaning too where an arm searches so, before the first session.

The arms take turns going first, question by question and repeat by repeat, as what a
session reads from the cache depends on what the sessions before it wrote. A run stops at
the first session billed to anything but the login, whose arm's server did not connect, or
that Claude Code failed: such a session measures something else. Each session is kept as it
ends, its stream beside it, under the run's own name, so a run stopped, or judged later, is
scored again without new sessions.

Every session spends the owner's subscription, about $0.05 to $0.12 each on Sonnet as Claude
Code reports it.
"""

import asyncio
import os
import subprocess  # nosec B404
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum, auto
from pathlib import Path
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from evals.agent_answers import CODE_FENCE, AnswerScore, AnswerScorer, SessionAnswer
from evals.corpus import Corpus, Query, QuestionKind
from evals.server import SERVER_MODULE, ServerOptions, ServerStarter, server_settings
from semantic_code_mcp.domain.models import SearchMode

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
# The settings a session loads: the account's own, its config directory holding none; the
# corpus's project and local settings, with its rules and skills, are left out
SETTING_SOURCES = "user"
# What Claude Code needs of the caller's environment to run on the login and start a server
# that finds git; nothing else reaches a session
SESSION_VARIABLES = ("PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "TMPDIR")
JSON_OBJECT_START = "{"
SYSTEM_EVENT = "system"
INIT_SUBTYPE = "init"
ASSISTANT_EVENT = "assistant"
RESULT_EVENT = "result"
TOOL_USE_BLOCK = "tool_use"
# How a session's init names a session billed to the login, and a server that connected
LOGIN_KEY_SOURCE = "none"
CONNECTED_STATUS = "connected"
EXIT_OK = 0
# How a stream cut short by the timeout is read where it is not text
STREAM_ENCODING = "utf-8"
STREAM_DECODE_ERRORS = "replace"
# A search that brings an arm's index up to date in its mode: by meaning it first embeds
# every chunk the index holds no vector of
UP_TO_DATE_QUERY = "bring the index up to date"
UP_TO_DATE_RESULTS = 1
# What a run keeps under its directory
RUN_FILE_NAME = "run.json"
SESSIONS_FILE_NAME = "sessions.jsonl"
REPORT_FILE_NAME = "report.json"
STREAMS_DIRECTORY_NAME = "streams"
MCP_CONFIG_NAME = "mcp-{corpus}-{arm}.json"
STREAM_FILE_NAME = "{corpus}-r{repeat}-q{question:02}-{arm}.jsonl"
JSON_INDENT = 2
DURATION_DECIMALS = 1


class AgentArm(BaseModel):
    """What an agent is given besides the question, by name: Claude Code's own tools alone,
    or with them this server's search, its server set to the arm's search mode."""

    model_config = ConfigDict(frozen=True)

    name: str
    # None for Claude Code's own tools alone, which start no server
    search_mode: SearchMode | None


OWN_TOOLS_ARM = AgentArm(name="own_tools", search_mode=None)
# Every arm a run can name: own tools, and the search in each mode the server has
AGENT_ARMS: Mapping[str, AgentArm] = MappingProxyType(
    {
        arm.name: arm
        for arm in (
            OWN_TOOLS_ARM,
            *(AgentArm(name=mode, search_mode=mode) for mode in SearchMode),
        )
    }
)
# The arms of a run that names none: the agent's own tools, and the server as it ships
DEFAULT_ARMS = (OWN_TOOLS_ARM, AGENT_ARMS[SearchMode.words])


class SessionOutcome(StrEnum):
    """How a session ended."""

    # It closed with an answer in the form asked
    answered = auto()
    # It closed without one
    no_answer = auto()
    # It ran past SESSION_TIMEOUT_SECONDS and was stopped
    timed_out = auto()
    # Claude Code failed it: an error result, an exit other than 0, or no result at all
    failed = auto()


# The outcomes of the sessions that closed, whose turns and prompt count
CLOSED_OUTCOMES = frozenset({SessionOutcome.answered, SessionOutcome.no_answer})


class SessionUsage(BaseModel):
    """The tokens a session spent, and its cost as Claude Code reports it."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    output_tokens: int
    cost_usd: float

    @property
    def prompt_tokens(self) -> int:
        """Every token the session sent, read from the cache, written to it or neither: what
        one session reads from the cache, another wrote."""
        return self.input_tokens + self.cache_read_tokens + self.cache_creation_tokens


class SessionStart(BaseModel):
    """What a session's init said it runs on: the model, Claude Code's version, what it
    bills, and the status of each MCP server by name."""

    model_config = ConfigDict(frozen=True)

    model: str | None
    claude_code_version: str | None
    api_key_source: str | None
    servers: dict[str, str]


class SessionClosing(BaseModel):
    """What a session's closing result said: whether it is an error, the answer, the turns
    and the tokens."""

    model_config = ConfigDict(frozen=True)

    is_error: bool
    answer: SessionAnswer | None
    turns: int
    usage: SessionUsage


class TranscriptReading(BaseModel):
    """What a session's transcript says: how it started and closed, where it did, and the
    tools it called, by name."""

    model_config = ConfigDict(frozen=True)

    start: SessionStart | None
    closing: SessionClosing | None
    tool_calls: dict[str, int]


class SessionRun(BaseModel):
    """What Claude Code gave back for a session: everything it wrote, its stream then its
    errors; its exit code, None where the timeout stopped it; and its time."""

    model_config = ConfigDict(frozen=True)

    output: str
    exit_code: int | None
    duration_seconds: float


class SessionRecord(BaseModel):
    """One session as it ran: its question, its arm and its place in the run, how it ended,
    what it answered and spent, what it ran on, and the file its stream is kept in."""

    model_config = ConfigDict(frozen=True)

    corpus: str
    question: str
    arm: str
    repeat: int
    # Its arm's place among its question's arms in this repeat, from 0
    position: int
    outcome: SessionOutcome
    answer: SessionAnswer | None
    turns: int
    tool_calls: dict[str, int]
    usage: SessionUsage
    duration_seconds: float
    model: str | None
    claude_code_version: str | None
    # LOGIN_KEY_SOURCE for a session billed to the login
    api_key_source: str | None
    # The status of the arm's server as the session started; None in an arm without one
    server_status: str | None
    stream_file: str


class ScoredSession(BaseModel):
    """A session and its answer's score by the corpus's keys at the time it is scored; no
    score where it did not answer."""

    model_config = ConfigDict(frozen=True)

    session: SessionRecord
    score: AnswerScore | None


# A session whose stream holds no init, and one that never closed
NOT_STARTED = SessionStart(model=None, claude_code_version=None, api_key_source=None, servers={})
NOT_CLOSED = SessionClosing(
    is_error=True,
    answer=None,
    turns=0,
    usage=SessionUsage(
        input_tokens=0,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        output_tokens=0,
        cost_usd=0.0,
    ),
)


class AgentsRun(BaseModel):
    """What a run asks, as its directory keeps it: its name, the corpora by their files, the
    arms, how often each question, and the model by the name it was asked for."""

    model_config = ConfigDict(frozen=True)

    name: str
    corpus_files: tuple[Path, ...]
    arms: tuple[AgentArm, ...]
    repeats: int
    model: str


class CorpusCheckout(BaseModel):
    """A corpus as a run asks it: its questions, its working tree, and the index kept for it
    between runs."""

    model_config = ConfigDict(frozen=True)

    corpus: Corpus
    working_tree: Path
    index_directory: Path

    @property
    def name(self) -> str:
        return self.corpus.corpus.name


class SessionSetting(BaseModel):
    """Where and how one arm's sessions of a corpus run."""

    model_config = ConfigDict(frozen=True)

    corpus: str
    arm: AgentArm
    model: str
    working_tree: Path
    config_directory: Path
    mcp_config: Path


class SessionTurn(BaseModel):
    """One session's place in a run: its question and the question's number in the corpus,
    the repeat, and its arm with the arm's place among the question's arms."""

    model_config = ConfigDict(frozen=True)

    question_number: int
    query: Query
    repeat: int
    position: int
    arm: AgentArm

    def stream_file_name(self, corpus: str) -> str:
        return STREAM_FILE_NAME.format(
            corpus=corpus, repeat=self.repeat, question=self.question_number, arm=self.arm.name
        )


class McpServerEntry(BaseModel):
    """A server in Claude Code's MCP config: the command that starts it, and the settings it
    is started with beside the session's environment."""

    model_config = ConfigDict(frozen=True)

    command: str
    args: tuple[str, ...]
    env: dict[str, str]


class McpConfig(BaseModel):
    """Claude Code's MCP config, as `--mcp-config` reads it."""

    model_config = ConfigDict(frozen=True)

    mcp_servers: dict[str, McpServerEntry] = Field(serialization_alias="mcpServers")


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


class StreamServer(BaseModel):
    """An MCP server as a session's init names it."""

    model_config = ConfigDict(frozen=True)

    name: str
    status: str


class StreamEvent(BaseModel):
    """A line of Claude Code's `stream-json` output, as far as the harness reads it: the
    init, the agent's messages and the closing result."""

    model_config = ConfigDict(frozen=True)

    type: str
    subtype: str | None = None
    message: StreamMessage | None = None
    result: str | None = None
    is_error: bool = False
    num_turns: int | None = None
    usage: StreamUsage | None = None
    total_cost_usd: float | None = None
    model: str | None = None
    claude_code_version: str | None = None
    api_key_source: str | None = Field(default=None, alias="apiKeySource")
    mcp_servers: list[StreamServer] = []


class AgentSessionError(RuntimeError):
    """Claude Code could not be started for a session."""


class AgentRunStoppedError(RuntimeError):
    """A session showed the run measures something else: billed to an API key, its arm's
    server not connected, or failed by Claude Code."""


class QuestionNotAskedError(ValueError):
    """A kept run's session asked a question its corpus no longer asks."""


class RunDirectory:
    """Where a run keeps, under its own name, what it asked, each session as it ends with its
    stream, and its report; and the config directory its sessions run in, which the runs
    share."""

    def __init__(self, root: Path, config_directory: Path) -> None:
        self.root = root
        self.config_directory = config_directory

    def start(self, run: AgentsRun) -> None:
        """Make the directory of a new run, record what it asks, and make the sessions'
        config directory."""
        (self.root / STREAMS_DIRECTORY_NAME).mkdir(parents=True)
        (self.root / RUN_FILE_NAME).write_text(run.model_dump_json(indent=JSON_INDENT))
        prepare_config_directory(self.config_directory)

    def run(self) -> AgentsRun:
        """What the run asked."""
        return AgentsRun.model_validate_json((self.root / RUN_FILE_NAME).read_text())

    def write_mcp_config(self, corpus: str, arm: AgentArm, index_directory: Path) -> Path:
        """An arm's MCP config for one corpus's sessions."""
        path = self.root / MCP_CONFIG_NAME.format(corpus=corpus, arm=arm.name)
        path.write_text(mcp_config(arm, index_directory).model_dump_json(by_alias=True))
        return path

    def keep_stream(self, name: str, output: str) -> str:
        """Keep everything Claude Code wrote for a session; the name it is kept under."""
        (self.root / STREAMS_DIRECTORY_NAME / name).write_text(output)
        return name

    def add_session(self, session: SessionRecord) -> None:
        """Keep a session as it ends, so a run stopped midway keeps every session it ran."""
        with (self.root / SESSIONS_FILE_NAME).open("a") as sessions_file:
            sessions_file.write(session.model_dump_json() + "\n")

    def sessions(self) -> list[SessionRecord]:
        """Every session the run kept, in the order they ran."""
        lines = (self.root / SESSIONS_FILE_NAME).read_text().splitlines()
        return [SessionRecord.model_validate_json(line) for line in lines]

    def save_report(self, report: BaseModel) -> Path:
        """Keep the run's report, the last scoring's in place of any before it."""
        path = self.root / REPORT_FILE_NAME
        path.write_text(report.model_dump_json(indent=JSON_INDENT) + "\n")
        return path


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


def read_transcript(stream: str) -> TranscriptReading:
    """What a session did, from its `stream-json` output: from its init what it ran on, the
    tools it called, by name, and from its closing result its answer, turns and tokens."""
    tool_calls: Counter[str] = Counter()
    start: SessionStart | None = None
    closing: SessionClosing | None = None
    for line in stream.splitlines():
        try:
            event = StreamEvent.model_validate_json(line)
        except ValidationError:
            continue
        if event.type == SYSTEM_EVENT and event.subtype == INIT_SUBTYPE:
            start = session_start(event)
        if event.type == ASSISTANT_EVENT and event.message is not None:
            tool_calls.update(tool_names(event.message))
        if event.type == RESULT_EVENT:
            closing = session_closing(event)
    return TranscriptReading(start=start, closing=closing, tool_calls=dict(tool_calls))


def session_start(init: StreamEvent) -> SessionStart:
    return SessionStart(
        model=init.model,
        claude_code_version=init.claude_code_version,
        api_key_source=init.api_key_source,
        servers={server.name: server.status for server in init.mcp_servers},
    )


def session_closing(result: StreamEvent) -> SessionClosing:
    counted = result.usage or StreamUsage()
    return SessionClosing(
        is_error=result.is_error,
        answer=answer_of(result.result or ""),
        turns=result.num_turns or 0,
        usage=SessionUsage(
            input_tokens=counted.input_tokens,
            cache_read_tokens=counted.cache_read_input_tokens,
            cache_creation_tokens=counted.cache_creation_input_tokens,
            output_tokens=counted.output_tokens,
            cost_usd=result.total_cost_usd or 0.0,
        ),
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


def outcome_of(run: SessionRun, reading: TranscriptReading) -> SessionOutcome:
    """How a session ended, by its exit and its closing result."""
    if run.exit_code is None:
        return SessionOutcome.timed_out
    closing = reading.closing
    if run.exit_code != EXIT_OK or closing is None or closing.is_error:
        return SessionOutcome.failed
    if closing.answer is None:
        return SessionOutcome.no_answer
    return SessionOutcome.answered


def stop_reason(session: SessionRecord, arm: AgentArm) -> str | None:
    """Why a session shows its run measures something else, or None where it does not: billed
    to anything but the login, failed by Claude Code, or its arm's server not connected. A
    session that ran out of time is the agent's own."""
    if session.api_key_source is not None and session.api_key_source != LOGIN_KEY_SOURCE:
        return f"billed to {session.api_key_source}, not the login"
    if session.outcome == SessionOutcome.failed:
        return "Claude Code failed the session"
    if arm.search_mode is not None and session.server_status != CONNECTED_STATUS:
        return f"the arm's server was {session.server_status}, not connected"
    return None


def arms_in_turn(arms: Sequence[AgentArm], question_number: int, repeat: int) -> list[AgentArm]:
    """The arms in the order a question's sessions run in, the first moved on by one each
    question and each repeat."""
    shift = (question_number + repeat) % len(arms)
    return [*arms[shift:], *arms[:shift]]


def session_turns(queries: Sequence[Query], run: AgentsRun) -> list[SessionTurn]:
    """Every session of a corpus in the order they run: repeat by repeat, question by
    question, the arms in turn."""
    return [
        SessionTurn(
            question_number=question_number,
            query=query,
            repeat=repeat,
            position=position,
            arm=arm,
        )
        for repeat in range(run.repeats)
        for question_number, query in enumerate(queries)
        for position, arm in enumerate(arms_in_turn(run.arms, question_number, repeat))
    ]


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


def mcp_config(arm: AgentArm, index_directory: Path) -> McpConfig:
    """An arm's MCP config: no server, or this environment's server over the kept index, set
    to the arm's search mode."""
    if arm.search_mode is None:
        return McpConfig(mcp_servers={})
    options = ServerOptions(index_store=None, search_mode=arm.search_mode)
    server = McpServerEntry(
        command=sys.executable,
        args=("-m", SERVER_MODULE),
        env=server_settings(index_directory, options),
    )
    return McpConfig(mcp_servers={SERVER_NAME: server})


def allowed_tools(arm: AgentArm) -> tuple[str, ...]:
    if arm.search_mode is None:
        return OWN_TOOLS
    return (*OWN_TOOLS, SEARCH_TOOL, STATUS_TOOL)


def session_environment(config_directory: Path) -> dict[str, str]:
    """The environment a session runs in: what Claude Code needs of the caller's, and the
    agents' config directory."""
    caller = os.environ
    kept = {name: caller[name] for name in SESSION_VARIABLES if name in caller}
    return {**kept, CONFIG_DIRECTORY_VARIABLE: str(config_directory)}


def session_command(query: Query, setting: SessionSetting) -> list[str]:
    return [
        CLAUDE_COMMAND,
        "--print",
        session_prompt(query),
        "--model",
        setting.model,
        "--output-format",
        "stream-json",
        "--verbose",
        "--no-session-persistence",
        "--setting-sources",
        SETTING_SOURCES,
        "--strict-mcp-config",
        "--mcp-config",
        str(setting.mcp_config),
        "--allowedTools",
        *allowed_tools(setting.arm),
        "--disallowedTools",
        *FORBIDDEN_TOOLS,
    ]


def run_session(query: Query, setting: SessionSetting) -> SessionRun:
    """Ask one question of Claude Code headless in the corpus's checkout, as an arm has it."""
    started = time.perf_counter()
    try:
        # Claude Code from the PATH, given the harness's own question and flags
        completed = subprocess.run(  # nosec B603, B607
            session_command(query, setting),
            cwd=setting.working_tree,
            env=session_environment(setting.config_directory),
            capture_output=True,
            text=True,
            timeout=SESSION_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as expired:
        return SessionRun(
            output=joined_output(captured_text(expired.stdout), captured_text(expired.stderr)),
            exit_code=None,
            duration_seconds=time.perf_counter() - started,
        )
    except OSError as error:
        message = f"{CLAUDE_COMMAND} did not start: {error}"
        raise AgentSessionError(message) from error
    return SessionRun(
        output=joined_output(completed.stdout, completed.stderr),
        exit_code=completed.returncode,
        duration_seconds=time.perf_counter() - started,
    )


def captured_text(captured: bytes | str | None) -> str:
    """What a stopped process had written, as text: the timeout gives it as bytes."""
    if isinstance(captured, bytes):
        return captured.decode(STREAM_ENCODING, errors=STREAM_DECODE_ERRORS)
    return captured or ""


def joined_output(stream: str, errors: str) -> str:
    """A session's stream and then its errors, never on one line."""
    return f"{stream}\n{errors}"


def session_record(
    corpus: str, turn: SessionTurn, run: SessionRun, stream_file: str
) -> SessionRecord:
    """A session's record, read off what Claude Code gave back."""
    reading = read_transcript(run.output)
    start = reading.start or NOT_STARTED
    closing = reading.closing or NOT_CLOSED
    return SessionRecord(
        corpus=corpus,
        question=turn.query.text,
        arm=turn.arm.name,
        repeat=turn.repeat,
        position=turn.position,
        outcome=outcome_of(run, reading),
        answer=closing.answer,
        turns=closing.turns,
        tool_calls=reading.tool_calls,
        usage=closing.usage,
        duration_seconds=round(run.duration_seconds, DURATION_DECIMALS),
        model=start.model,
        claude_code_version=start.claude_code_version,
        api_key_source=start.api_key_source,
        server_status=start.servers.get(SERVER_NAME),
        stream_file=stream_file,
    )


def score_session(session: SessionRecord, query: Query, scorer: AnswerScorer) -> ScoredSession:
    """A session with its answer scored by the question's keys; no score where it did not
    answer."""
    if session.outcome != SessionOutcome.answered or session.answer is None:
        return ScoredSession(session=session, score=None)
    return ScoredSession(session=session, score=scorer.score(session.answer, query.answers))


def rescored(
    sessions: Sequence[SessionRecord], corpus: Corpus, scorer: AnswerScorer
) -> list[ScoredSession]:
    """A kept run's sessions of one corpus scored again by its keys as they are now."""
    queries = {query.text: query for query in corpus.queries}
    gone = sorted({session.question for session in sessions} - queries.keys())
    if gone:
        message = f"{corpus.corpus.name} no longer asks: {gone}"
        raise QuestionNotAskedError(message)
    return [score_session(session, queries[session.question], scorer) for session in sessions]


async def bring_index_up_to_date(
    checkout: CorpusCheckout, arms: Sequence[AgentArm], servers: ServerStarter
) -> None:
    """The kept index made current in every mode an arm searches in, so no session pays a
    first index: the words by an index run, and by meaning a search that embeds every chunk
    without its vector."""
    modes = dict.fromkeys(arm.search_mode for arm in arms if arm.search_mode is not None)
    for mode in modes:
        async with servers.searching(mode).connect(checkout.index_directory) as connection:
            await connection.index(checkout.working_tree)
            await connection.search(checkout.working_tree, UP_TO_DATE_QUERY, UP_TO_DATE_RESULTS)


async def measure_agents(
    checkout: CorpusCheckout,
    run: AgentsRun,
    directory: RunDirectory,
    servers: ServerStarter,
    report_session: Callable[[ScoredSession], None],
) -> list[ScoredSession]:
    """Ask every chosen question of a corpus in each arm of a run, the arms taking turns, as
    often as the run says, once the kept index is up to date; each session is kept, scored
    and reported as it ends, and the first that shows the run measures something else stops
    it."""
    await bring_index_up_to_date(checkout, run.arms, servers)
    corpus = checkout.name
    settings = {
        arm.name: SessionSetting(
            corpus=corpus,
            arm=arm,
            model=run.model,
            working_tree=checkout.working_tree,
            config_directory=directory.config_directory,
            mcp_config=directory.write_mcp_config(corpus, arm, checkout.index_directory),
        )
        for arm in run.arms
    }
    scorer = AnswerScorer(checkout.working_tree)
    sessions: list[ScoredSession] = []
    for turn in session_turns(checkout.corpus.queries, run):
        session_run = await asyncio.to_thread(run_session, turn.query, settings[turn.arm.name])
        stream_file = directory.keep_stream(turn.stream_file_name(corpus), session_run.output)
        session = session_record(corpus, turn, session_run, stream_file)
        directory.add_session(session)
        scored = score_session(session, turn.query, scorer)
        report_session(scored)
        sessions.append(scored)
        reason = stop_reason(session, turn.arm)
        if reason is not None:
            message = f"{stream_file}: {reason}; the run's sessions are kept in {directory.root}"
            raise AgentRunStoppedError(message)
    return sessions
