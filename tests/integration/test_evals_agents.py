"""Tests for an agents run as it asks, keeps and scores its sessions, Claude Code faked by a
script on PATH, so no session spends the owner's subscription."""

import asyncio
import io
import json
import sys
from pathlib import Path

import pytest
from evals.agent_summary import rescore_corpus, summarize_corpus
from evals.agents import (
    OWN_TOOLS_ARM,
    AgentArm,
    AgentRunStoppedError,
    AgentsRun,
    CorpusCheckout,
    RunDirectory,
    RunNotKeptError,
    ScoredSession,
    measure_agents,
    prepare_config_directory,
)
from evals.corpus import Answer, Corpus, PinnedSource, Query, QuestionKind
from evals.server import DEFAULT_OPTIONS, ServerStarter

from tests.helpers import POSIX_HARNESS, WINDOWS, install_fake_claude

pytestmark = pytest.mark.skipif(sys.platform == WINDOWS, reason=POSIX_HARNESS)


LOGIN_FILE = ".credentials.json"
# A second arm without a server, so the arms can take turns without one
OWN_TOOLS_AGAIN_ARM = AgentArm(name="own_tools_again", search_mode=None)
KEY = Answer(path="src/progress.py", symbol="progressbar")
CORPUS = Corpus(
    corpus=PinnedSource(
        name="progress", repository="https://example.org/progress", commit="0" * 40
    ),
    queries=(
        Query(text="show a progress bar", kind=QuestionKind.code, answers=(KEY,)),
        Query(text="draw how far a task has come", kind=QuestionKind.code, answers=(KEY,)),
    ),
    held_out=(),
)
RUN = AgentsRun(
    name="20261028T120000Z",
    corpus_files=(),
    arms=(OWN_TOOLS_ARM, OWN_TOOLS_AGAIN_ARM),
    repeats=1,
    model="sonnet",
)
ARM_NAMES = [arm.name for arm in RUN.arms]


class FakeHarness:
    """A run over a checkout of one file, its sessions asked of the fake Claude Code."""

    def __init__(self, root: Path) -> None:
        self.bin_directory = root / "bin"
        self.sessions_home = root / "sessions-home"
        self.checkout = CorpusCheckout(
            corpus=CORPUS, working_tree=root / "checkout", index_directory=root / "index"
        )
        self.directory = RunDirectory(root / "runs" / RUN.name, self.sessions_home)
        self.reported: list[ScoredSession] = []

    def measure(self) -> list[ScoredSession]:
        servers = ServerStarter(io.StringIO(), DEFAULT_OPTIONS)
        return asyncio.run(
            measure_agents(self.checkout, RUN, self.directory, servers, self.reported.append)
        )

    def calls(self) -> list[dict]:
        lines = (self.bin_directory / "calls.jsonl").read_text().splitlines()
        return [json.loads(line) for line in lines]


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeHarness:
    fake = FakeHarness(tmp_path)
    (fake.checkout.working_tree / "src").mkdir(parents=True)
    (fake.checkout.working_tree / "src" / "progress.py").write_text(
        "def progressbar(items):\n    return items\n"
    )
    install_fake_claude(fake.bin_directory)
    owner_config = tmp_path / "owner-config"
    owner_config.mkdir()
    (owner_config / LOGIN_FILE).write_text("{}")
    monkeypatch.setenv("PATH", f"{fake.bin_directory}:/usr/bin:/bin")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(owner_config))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a key that would bill an account")
    fake.directory.start(RUN)
    return fake


def test_a_run_keeps_each_session_and_its_stream_as_it_ends_the_arms_taking_turns(
    harness: FakeHarness,
):
    harness.measure()

    kept = harness.directory.sessions()
    assert [(session.question, session.arm, session.position) for session in kept] == [
        ("show a progress bar", "own_tools", 0),
        ("show a progress bar", "own_tools_again", 1),
        ("draw how far a task has come", "own_tools_again", 0),
        ("draw how far a task has come", "own_tools", 1),
    ]
    assert [
        (harness.directory.root / "streams" / session.stream_file).is_file() for session in kept
    ] == [True, True, True, True]


def test_a_kept_run_scored_again_sums_as_the_run_did(harness: FakeHarness):
    measured = summarize_corpus(CORPUS.corpus.name, ARM_NAMES, harness.measure())

    again = rescore_corpus(harness.checkout, RUN, harness.directory.sessions())

    assert (again.arms, again.comparisons) == (measured.arms, measured.comparisons)


def test_a_session_leaves_out_the_corpus_s_settings_the_caller_s_key_and_the_owner_s_home(
    harness: FakeHarness,
):
    harness.measure()

    first = harness.calls()[0]
    arguments = first["arguments"]
    environment = first["environment"]
    assert (
        arguments[arguments.index("--setting-sources") + 1],
        "--strict-mcp-config" in arguments,
        "ANTHROPIC_API_KEY" in environment,
        environment["HOME"],
    ) == ("user", True, False, str(harness.sessions_home))


def test_a_run_stops_at_a_failed_session_and_keeps_it(harness: FakeHarness):
    (harness.bin_directory / "fail").touch()

    with pytest.raises(AgentRunStoppedError):
        harness.measure()

    assert len(harness.directory.sessions()) == 1


def test_a_run_just_started_keeps_no_session_yet(harness: FakeHarness):
    assert harness.directory.sessions() == []


def test_a_run_never_started_is_not_kept(tmp_path: Path):
    with pytest.raises(RunNotKeptError):
        RunDirectory(tmp_path / "runs" / "never", tmp_path / "sessions-home").run()


def test_the_sessions_config_links_the_owner_s_login_again_where_a_file_took_its_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    owner_config = tmp_path / "owner-config"
    owner_config.mkdir()
    (owner_config / LOGIN_FILE).write_text("{}")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(owner_config))
    sessions_config = tmp_path / "sessions-home" / ".claude"
    sessions_config.mkdir(parents=True)
    (sessions_config / LOGIN_FILE).write_text("a login written over the link")

    prepare_config_directory(sessions_config)

    assert (sessions_config / LOGIN_FILE).resolve() == (owner_config / LOGIN_FILE).resolve()
