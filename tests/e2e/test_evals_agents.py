"""End-to-end tests of the agents harness: an arm's index brought up to date through the
server's tools before its first session, Claude Code faked where a session is asked."""

import sys
from pathlib import Path

import pytest
from evals.agents import (
    AGENT_ARMS,
    AgentsRun,
    CorpusCheckout,
    RunDirectory,
    ScoredSession,
    bring_index_up_to_date,
    measure_agents,
)
from evals.corpus import Answer, Corpus, PinnedSource, Query, QuestionKind
from evals.server import DEFAULT_OPTIONS, ServerStarter

from semantic_code_mcp.domain.models import SearchMode
from tests.helpers import POSIX_HARNESS, WINDOWS, commit_all, install_fake_claude, make_repository

pytestmark = pytest.mark.skipif(sys.platform == WINDOWS, reason=POSIX_HARNESS)


PROGRESS_MODULE = '''def progressbar(items):
    """Show how far a loop over the items has come."""
    return items


def confirm(text):
    """Ask the user to say yes or no."""
    return input(text)
'''
# The module's two functions
MODULE_CHUNKS = 2
MODULE_FILES = 1
CORPUS = Corpus(
    corpus=PinnedSource(
        name="progress", repository="https://example.org/progress", commit="0" * 40
    ),
    queries=(
        Query(
            text="show a progress bar",
            kind=QuestionKind.code,
            answers=(Answer(path="progress.py", symbol="progressbar"),),
        ),
    ),
    held_out=(),
)


async def test_an_arm_by_meaning_finds_every_chunk_embedded_before_its_first_session(
    tmp_path: Path,
):
    """Else its first session would embed the whole repository inside its first search."""
    working_tree = make_repository(tmp_path / "project")
    (working_tree / "progress.py").write_text(PROGRESS_MODULE)
    commit_all(working_tree)
    checkout = CorpusCheckout(
        corpus=CORPUS, working_tree=working_tree, index_directory=tmp_path / "index"
    )
    meaning = SearchMode.words_and_meaning

    with (tmp_path / "server.log").open("w") as server_log:
        servers = ServerStarter(server_log, DEFAULT_OPTIONS)
        await bring_index_up_to_date(checkout, (AGENT_ARMS[meaning],), servers)
        async with servers.searching(meaning).connect(checkout.index_directory) as connection:
            status = await connection.status(working_tree)

    assert (status.stale_files_count, status.chunks_count, status.embedded_chunks_count) == (
        0,
        MODULE_CHUNKS,
        MODULE_CHUNKS,
    )


async def test_a_run_brings_its_arm_s_index_up_to_date_before_its_first_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    working_tree = make_repository(tmp_path / "project")
    (working_tree / "progress.py").write_text(PROGRESS_MODULE)
    commit_all(working_tree)
    checkout = CorpusCheckout(
        corpus=CORPUS, working_tree=working_tree, index_directory=tmp_path / "index"
    )
    words = AGENT_ARMS[SearchMode.words]
    run = AgentsRun(
        name="20261028T120000Z", corpus_files=(), arms=(words,), repeats=1, model="sonnet"
    )
    directory = RunDirectory(tmp_path / "runs" / run.name, tmp_path / "sessions-home")
    bin_directory = install_fake_claude(tmp_path / "bin")
    monkeypatch.setenv("PATH", f"{bin_directory}:/usr/bin:/bin")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "owner-config"))
    directory.start(run)

    reported: list[ScoredSession] = []

    with (tmp_path / "server.log").open("w") as server_log:
        servers = ServerStarter(server_log, DEFAULT_OPTIONS)
        await measure_agents(checkout, run, directory, servers, reported.append)
        async with servers.connect(checkout.index_directory) as connection:
            status = await connection.status(working_tree)

    assert (status.files_count, status.stale_files_count) == (MODULE_FILES, 0)
