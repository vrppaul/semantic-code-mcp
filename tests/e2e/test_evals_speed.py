"""End-to-end tests of the speed harness: a corpus timed through server processes of its own."""

from pathlib import Path

import pytest
from evals.corpus import Answer, Corpus, PinnedSource, Query, QuestionKind
from evals.server import DEFAULT_OPTIONS, ServerStarter
from evals.speed import BURST_DIRECTORY, SpeedRun, measure_speed, timed_search_after_burst

from tests.helpers import commit_all, make_repository

PROGRESS_MODULE = '''def progressbar(items):
    """Show how far a loop over the items has come."""
    return items
'''
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
ONE_EDIT = 1
# A burst of files too small to cost more than a moment
A_FEW_FILES = 3


@pytest.mark.parametrize(
    ("by_meaning", "meaning_left_out"),
    [
        pytest.param(False, True, id="by words alone"),
        pytest.param(True, False, id="by meaning too"),
    ],
)
async def test_a_run_by_words_alone_leaves_the_server_set_to_search_by_meaning_out(
    tmp_path: Path, by_meaning: bool, meaning_left_out: bool
):
    """Its first search would embed every chunk: an hour at Sentry's size."""
    working_tree = make_repository(tmp_path / "project")
    (working_tree / "progress.py").write_text(PROGRESS_MODULE)
    commit_all(working_tree)
    run = SpeedRun(edits=ONE_EDIT, burst_files=A_FEW_FILES, by_meaning=by_meaning)

    with (tmp_path / "server.log").open("w") as server_log:
        servers = ServerStarter(server_log, DEFAULT_OPTIONS)
        speed = await measure_speed(CORPUS, working_tree, tmp_path / "index", run, servers)

    assert (speed.chunks_indexed, speed.meaning is None) == (1, meaning_left_out)


async def test_a_burst_of_build_files_leaves_nothing_behind(tmp_path: Path):
    """Written where the index never looks, as a build's, and removed once the search it may
    slow is timed."""
    working_tree = make_repository(tmp_path / "project")
    (working_tree / "progress.py").write_text(PROGRESS_MODULE)
    commit_all(working_tree)

    with (tmp_path / "server.log").open("w") as server_log:
        servers = ServerStarter(server_log, DEFAULT_OPTIONS)
        async with servers.connect(tmp_path / "index") as connection:
            await connection.index(working_tree)
            await timed_search_after_burst(
                connection, working_tree, "show a progress bar", A_FEW_FILES
            )

    assert not (working_tree / BURST_DIRECTORY).exists()
