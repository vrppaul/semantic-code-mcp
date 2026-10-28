"""Tests for an agent's answer scored against the files of a checkout."""

from pathlib import Path

import pytest
from evals.agent_answers import AnswerScorer, AnswerVerdict, SessionAnswer
from evals.corpus import Answer

KEYS = (Answer(path="src/progress.py", symbol="progressbar"),)
PROGRESS_CODE = "def progressbar(items):\n    return items\n"


@pytest.mark.parametrize(
    ("given", "verdict"),
    [
        pytest.param(
            SessionAnswer(path="src/progress.py", symbol="progressbar"),
            AnswerVerdict.right,
            id="a file of the checkout",
        ),
        pytest.param(
            SessionAnswer(path="../outside.py", symbol="progressbar"),
            AnswerVerdict.unresolved,
            id="a file outside it",
        ),
    ],
)
def test_an_answer_names_only_chunks_of_the_checkout_s_own_files(
    tmp_path: Path, given: SessionAnswer, verdict: AnswerVerdict
):
    working_tree = tmp_path / "checkout"
    (working_tree / "src").mkdir(parents=True)
    (working_tree / "src" / "progress.py").write_text(PROGRESS_CODE)
    (tmp_path / "outside.py").write_text(PROGRESS_CODE)

    score = AnswerScorer(working_tree).score(given, KEYS)

    assert score.verdict == verdict
