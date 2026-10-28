"""Tests for an agent's answer scored through the chunks of the file it names."""

from pathlib import Path

import pytest
from evals.agent_answers import AnswerVerdict, SessionAnswer, score_answer
from evals.corpus import Answer

from semantic_code_mcp.chunkers.markdown import MarkdownChunker
from semantic_code_mcp.chunkers.python import PythonChunker

WORKING_TREE = Path("/checkouts/click")
TERMUI = "src/click/termui.py"
TERMUI_IMPL = "src/click/_termui_impl.py"
TYPES = "src/click/types.py"
UTILS = "src/click/utils.py"
ATTRIBUTION = "docs/attribution.md"
TERMUI_CODE = """
def progressbar(items):
    return ProgressBar(items)


def confirm(text):
    return input(text)
"""
TERMUI_IMPL_CODE = """
class ProgressBar:
    def render_progress(self):
        return self.bar

    def generator(self):
        yield self.bar
"""
TYPES_CODE = """
class Choice:
    def convert(self, value):
        return value


class _NumberRangeBase:
    def convert(self, value):
        return int(value)


def put_if_absent(cache, key):
    cache.setdefault(key, None)
"""
UTILS_CODE = """
def make_str(value):
    return str(value)


class ProgressBar:
    def draw(self):
        return self.bar
"""
ATTRIBUTION_TEXT = """# Attribution

Who wrote what.

## 11. Glossary

The words used above.
"""
CHUNKS_BY_PATH = {
    TERMUI: PythonChunker().chunk_string(TERMUI_CODE, Path(TERMUI)),
    TERMUI_IMPL: PythonChunker().chunk_string(TERMUI_IMPL_CODE, Path(TERMUI_IMPL)),
    TYPES: PythonChunker().chunk_string(TYPES_CODE, Path(TYPES)),
    UTILS: PythonChunker().chunk_string(UTILS_CODE, Path(UTILS)),
    ATTRIBUTION: MarkdownChunker().chunk_string(ATTRIBUTION_TEXT, Path(ATTRIBUTION)),
}
KEYS = (
    Answer(path=TERMUI, symbol="progressbar"),
    Answer(path=TERMUI_IMPL, symbol="ProgressBar"),
    Answer(path=TYPES, symbol="convert", parent="_NumberRangeBase"),
    Answer(path=TYPES, symbol="put_if_absent"),
    Answer(path=ATTRIBUTION, symbol="11. Glossary"),
)


def answer(path: str, symbol: str) -> SessionAnswer:
    return SessionAnswer(path=path, symbol=symbol)


@pytest.mark.parametrize(
    ("given", "verdict", "file_right"),
    [
        pytest.param(answer(TERMUI, "progressbar"), AnswerVerdict.right, True, id="as keyed"),
        pytest.param(
            answer(f"./{TERMUI}", "Progressbar"), AnswerVerdict.right, True, id="dot and case"
        ),
        pytest.param(
            answer(str(WORKING_TREE / TERMUI), "progressbar"),
            AnswerVerdict.right,
            True,
            id="absolute",
        ),
        pytest.param(
            answer(TERMUI, "click.termui.progressbar"),
            AnswerVerdict.right,
            True,
            id="by its module",
        ),
        pytest.param(answer(TYPES, "put_if_absent()"), AnswerVerdict.right, True, id="as a call"),
        pytest.param(
            answer(TYPES, "`_NumberRangeBase.convert`"),
            AnswerVerdict.right,
            True,
            id="by its class, in a fence",
        ),
        pytest.param(
            answer(ATTRIBUTION, "11. Glossary"), AnswerVerdict.right, True, id="a heading with dots"
        ),
        pytest.param(
            answer(ATTRIBUTION, "## 11. Glossary"),
            AnswerVerdict.right,
            True,
            id="a heading with its marks",
        ),
        pytest.param(
            answer(ATTRIBUTION, "Attribution > 11. Glossary"),
            AnswerVerdict.right,
            True,
            id="a heading under its parent",
        ),
        pytest.param(
            answer(TERMUI_IMPL, "render_progress"),
            AnswerVerdict.member,
            True,
            id="a member of the keyed class, bare",
        ),
        pytest.param(
            answer(TERMUI_IMPL, "ProgressBar.render_progress"),
            AnswerVerdict.member,
            True,
            id="a member of the keyed class, qualified",
        ),
        pytest.param(
            answer(TERMUI_IMPL, "click._termui_impl.ProgressBar.render_progress"),
            AnswerVerdict.member,
            True,
            id="a member of the keyed class, by its module and class",
        ),
        pytest.param(
            answer(UTILS, "ProgressBar.draw"),
            AnswerVerdict.other_chunk,
            False,
            id="a member of a class of the keyed name in another file",
        ),
        pytest.param(
            answer(TYPES, "_NumberRangeBase"),
            AnswerVerdict.member,
            True,
            id="the class of a keyed method",
        ),
        pytest.param(
            answer(TYPES, "Choice.convert"),
            AnswerVerdict.other_chunk,
            True,
            id="the keyed name under another class",
        ),
        pytest.param(
            answer(TERMUI, "confirm"),
            AnswerVerdict.other_chunk,
            True,
            id="another chunk of a keyed file",
        ),
        pytest.param(
            answer(UTILS, "make_str"), AnswerVerdict.other_chunk, False, id="another file"
        ),
        pytest.param(
            answer(TYPES, "convert"),
            AnswerVerdict.ambiguous,
            True,
            id="a bare name of several parents",
        ),
        pytest.param(
            answer(TERMUI, "echo"), AnswerVerdict.unresolved, True, id="a name the file lacks"
        ),
        pytest.param(
            answer(TYPES, "Missing.convert"),
            AnswerVerdict.unresolved,
            True,
            id="a class the file lacks",
        ),
        pytest.param(
            answer("frontend/termui.ts", "progressbar"),
            AnswerVerdict.unresolved,
            False,
            id="a file the server does not chunk",
        ),
    ],
)
def test_an_answer_s_verdict_is_how_the_chunks_it_names_in_its_file_meet_the_keys(
    given: SessionAnswer, verdict: AnswerVerdict, file_right: bool
):
    score = score_answer(given, KEYS, CHUNKS_BY_PATH, WORKING_TREE)

    assert (score.verdict, score.file_right) == (verdict, file_right)
