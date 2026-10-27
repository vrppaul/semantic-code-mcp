"""Tests for search judged by agents: how a session's transcript is read and its answer scored."""

import json
from pathlib import Path

import pytest
from evals.agents import (
    AgentArm,
    SessionAnswer,
    SessionRecord,
    SessionUsage,
    answer_of,
    is_right_answer,
    is_right_file,
    read_transcript,
    summarize_arm,
)
from evals.corpus import Answer

WORKING_TREE = Path("/checkouts/click")
KEYS = (
    Answer(path="src/click/termui.py", symbol="progressbar"),
    Answer(path="src/click/_termui_impl.py", symbol="ProgressBar"),
    Answer(path="docs/projects/attribution.md", symbol="11. Glossary"),
)
NO_TOOLS: dict[str, int] = {}
SEARCH_TOOL_CALLS = {"mcp__semantic-code__search_code": 1}
GREP_AND_READ_CALLS = {"Grep": 2, "Read": 1}


def answer(path: str, symbol: str) -> SessionAnswer:
    return SessionAnswer(path=path, symbol=symbol)


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        pytest.param(
            'It is there.\n{"path": "a.py", "symbol": "f"}', answer("a.py", "f"), id="last line"
        ),
        pytest.param(
            'It is there.\n```json\n{"path": "a.py", "symbol": "f"}\n```',
            answer("a.py", "f"),
            id="in a fence",
        ),
        pytest.param(
            '{"path": "old.py", "symbol": "g"}\nNo, rather:\n{"path": "a.py", "symbol": "f"}',
            answer("a.py", "f"),
            id="the last of two",
        ),
        pytest.param("I could not find it.", None, id="no JSON"),
        pytest.param('{"path": "a.py"}', None, id="no symbol"),
        pytest.param('{"path": "a.py", "symbol": ', None, id="broken JSON"),
    ],
)
def test_a_session_s_answer_is_the_last_json_line_of_its_reply(
    reply: str, expected: SessionAnswer | None
):
    assert answer_of(reply) == expected


@pytest.mark.parametrize(
    ("given", "right"),
    [
        pytest.param(answer("src/click/termui.py", "progressbar"), True, id="as keyed"),
        pytest.param(answer("./src/click/termui.py", "Progressbar"), True, id="dot and case"),
        pytest.param(
            answer(str(WORKING_TREE / "src/click/termui.py"), "progressbar"), True, id="absolute"
        ),
        pytest.param(
            answer("src/click/termui.py", "click.termui.progressbar"), True, id="qualified"
        ),
        pytest.param(
            answer("src/click/_termui_impl.py", "ProgressBar.render_progress"),
            True,
            id="a member of the keyed class",
        ),
        pytest.param(
            answer("docs/projects/attribution.md", "11. Glossary"), True, id="a heading with dots"
        ),
        pytest.param(answer("src/click/termui.py", "confirm"), False, id="another symbol"),
        pytest.param(answer("src/click/utils.py", "progressbar"), False, id="another file"),
    ],
)
def test_an_answer_is_right_when_a_key_names_its_file_and_its_symbol(
    given: SessionAnswer, right: bool
):
    assert is_right_answer(given, KEYS, WORKING_TREE) is right


@pytest.mark.parametrize(
    ("given", "right"),
    [
        pytest.param(answer("src/click/termui.py", "confirm"), True, id="a keyed file"),
        pytest.param(answer("src/click/utils.py", "progressbar"), False, id="another file"),
    ],
)
def test_an_answer_s_file_is_right_when_a_key_names_it(given: SessionAnswer, right: bool):
    assert is_right_file(given, KEYS, WORKING_TREE) is right


def stream_line(event: dict) -> str:
    return json.dumps(event)


def tool_use_event(*names: str) -> dict:
    content = [{"type": "tool_use", "name": name, "input": {}} for name in names]
    return {"type": "assistant", "message": {"content": content}}


TRANSCRIPT = "\n".join(
    [
        stream_line({"type": "system", "subtype": "init", "tools": ["Grep", "Read"]}),
        stream_line(tool_use_event("Grep")),
        "not a line of JSON",
        stream_line(tool_use_event("Grep", "Read")),
        stream_line(
            {
                "type": "result",
                "num_turns": 3,
                "result": 'Found.\n{"path": "a.py", "symbol": "f"}',
                "total_cost_usd": 0.05,
                "usage": {
                    "input_tokens": 4,
                    "cache_read_input_tokens": 30000,
                    "cache_creation_input_tokens": 9000,
                    "output_tokens": 250,
                },
            }
        ),
    ]
)


def test_a_transcript_gives_the_session_s_answer_turns_tool_calls_and_tokens():
    reading = read_transcript(TRANSCRIPT)

    assert (reading.answer, reading.turns, reading.tool_calls, reading.usage) == (
        answer("a.py", "f"),
        3,
        GREP_AND_READ_CALLS,
        SessionUsage(
            input_tokens=4,
            cache_read_tokens=30000,
            cache_creation_tokens=9000,
            output_tokens=250,
            cost_usd=0.05,
        ),
    )


def test_a_transcript_that_never_ends_gives_no_answer_and_no_turns():
    reading = read_transcript(stream_line(tool_use_event("Grep")))

    assert (reading.answer, reading.turns, reading.tool_calls) == (None, 0, {"Grep": 1})


def record(
    *, right: bool, turns: int, cache_read: int, tool_calls: dict[str, int], cost: float
) -> SessionRecord:
    return SessionRecord(
        corpus="click",
        question="show a progress bar",
        arm=AgentArm.with_search,
        repeat=0,
        answer=answer("a.py", "f"),
        file_right=True,
        answer_right=right,
        turns=turns,
        tool_calls=tool_calls,
        usage=SessionUsage(
            input_tokens=0,
            cache_read_tokens=cache_read,
            cache_creation_tokens=0,
            output_tokens=100,
            cost_usd=cost,
        ),
        duration_seconds=10.0,
    )


def test_an_arm_s_summary_counts_its_right_answers_searches_and_reads_and_averages_the_rest():
    sessions = [
        record(right=True, turns=2, cache_read=40000, tool_calls=SEARCH_TOOL_CALLS, cost=0.1),
        record(right=False, turns=4, cache_read=80000, tool_calls=GREP_AND_READ_CALLS, cost=0.2),
        record(right=True, turns=3, cache_read=60000, tool_calls=NO_TOOLS, cost=0.3),
    ]

    summary = summarize_arm(AgentArm.with_search, sessions)

    assert (
        summary.sessions,
        summary.right_answers,
        summary.right_files,
        summary.mean_turns,
        summary.mean_context_read_tokens,
        summary.sessions_that_searched,
        summary.read_calls,
        summary.cost_usd,
    ) == (3, 2, 3, 3.0, 60000, 1, 1, 0.6)
