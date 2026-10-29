"""Tests for search judged by agents: how a session is started, how its transcript is read,
how it ended, and whether it leaves the run to be trusted."""

import json
import os
import sys
from pathlib import Path

import pytest
from evals.agent_answers import AnswerScorer, SessionAnswer
from evals.agents import (
    AGENT_ARMS,
    CONFIG_DIRECTORY_VARIABLE,
    OWN_TOOLS_ARM,
    SEARCH_TOOL,
    AgentArm,
    CorpusNameSharedError,
    QuestionNotAskedError,
    SessionClosing,
    SessionOutcome,
    SessionRecord,
    SessionRun,
    SessionStart,
    SessionTurn,
    SessionUsage,
    answer_of,
    arms_in_turn,
    mcp_config,
    outcome_of,
    read_transcript,
    refuse_shared_names,
    rescored,
    score_session,
    session_environment,
    session_record,
    stop_reason,
)
from evals.corpus import Answer, Corpus, CorpusSource, Query, QuestionKind
from evals.server import CACHE_DIR_VARIABLE, SEARCH_MODE_VARIABLE

from semantic_code_mcp.domain.models import SearchMode
from tests.helpers import POSIX_HARNESS, WINDOWS

WORDS_ARM = AGENT_ARMS[SearchMode.words]
MEANING_ARM = AGENT_ARMS[SearchMode.words_and_meaning]
INDEX_DIRECTORY = Path("/indexes/click-0123456789ab")
GREP_AND_READ_CALLS = {"Grep": 2, "Read": 1}
LOGIN = "none"
CONNECTED = "connected"
WORKING_TREE = Path("/checkouts/click")
QUERY = Query(
    text="show a progress bar",
    kind=QuestionKind.code,
    answers=(Answer(path="a.py", symbol="f"),),
)


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


def stream_line(event: dict) -> str:
    return json.dumps(event)


def init_event(api_key_source: str, server_status: str) -> dict:
    return {
        "type": "system",
        "subtype": "init",
        "model": "claude-sonnet-5-5",
        "claude_code_version": "2.1.312",
        "apiKeySource": api_key_source,
        "mcp_servers": [{"name": "semantic-code", "status": server_status, "source": "dynamic"}],
    }


def tool_use_event(*names: str) -> dict:
    content = [{"type": "tool_use", "name": name, "input": {}} for name in names]
    return {"type": "assistant", "message": {"content": content}}


def result_event(reply: str, *, is_error: bool) -> dict:
    return {
        "type": "result",
        "subtype": "success",
        "is_error": is_error,
        "num_turns": 3,
        "result": reply,
        "total_cost_usd": 0.05,
        "usage": {
            "input_tokens": 4,
            "cache_read_input_tokens": 30000,
            "cache_creation_input_tokens": 9000,
            "output_tokens": 250,
        },
    }


ANSWERED_REPLY = 'Found.\n{"path": "a.py", "symbol": "f"}'
TRANSCRIPT = "\n".join(
    [
        stream_line(init_event(LOGIN, CONNECTED)),
        stream_line(tool_use_event("Grep")),
        "not a line of JSON",
        stream_line(tool_use_event("Grep", "Read")),
        stream_line(result_event(ANSWERED_REPLY, is_error=False)),
    ]
)


def test_a_transcript_gives_what_the_session_ran_on_and_its_answer_turns_tool_calls_and_tokens():
    reading = read_transcript(TRANSCRIPT)

    assert (reading.start, reading.closing, reading.tool_calls) == (
        SessionStart(
            model="claude-sonnet-5-5",
            claude_code_version="2.1.312",
            api_key_source=LOGIN,
            servers={"semantic-code": CONNECTED},
        ),
        SessionClosing(
            is_error=False,
            answer=answer("a.py", "f"),
            turns=3,
            usage=SessionUsage(
                input_tokens=4,
                cache_read_tokens=30000,
                cache_creation_tokens=9000,
                output_tokens=250,
                cost_usd=0.05,
            ),
        ),
        GREP_AND_READ_CALLS,
    )


def test_a_session_s_prompt_is_every_token_it_sent_read_from_the_cache_or_written_to_it():
    usage = SessionUsage(
        input_tokens=4,
        cache_read_tokens=30000,
        cache_creation_tokens=9000,
        output_tokens=250,
        cost_usd=0.05,
    )

    assert usage.prompt_tokens == 39004


def session_run(stream: str, exit_code: int | None) -> SessionRun:
    return SessionRun(output=stream, exit_code=exit_code, duration_seconds=10.0)


def started(*events: dict) -> str:
    """A stream that opens with the init of a session on the login, its server connected."""
    return "\n".join(stream_line(event) for event in (init_event(LOGIN, CONNECTED), *events))


@pytest.mark.parametrize(
    ("stream", "exit_code", "arm", "outcome"),
    [
        pytest.param(TRANSCRIPT, 0, WORDS_ARM, SessionOutcome.answered, id="answered"),
        pytest.param(
            started(result_event("I could not find it.", is_error=False)),
            0,
            WORDS_ARM,
            SessionOutcome.no_answer,
            id="closed without an answer",
        ),
        pytest.param(
            started(result_event(ANSWERED_REPLY, is_error=True)),
            0,
            WORDS_ARM,
            SessionOutcome.failed,
            id="an error result",
        ),
        pytest.param(TRANSCRIPT, 1, WORDS_ARM, SessionOutcome.failed, id="an exit other than 0"),
        pytest.param(
            started(tool_use_event("Grep")), 0, WORDS_ARM, SessionOutcome.failed, id="no result"
        ),
        pytest.param(
            stream_line(result_event(ANSWERED_REPLY, is_error=False)),
            0,
            WORDS_ARM,
            SessionOutcome.failed,
            id="no init",
        ),
        pytest.param(
            started(tool_use_event("Grep")),
            None,
            WORDS_ARM,
            SessionOutcome.timed_out,
            id="stopped at its timeout",
        ),
        pytest.param(
            "\n".join(
                [
                    stream_line(init_event("ANTHROPIC_API_KEY", CONNECTED)),
                    stream_line(result_event(ANSWERED_REPLY, is_error=False)),
                ]
            ),
            0,
            WORDS_ARM,
            SessionOutcome.billed_elsewhere,
            id="billed to an API key",
        ),
        pytest.param(
            "\n".join(
                [
                    stream_line(init_event(LOGIN, "failed")),
                    stream_line(result_event(ANSWERED_REPLY, is_error=False)),
                ]
            ),
            0,
            WORDS_ARM,
            SessionOutcome.server_not_connected,
            id="its server down",
        ),
        pytest.param(
            "\n".join(
                [
                    stream_line(init_event(LOGIN, "failed")),
                    stream_line(result_event(ANSWERED_REPLY, is_error=False)),
                ]
            ),
            0,
            OWN_TOOLS_ARM,
            SessionOutcome.answered,
            id="own tools, whatever a server did",
        ),
    ],
)
def test_a_session_s_outcome_is_what_its_init_its_result_and_its_exit_say(
    stream: str, exit_code: int | None, arm: AgentArm, outcome: SessionOutcome
):
    run = session_run(stream, exit_code)

    assert outcome_of(run, read_transcript(stream), arm) == outcome


def record(
    arm: AgentArm, outcome: SessionOutcome, api_key_source: str | None, server_status: str | None
) -> SessionRecord:
    return SessionRecord(
        corpus="click",
        question="show a progress bar",
        arm=arm.name,
        repeat=0,
        position=0,
        outcome=outcome,
        answer=answer("a.py", "f"),
        turns=3,
        tool_calls={SEARCH_TOOL: 1},
        usage=SessionUsage(
            input_tokens=4,
            cache_read_tokens=30000,
            cache_creation_tokens=9000,
            output_tokens=250,
            cost_usd=0.05,
        ),
        duration_seconds=10.0,
        model="claude-sonnet-5-5",
        claude_code_version="2.1.312",
        api_key_source=api_key_source,
        server_status=server_status,
        stream_file="click-r0-q00-words.jsonl",
    )


@pytest.mark.parametrize(
    ("outcome", "stops"),
    [
        pytest.param(SessionOutcome.answered, False, id="answered"),
        pytest.param(SessionOutcome.no_answer, False, id="no answer"),
        pytest.param(SessionOutcome.timed_out, False, id="timed out, the agent's own"),
        pytest.param(SessionOutcome.billed_elsewhere, True, id="billed to an API key"),
        pytest.param(SessionOutcome.server_not_connected, True, id="its server down"),
        pytest.param(SessionOutcome.failed, True, id="failed"),
    ],
)
def test_a_run_stops_on_a_session_billed_elsewhere_without_its_server_or_failed(
    outcome: SessionOutcome, stops: bool
):
    session = record(WORDS_ARM, outcome, LOGIN, CONNECTED)

    assert (stop_reason(session) is not None) is stops


def test_a_session_s_environment_holds_only_what_claude_code_needs_to_run_on_the_login(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """An API key would bill an account, the caller's effort and thinking budget and the
    shell its home sets up would change how the agent works, and the server's settings would
    reach the arms' servers."""
    caller = {
        "PATH": "/usr/bin",
        "HOME": "/home/owner",
        "ANTHROPIC_API_KEY": "a key",
        "CLAUDE_EFFORT": "xhigh",
        "MAX_THINKING_TOKENS": "128000",
        "SEMANTIC_CODE_MCP_INDEX_STORE": "lancedb",
    }
    monkeypatch.setattr(os, "environ", caller)

    environment = session_environment(tmp_path)

    assert environment == {
        "PATH": "/usr/bin",
        "HOME": str(tmp_path),
        CONFIG_DIRECTORY_VARIABLE: str(tmp_path / ".claude"),
    }


@pytest.mark.parametrize(
    ("arm", "settings_by_server"),
    [
        pytest.param(OWN_TOOLS_ARM, {}, id="own tools, no server"),
        pytest.param(
            WORDS_ARM,
            {
                "semantic-code": {
                    CACHE_DIR_VARIABLE: str(INDEX_DIRECTORY),
                    SEARCH_MODE_VARIABLE: SearchMode.words,
                    "HOME": "/home/owner",
                }
            },
            id="words",
        ),
        pytest.param(
            MEANING_ARM,
            {
                "semantic-code": {
                    CACHE_DIR_VARIABLE: str(INDEX_DIRECTORY),
                    SEARCH_MODE_VARIABLE: SearchMode.words_and_meaning,
                    "HOME": "/home/owner",
                }
            },
            id="words and meaning",
        ),
    ],
)
@pytest.mark.skipif(sys.platform == WINDOWS, reason=POSIX_HARNESS)
def test_an_arm_s_server_searches_the_kept_index_in_the_mode_the_arm_is_named_by(
    monkeypatch: pytest.MonkeyPatch, arm: AgentArm, settings_by_server: dict[str, dict[str, str]]
):
    """In the owner's home, as a user's server runs in the user's, with the model's weights
    cached there."""
    monkeypatch.setenv("HOME", "/home/owner")

    servers = mcp_config(arm, INDEX_DIRECTORY).mcp_servers

    assert {name: server.env for name, server in servers.items()} == settings_by_server


@pytest.mark.parametrize(
    ("question_number", "repeat", "names"),
    [
        pytest.param(0, 0, ["own_tools", "words", "words_and_meaning"], id="first question"),
        pytest.param(1, 0, ["words", "words_and_meaning", "own_tools"], id="the next question"),
        pytest.param(0, 1, ["words", "words_and_meaning", "own_tools"], id="the next repeat"),
        pytest.param(2, 1, ["own_tools", "words", "words_and_meaning"], id="round again"),
    ],
)
def test_the_arms_take_turns_to_go_first_question_by_question_and_repeat_by_repeat(
    question_number: int, repeat: int, names: list[str]
):
    arms = (OWN_TOOLS_ARM, WORDS_ARM, MEANING_ARM)

    in_turn = arms_in_turn(arms, question_number, repeat)

    assert [arm.name for arm in in_turn] == names


def turn_of(arm: AgentArm) -> SessionTurn:
    return SessionTurn(question_number=0, query=QUERY, repeat=0, position=1, arm=arm)


def test_a_session_s_record_takes_what_it_ran_on_from_its_init_and_its_answer_from_its_result():
    session = session_record(
        "click", turn_of(WORDS_ARM), session_run(TRANSCRIPT, 0), "click-r0-q00-words.jsonl"
    )

    assert (
        session.outcome,
        session.answer,
        session.turns,
        session.model,
        session.claude_code_version,
        session.api_key_source,
        session.server_status,
        session.usage.prompt_tokens,
    ) == (
        SessionOutcome.answered,
        answer("a.py", "f"),
        3,
        "claude-sonnet-5-5",
        "2.1.312",
        LOGIN,
        CONNECTED,
        39004,
    )


def test_a_session_that_never_started_is_a_failed_one_with_nothing_it_ran_on():
    session = session_record(
        "click", turn_of(WORDS_ARM), session_run("", 1), "click-r0-q00-words.jsonl"
    )

    assert (session.outcome, session.turns, session.model, session.server_status) == (
        SessionOutcome.failed,
        0,
        None,
        None,
    )


@pytest.mark.parametrize(
    ("outcome", "scored"),
    [
        pytest.param(SessionOutcome.answered, True, id="answered"),
        pytest.param(SessionOutcome.failed, False, id="failed, whatever it said"),
    ],
)
def test_only_a_session_that_answered_is_scored(outcome: SessionOutcome, scored: bool):
    session = record(WORDS_ARM, outcome, LOGIN, CONNECTED)

    given = score_session(session, QUERY, AnswerScorer(WORKING_TREE))

    assert (given.score is not None) is scored


def corpus_of(name: str, *questions: str) -> Corpus:
    return Corpus(
        corpus=CorpusSource(name=name, repository="https://example.org/click", commit="0" * 40),
        queries=tuple(
            Query(text=question, kind=QuestionKind.code, answers=()) for question in questions
        ),
        held_out=(),
    )


def test_a_kept_run_is_never_scored_against_a_corpus_that_no_longer_asks_its_question():
    corpus = corpus_of("click", "echo to the terminal")

    with pytest.raises(QuestionNotAskedError):
        rescored(
            [record(WORDS_ARM, SessionOutcome.answered, LOGIN, CONNECTED)],
            corpus,
            AnswerScorer(WORKING_TREE),
        )


def test_a_run_never_asks_two_corpus_files_of_one_name():
    """It keeps and scores its sessions by the corpus's name, so the two would mix: their
    questions go into one file."""
    corpora = [corpus_of("click", "show a progress bar"), corpus_of("click", "echo a line")]

    with pytest.raises(CorpusNameSharedError):
        refuse_shared_names(corpora)
