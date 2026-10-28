"""Tests for a run's sessions summed by arm and compared question by question."""

from evals.agent_answers import AnswerScore, AnswerVerdict, SessionAnswer
from evals.agent_summary import (
    AnswerGiven,
    AnswerToJudge,
    PairedDifference,
    answers_to_judge,
    compare_arms,
    paired_difference,
    summarize_arm,
    summarize_corpus,
)
from evals.agents import SEARCH_TOOL, ScoredSession, SessionOutcome, SessionRecord, SessionUsage

SEARCHED = {SEARCH_TOOL: 1}
GREPPED_AND_READ = {"Grep": 2, "Read": 1}
GREPPED = {"Grep": 5}
NO_TOOLS: dict[str, int] = {}
FIRST_QUESTION = "show a progress bar"
SECOND_QUESTION = "ask the user to confirm"
THIRD_QUESTION = "open a file lazily"
FOURTH_QUESTION = "echo to the terminal"
FIFTH_QUESTION = "pick a color"


def scored(
    *,
    arm: str,
    question: str,
    outcome: SessionOutcome,
    score: AnswerScore | None,
    turns: int,
    prompt_tokens: int,
    tool_calls: dict[str, int],
) -> ScoredSession:
    session = SessionRecord(
        corpus="click",
        question=question,
        arm=arm,
        repeat=0,
        position=0,
        outcome=outcome,
        answer=SessionAnswer(path="a.py", symbol="f"),
        turns=turns,
        tool_calls=tool_calls,
        usage=SessionUsage(
            input_tokens=0,
            cache_read_tokens=prompt_tokens,
            cache_creation_tokens=0,
            output_tokens=100,
            cost_usd=0.05,
        ),
        duration_seconds=10.0,
        model="claude-sonnet-5-5",
        claude_code_version="2.1.312",
        api_key_source="none",
        server_status="connected",
        stream_file="click-r0-q00-words.jsonl",
    )
    return ScoredSession(session=session, score=score)


def score(verdict: AnswerVerdict, *, file_right: bool, path: str = "a.py") -> AnswerScore:
    return AnswerScore(path=path, file_right=file_right, verdict=verdict)


def test_an_arm_s_summary_counts_its_outcomes_and_answers_and_averages_the_sessions_that_closed():
    """A session that ran out of time has no turns or prompt to count."""
    sessions = [
        scored(
            arm="words",
            question=FIRST_QUESTION,
            outcome=SessionOutcome.answered,
            score=score(AnswerVerdict.right, file_right=True),
            turns=2,
            prompt_tokens=40000,
            tool_calls=SEARCHED,
        ),
        scored(
            arm="words",
            question=SECOND_QUESTION,
            outcome=SessionOutcome.answered,
            score=score(AnswerVerdict.member, file_right=True),
            turns=4,
            prompt_tokens=80000,
            tool_calls=GREPPED_AND_READ,
        ),
        scored(
            arm="words",
            question=THIRD_QUESTION,
            outcome=SessionOutcome.answered,
            score=score(AnswerVerdict.unresolved, file_right=False),
            turns=3,
            prompt_tokens=60000,
            tool_calls=NO_TOOLS,
        ),
        scored(
            arm="words",
            question=FOURTH_QUESTION,
            outcome=SessionOutcome.no_answer,
            score=None,
            turns=3,
            prompt_tokens=60000,
            tool_calls=NO_TOOLS,
        ),
        scored(
            arm="words",
            question=FIFTH_QUESTION,
            outcome=SessionOutcome.answered,
            score=score(AnswerVerdict.judged_right, file_right=False),
            turns=3,
            prompt_tokens=60000,
            tool_calls=NO_TOOLS,
        ),
        scored(
            arm="words",
            question=FOURTH_QUESTION,
            outcome=SessionOutcome.timed_out,
            score=None,
            turns=0,
            prompt_tokens=0,
            tool_calls=GREPPED,
        ),
    ]

    summary = summarize_arm("words", sessions)

    assert (
        summary.sessions,
        summary.outcomes,
        summary.right_answers,
        summary.answers_as_members,
        summary.answers_judged_right,
        summary.right_files,
        summary.answers_to_judge,
        summary.mean_turns,
        summary.mean_prompt_tokens,
        summary.sessions_that_searched,
        summary.read_calls,
    ) == (
        6,
        {SessionOutcome.answered: 4, SessionOutcome.no_answer: 1, SessionOutcome.timed_out: 1},
        3,
        1,
        1,
        2,
        1,
        3.0,
        60000,
        1,
        1,
    )


def closed(arm: str, question: str, turns: int, prompt_tokens: int) -> ScoredSession:
    return scored(
        arm=arm,
        question=question,
        outcome=SessionOutcome.answered,
        score=score(AnswerVerdict.right, file_right=True),
        turns=turns,
        prompt_tokens=prompt_tokens,
        tool_calls=SEARCHED,
    )


def test_two_arms_compare_on_the_questions_both_asked_each_the_mean_of_its_sessions_that_closed():
    sessions = [
        closed("words", FIRST_QUESTION, 2, 40000),
        closed("words", FIRST_QUESTION, 4, 40000),
        scored(
            arm="words",
            question=FIRST_QUESTION,
            outcome=SessionOutcome.timed_out,
            score=None,
            turns=0,
            prompt_tokens=0,
            tool_calls=GREPPED,
        ),
        closed("own_tools", FIRST_QUESTION, 5, 90000),
        closed("own_tools", FIRST_QUESTION, 5, 90000),
        closed("words", SECOND_QUESTION, 3, 50000),
        closed("own_tools", SECOND_QUESTION, 3, 50000),
        closed("words", THIRD_QUESTION, 6, 70000),
        closed("own_tools", THIRD_QUESTION, 4, 60000),
        closed("words", FOURTH_QUESTION, 2, 30000),
    ]

    comparison = compare_arms("words", "own_tools", sessions)

    assert (
        comparison.questions,
        comparison.sessions_left_out,
        comparison.turns.mean,
        (comparison.turns.fewer, comparison.turns.equal, comparison.turns.more),
        comparison.prompt_tokens.mean,
        (
            comparison.prompt_tokens.fewer,
            comparison.prompt_tokens.equal,
            comparison.prompt_tokens.more,
        ),
    ) == (3, 1, 0.0, (1, 1, 1), -13333.3, (1, 1, 1))


def test_a_difference_that_varies_has_an_interval_inside_its_range_around_its_mean():
    difference = paired_difference([-1.0, 0.0, 1.0, 2.0])

    assert -1.0 < difference.low < difference.mean < difference.high < 2.0


def test_a_difference_the_same_on_every_question_has_no_spread():
    assert paired_difference([-2.0, -2.0, -2.0]) == PairedDifference(
        mean=-2.0, low=-2.0, high=-2.0, fewer=3, equal=0, more=0
    )


def test_answers_no_key_settles_are_listed_once_each_with_the_arms_that_gave_them():
    unresolved = score(AnswerVerdict.unresolved, file_right=False, path="frontend/bar.ts")
    another_chunk = score(AnswerVerdict.other_chunk, file_right=True, path="src/termui.py")
    sessions = [
        scored(
            arm="words",
            question=FIRST_QUESTION,
            outcome=SessionOutcome.answered,
            score=unresolved,
            turns=2,
            prompt_tokens=40000,
            tool_calls=SEARCHED,
        ),
        scored(
            arm="own_tools",
            question=FIRST_QUESTION,
            outcome=SessionOutcome.answered,
            score=unresolved,
            turns=4,
            prompt_tokens=80000,
            tool_calls=GREPPED,
        ),
        scored(
            arm="words",
            question=FIRST_QUESTION,
            outcome=SessionOutcome.answered,
            score=score(AnswerVerdict.right, file_right=True),
            turns=2,
            prompt_tokens=40000,
            tool_calls=SEARCHED,
        ),
        scored(
            arm="words",
            question=SECOND_QUESTION,
            outcome=SessionOutcome.answered,
            score=another_chunk,
            turns=3,
            prompt_tokens=50000,
            tool_calls=SEARCHED,
        ),
        scored(
            arm="words",
            question=THIRD_QUESTION,
            outcome=SessionOutcome.no_answer,
            score=None,
            turns=3,
            prompt_tokens=50000,
            tool_calls=SEARCHED,
        ),
    ]

    assert answers_to_judge(sessions) == [
        AnswerToJudge(
            answer=AnswerGiven(
                question=FIRST_QUESTION,
                path="frontend/bar.ts",
                symbol="f",
                verdict=AnswerVerdict.unresolved,
                file_right=False,
            ),
            arms=("words", "own_tools"),
        ),
        AnswerToJudge(
            answer=AnswerGiven(
                question=SECOND_QUESTION,
                path="src/termui.py",
                symbol="f",
                verdict=AnswerVerdict.other_chunk,
                file_right=True,
            ),
            arms=("words",),
        ),
    ]


def test_each_arm_is_compared_with_every_arm_named_before_it():
    sessions = [
        closed(arm, FIRST_QUESTION, 3, 50000) for arm in ("own_tools", "words", "words_and_meaning")
    ]

    corpus = summarize_corpus("click", ["own_tools", "words", "words_and_meaning"], sessions)

    assert [(comparison.arm, comparison.against) for comparison in corpus.comparisons] == [
        ("words", "own_tools"),
        ("words_and_meaning", "own_tools"),
        ("words_and_meaning", "words"),
    ]
