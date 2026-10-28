"""A run's sessions summed by arm, and every two arms compared question by question (ADR 008,
D4 and D5).

An agent's path through a question varies from session to session, and from question to
question far more, so two arms are compared on the questions both asked: each question's
figure is the mean of its sessions in an arm, and the difference is given as its mean over
the questions, the 95% interval of that mean by a paired bootstrap over them, and how many
questions it went down, stayed or went up on. The means hold the sessions that closed,
answered or not: one that ran out of time or failed has no turns or prompt to count, and is
counted apart. A session's context is its whole prompt, read from the cache, written to it or
neither, as what one session reads from the cache another wrote. Cost is never compared:
most of what Claude Code reports is writing to the cache, which the order of the sessions
decides.
"""

from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from itertools import combinations
from statistics import mean

import numpy
from pydantic import BaseModel, ConfigDict

from evals.agent_answers import RIGHT_VERDICTS, VERDICTS_TO_JUDGE, AnswerVerdict
from evals.agents import (
    CLOSED_OUTCOMES,
    READ_TOOL,
    SEARCH_TOOL,
    AgentsRun,
    ScoredSession,
    SessionOutcome,
    SessionRecord,
)

# Arms are compared two at a time
PAIR = 2
# The paired bootstrap's resamples, and its seed, fixed so a run scored again reports the
# same intervals
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261028
# The ends of a 95% interval
INTERVAL_QUANTILES = (0.025, 0.975)
# A question on which two arms' figures are the same
NO_DIFFERENCE = 0.0
MEAN_DECIMALS = 1
DIFFERENCE_DECIMALS = 1
COST_DECIMALS = 2


class ArmSummary(BaseModel):
    """The sessions of one arm: how they ended, what they answered, and what those that
    closed took on average; a mean is None where none closed."""

    model_config = ConfigDict(frozen=True)

    arm: str
    sessions: int
    outcomes: dict[SessionOutcome, int]
    right_answers: int
    # Of the right answers, those naming a member of a keyed chunk or the chunk a keyed one
    # sits in
    answers_as_members: int
    right_files: int
    answers_to_judge: int
    mean_turns: float | None
    mean_prompt_tokens: int | None
    mean_output_tokens: int | None
    mean_duration_seconds: float | None
    sessions_that_searched: int
    read_calls: int


class PairedDifference(BaseModel):
    """How an arm's figure differs from another's over the questions both closed sessions of:
    the mean of each question's difference, its 95% interval, and on how many questions the
    arm's figure was lower, the same and higher."""

    model_config = ConfigDict(frozen=True)

    mean: float
    low: float
    high: float
    fewer: int
    equal: int
    more: int


class ArmComparison(BaseModel):
    """An arm against an earlier arm of its run, in turns and in prompt tokens, over the
    questions both closed sessions of."""

    model_config = ConfigDict(frozen=True)

    arm: str
    against: str
    questions: int
    turns: PairedDifference
    prompt_tokens: PairedDifference


class AnswerGiven(BaseModel):
    """An answer to a question as sessions gave it, and how the chunks it names met the
    keys."""

    model_config = ConfigDict(frozen=True)

    question: str
    path: str
    symbol: str
    verdict: AnswerVerdict
    file_right: bool


class AnswerToJudge(BaseModel):
    """An answer no key settles, with the arms of every session that gave it: for a judge to
    put into the corpus's keys or leave out."""

    model_config = ConfigDict(frozen=True)

    answer: AnswerGiven
    arms: tuple[str, ...]


class CorpusAgents(BaseModel):
    """Every session asked of one corpus, each arm's summary, the arms compared, and the
    answers left to judge."""

    model_config = ConfigDict(frozen=True)

    corpus: str
    arms: tuple[ArmSummary, ...]
    comparisons: tuple[ArmComparison, ...]
    to_judge: tuple[AnswerToJudge, ...]
    sessions: tuple[ScoredSession, ...]


class AgentsReport(BaseModel):
    """A run's report: what it asked, what its sessions ran on as their init said, what
    Claude Code reports they cost, and each corpus's sessions."""

    model_config = ConfigDict(frozen=True)

    run: str
    # The model as the run asked for it, an alias Claude Code resolves
    model_asked: str
    models: tuple[str, ...]
    claude_code_versions: tuple[str, ...]
    sessions: int
    # To watch what the subscription spends; never compared
    reported_cost_usd: float
    corpora: tuple[CorpusAgents, ...]


def agents_report(run: AgentsRun, corpora: Sequence[CorpusAgents]) -> AgentsReport:
    """A run's report over the corpora it asked."""
    sessions = [scored.session for corpus in corpora for scored in corpus.sessions]
    return AgentsReport(
        run=run.name,
        model_asked=run.model,
        models=tuple(sorted({session.model for session in sessions if session.model})),
        claude_code_versions=tuple(
            sorted(
                {session.claude_code_version for session in sessions if session.claude_code_version}
            )
        ),
        sessions=len(sessions),
        reported_cost_usd=round(sum(session.usage.cost_usd for session in sessions), COST_DECIMALS),
        corpora=tuple(corpora),
    )


def summarize_corpus(
    corpus: str, arms: Sequence[str], sessions: Sequence[ScoredSession]
) -> CorpusAgents:
    """One corpus's sessions: each arm summed, each arm compared with every arm before it, and
    the answers left to judge."""
    compared = [
        compare_arms(later, earlier, sessions) for earlier, later in combinations(arms, PAIR)
    ]
    return CorpusAgents(
        corpus=corpus,
        arms=tuple(summarize_arm(arm, sessions) for arm in arms),
        comparisons=tuple(comparison for comparison in compared if comparison is not None),
        to_judge=tuple(answers_to_judge(sessions)),
        sessions=tuple(sessions),
    )


def summarize_arm(arm: str, sessions: Sequence[ScoredSession]) -> ArmSummary:
    """One arm's sessions among a corpus's: how they ended, what they answered, and the means
    of those that closed."""
    own = [scored for scored in sessions if scored.session.arm == arm]
    closed = [scored.session for scored in own if scored.session.outcome in CLOSED_OUTCOMES]
    scores = [scored.score for scored in own if scored.score is not None]
    return ArmSummary(
        arm=arm,
        sessions=len(own),
        outcomes=dict(Counter(scored.session.outcome for scored in own)),
        right_answers=sum(score.verdict in RIGHT_VERDICTS for score in scores),
        answers_as_members=sum(score.verdict == AnswerVerdict.member for score in scores),
        right_files=sum(score.file_right for score in scores),
        answers_to_judge=sum(score.verdict in VERDICTS_TO_JUDGE for score in scores),
        mean_turns=rounded_mean([session.turns for session in closed]),
        mean_prompt_tokens=whole_mean([session.usage.prompt_tokens for session in closed]),
        mean_output_tokens=whole_mean([session.usage.output_tokens for session in closed]),
        mean_duration_seconds=rounded_mean([session.duration_seconds for session in closed]),
        sessions_that_searched=sum(SEARCH_TOOL in scored.session.tool_calls for scored in own),
        read_calls=sum(scored.session.tool_calls.get(READ_TOOL, 0) for scored in own),
    )


def rounded_mean(values: Sequence[float]) -> float | None:
    return round(mean(values), MEAN_DECIMALS) if values else None


def whole_mean(values: Sequence[int]) -> int | None:
    return round(mean(values)) if values else None


def compare_arms(arm: str, against: str, sessions: Sequence[ScoredSession]) -> ArmComparison | None:
    """An arm against another over the questions both closed sessions of; None where they
    share none."""
    turns = figure_differences(sessions, arm, against, turns_of)
    if not turns:
        return None
    return ArmComparison(
        arm=arm,
        against=against,
        questions=len(turns),
        turns=paired_difference(turns),
        prompt_tokens=paired_difference(
            figure_differences(sessions, arm, against, prompt_tokens_of)
        ),
    )


def turns_of(session: SessionRecord) -> float:
    return session.turns


def prompt_tokens_of(session: SessionRecord) -> float:
    return session.usage.prompt_tokens


def figure_differences(
    sessions: Sequence[ScoredSession],
    arm: str,
    against: str,
    figure: Callable[[SessionRecord], float],
) -> list[float]:
    """Each question both arms closed sessions of: the arm's mean figure less the other's."""
    ours = question_means(sessions, arm, figure)
    theirs = question_means(sessions, against, figure)
    return [ours[question] - theirs[question] for question in ours if question in theirs]


def question_means(
    sessions: Sequence[ScoredSession], arm: str, figure: Callable[[SessionRecord], float]
) -> dict[str, float]:
    """Each question's mean figure over an arm's sessions of it that closed."""
    by_question: defaultdict[str, list[float]] = defaultdict(list)
    for scored in sessions:
        session = scored.session
        if session.arm == arm and session.outcome in CLOSED_OUTCOMES:
            by_question[session.question].append(figure(session))
    return {question: mean(figures) for question, figures in by_question.items()}


def paired_difference(differences: Sequence[float]) -> PairedDifference:
    """The mean of the questions' differences, its interval by a paired bootstrap over the
    questions, and how many went each way."""
    values = numpy.asarray(differences, dtype=float)
    generator = numpy.random.default_rng(BOOTSTRAP_SEED)
    resampled_means = generator.choice(values, size=(BOOTSTRAP_RESAMPLES, values.size)).mean(axis=1)
    low, high = numpy.quantile(resampled_means, INTERVAL_QUANTILES)
    return PairedDifference(
        mean=round(float(values.mean()), DIFFERENCE_DECIMALS),
        low=round(float(low), DIFFERENCE_DECIMALS),
        high=round(float(high), DIFFERENCE_DECIMALS),
        fewer=int((values < NO_DIFFERENCE).sum()),
        equal=int((values == NO_DIFFERENCE).sum()),
        more=int((values > NO_DIFFERENCE).sum()),
    )


def answers_to_judge(sessions: Sequence[ScoredSession]) -> list[AnswerToJudge]:
    """The answers no key settles, each once, in the order first given, with the arms of the
    sessions that gave it."""
    arms_by_answer: defaultdict[AnswerGiven, list[str]] = defaultdict(list)
    for scored in sessions:
        session = scored.session
        score = scored.score
        if score is None or session.answer is None or score.verdict not in VERDICTS_TO_JUDGE:
            continue
        given = AnswerGiven(
            question=session.question,
            path=score.path,
            symbol=session.answer.symbol,
            verdict=score.verdict,
            file_right=score.file_right,
        )
        arms_by_answer[given].append(session.arm)
    return [AnswerToJudge(answer=given, arms=tuple(arms)) for given, arms in arms_by_answer.items()]
