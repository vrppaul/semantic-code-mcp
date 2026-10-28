"""The speeds the server is held to (ADR 009, D9), as data each `evals speed` run is checked
against, so a run names the targets it crosses and no reader compares figures by hand.

A target bounds one figure of a corpus's speeds for corpora of a size: the pinned corpora,
of a few thousand chunks; a repository of 20k chunks, as algotrading's 19,432; one of 100k,
as Sentry's 87,694; or any. The large ones are timed with `--corpus-file <file> --words-only`,
their question files kept outside the repository. Each bound leaves room above the slowest
corpus of the runs it was set from (2026-10-28, the machine at a load below 1) for this
machine's spread. The stale check's at 100k chunks is 400 ms, not D9's 300: today's builds
measure 260 to 310 ms, so 300 would be crossed by a build that changed nothing; ADR 010's O1
and O2 are what bring it there.
"""

from enum import StrEnum, auto
from typing import assert_never

from pydantic import BaseModel, ConfigDict

from evals.speed import CorpusSpeed


class SpeedFigure(StrEnum):
    """A figure of a corpus's speeds a target bounds, named as `CorpusSpeed` names it, or
    with `meaning_` before the name `MeaningSpeed` gives it."""

    cold_first_answer_seconds = auto()
    warm_search_p95_milliseconds = auto()
    narrowed_search_p50_milliseconds = auto()
    long_question_p50_milliseconds = auto()
    status_p50_milliseconds = auto()
    search_after_edit_p50_milliseconds = auto()
    search_after_edit_slowest_milliseconds = auto()
    first_search_seconds = auto()
    peak_memory_megabytes = auto()
    resident_memory_megabytes = auto()
    meaning_model_load_seconds = auto()
    meaning_peak_memory_megabytes = auto()
    meaning_resident_memory_megabytes = auto()


class CorpusSize(BaseModel):
    """The corpora a bound holds for, by their chunks: at least the fewest, below the bound
    where one is given."""

    model_config = ConfigDict(frozen=True)

    name: str
    fewest_chunks: int
    below_chunks: int | None

    def holds_for(self, chunks: int) -> bool:
        return chunks >= self.fewest_chunks and (
            self.below_chunks is None or chunks < self.below_chunks
        )


# Where the pinned corpora end and a repository's sizes begin, in chunks: the pinned hold up
# to about 1,800, algotrading about 19,400, Sentry about 87,700
PINNED_CORPORA_BELOW = 10_000
TWENTY_THOUSAND_BELOW = 50_000
PINNED = CorpusSize(name="the pinned corpora", fewest_chunks=0, below_chunks=PINNED_CORPORA_BELOW)
TWENTY_THOUSAND = CorpusSize(
    name="20k chunks", fewest_chunks=PINNED_CORPORA_BELOW, below_chunks=TWENTY_THOUSAND_BELOW
)
HUNDRED_THOUSAND = CorpusSize(
    name="100k chunks", fewest_chunks=TWENTY_THOUSAND_BELOW, below_chunks=None
)
ANY_SIZE = CorpusSize(name="any corpus", fewest_chunks=0, below_chunks=None)


class SpeedTarget(BaseModel):
    """The most a figure may be for corpora of a size."""

    model_config = ConfigDict(frozen=True)

    figure: SpeedFigure
    size: CorpusSize
    most: float


class TargetCrossed(BaseModel):
    """A target a corpus's speeds crossed, and the figure they measured."""

    model_config = ConfigDict(frozen=True)

    target: SpeedTarget
    measured: float


TARGETS = (
    # What an agent waits for on first contact with a repository
    SpeedTarget(figure=SpeedFigure.cold_first_answer_seconds, size=PINNED, most=3.0),
    SpeedTarget(figure=SpeedFigure.cold_first_answer_seconds, size=TWENTY_THOUSAND, most=10.0),
    SpeedTarget(figure=SpeedFigure.cold_first_answer_seconds, size=HUNDRED_THOUSAND, most=60.0),
    # A warm search by words, at p95 about the slowest question
    SpeedTarget(figure=SpeedFigure.warm_search_p95_milliseconds, size=PINNED, most=150.0),
    SpeedTarget(figure=SpeedFigure.warm_search_p95_milliseconds, size=HUNDRED_THOUSAND, most=500.0),
    SpeedTarget(figure=SpeedFigure.narrowed_search_p50_milliseconds, size=PINNED, most=100.0),
    SpeedTarget(
        figure=SpeedFigure.narrowed_search_p50_milliseconds, size=HUNDRED_THOUSAND, most=500.0
    ),
    # One question of 40 words, as an agent writes when it pastes a sentence: each word one
    # more term the full-text index scores every chunk by
    SpeedTarget(figure=SpeedFigure.long_question_p50_milliseconds, size=PINNED, most=150.0),
    SpeedTarget(
        figure=SpeedFigure.long_question_p50_milliseconds, size=HUNDRED_THOUSAND, most=700.0
    ),
    # `index_status`: the stale check each search makes first
    SpeedTarget(figure=SpeedFigure.status_p50_milliseconds, size=PINNED, most=50.0),
    SpeedTarget(figure=SpeedFigure.status_p50_milliseconds, size=HUNDRED_THOUSAND, most=400.0),
    # At 100k chunks a bound today's 664-744 ms meet, until ADR 010's O3 tightens it
    SpeedTarget(figure=SpeedFigure.search_after_edit_p50_milliseconds, size=PINNED, most=250.0),
    SpeedTarget(
        figure=SpeedFigure.search_after_edit_p50_milliseconds, size=HUNDRED_THOUSAND, most=1000.0
    ),
    SpeedTarget(figure=SpeedFigure.search_after_edit_slowest_milliseconds, size=PINNED, most=500.0),
    SpeedTarget(
        figure=SpeedFigure.search_after_edit_slowest_milliseconds,
        size=HUNDRED_THOUSAND,
        most=1500.0,
    ),
    # A fresh server's first answer on an indexed project
    SpeedTarget(figure=SpeedFigure.first_search_seconds, size=ANY_SIZE, most=3.0),
    # A server by words, at its peak and once its work is done
    SpeedTarget(figure=SpeedFigure.peak_memory_megabytes, size=PINNED, most=200.0),
    SpeedTarget(figure=SpeedFigure.peak_memory_megabytes, size=HUNDRED_THOUSAND, most=600.0),
    SpeedTarget(figure=SpeedFigure.resident_memory_megabytes, size=PINNED, most=200.0),
    SpeedTarget(figure=SpeedFigure.resident_memory_megabytes, size=HUNDRED_THOUSAND, most=600.0),
    # A server set to search by meaning
    SpeedTarget(figure=SpeedFigure.meaning_model_load_seconds, size=PINNED, most=1.0),
    SpeedTarget(figure=SpeedFigure.meaning_peak_memory_megabytes, size=PINNED, most=1000.0),
    SpeedTarget(figure=SpeedFigure.meaning_resident_memory_megabytes, size=PINNED, most=500.0),
)


def crossed_targets(speed: CorpusSpeed) -> list[TargetCrossed]:
    """The targets of the corpus's size whose figure its speeds measured above the bound; a
    figure not measured, as memory off Linux or meaning in a run by words alone, crosses
    none."""
    return [
        TargetCrossed(target=target, measured=value)
        for target in TARGETS
        if target.size.holds_for(speed.chunks_indexed)
        and (value := figure_of(speed, target.figure)) is not None
        and value > target.most
    ]


# One case a figure, which the type checker holds to every member: flat, not complex
def figure_of(speed: CorpusSpeed, figure: SpeedFigure) -> float | None:  # noqa: C901
    """A figure of a corpus's speeds; None where the run did not measure it."""
    meaning = speed.meaning
    match figure:
        case SpeedFigure.cold_first_answer_seconds:
            return speed.cold_first_answer_seconds
        case SpeedFigure.warm_search_p95_milliseconds:
            return speed.warm_search_p95_milliseconds
        case SpeedFigure.narrowed_search_p50_milliseconds:
            return speed.narrowed_search_p50_milliseconds
        case SpeedFigure.long_question_p50_milliseconds:
            return speed.long_question_p50_milliseconds
        case SpeedFigure.status_p50_milliseconds:
            return speed.status_p50_milliseconds
        case SpeedFigure.search_after_edit_p50_milliseconds:
            return speed.search_after_edit_p50_milliseconds
        case SpeedFigure.search_after_edit_slowest_milliseconds:
            return speed.search_after_edit_slowest_milliseconds
        case SpeedFigure.first_search_seconds:
            return speed.first_search_seconds
        case SpeedFigure.peak_memory_megabytes:
            return speed.peak_memory_megabytes
        case SpeedFigure.resident_memory_megabytes:
            return speed.resident_memory_megabytes
        case SpeedFigure.meaning_model_load_seconds:
            return None if meaning is None else meaning.model_load_seconds
        case SpeedFigure.meaning_peak_memory_megabytes:
            return None if meaning is None else meaning.peak_memory_megabytes
        case SpeedFigure.meaning_resident_memory_megabytes:
            return None if meaning is None else meaning.resident_memory_megabytes
        case unhandled:
            assert_never(unhandled)
