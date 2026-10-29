"""Tests for the speed targets: which of them a corpus's speeds cross, at its size."""

from evals.speed import CorpusSpeed, MeaningSpeed
from evals.targets import SpeedFigure, crossed_targets

PINNED_CHUNKS = 1746
SENTRY_CHUNKS = 87694


def speeds(
    *,
    chunks: int,
    cold_first_answer_seconds: float,
    status_milliseconds: float,
    search_after_edit_milliseconds: float,
    peak_memory_megabytes: float,
    meaning: MeaningSpeed | None,
) -> CorpusSpeed:
    """A corpus's speeds, each within every target but the five given."""
    return CorpusSpeed(
        corpus="click",
        cold_first_answer_seconds=cold_first_answer_seconds,
        files_indexed=133,
        chunks_indexed=chunks,
        full_index_seconds=0.3,
        files_per_second=480.0,
        chunks_per_second=6000.0,
        warm_search_p50_milliseconds=17.0,
        warm_search_p95_milliseconds=27.0,
        narrowed_search_p50_milliseconds=17.0,
        long_question_p50_milliseconds=40.0,
        status_p50_milliseconds=status_milliseconds,
        index_growth_megabytes_per_edit=0.02,
        search_after_edit_p50_milliseconds=search_after_edit_milliseconds,
        search_after_edit_slowest_milliseconds=search_after_edit_milliseconds,
        peak_memory_megabytes=peak_memory_megabytes,
        resident_memory_megabytes=peak_memory_megabytes,
        meaning=meaning,
        first_search_seconds=0.8,
    )


def by_meaning(peak_memory_megabytes: float) -> MeaningSpeed:
    return MeaningSpeed(
        model_load_seconds=0.3,
        first_search_seconds=70.0,
        warm_search_p50_milliseconds=38.0,
        peak_memory_megabytes=peak_memory_megabytes,
        resident_memory_megabytes=300.0,
    )


def test_a_pinned_corpus_crosses_the_targets_its_figures_pass_and_no_larger_size_s():
    measured = speeds(
        chunks=PINNED_CHUNKS,
        cold_first_answer_seconds=1.0,
        status_milliseconds=60.0,
        search_after_edit_milliseconds=300.0,
        peak_memory_megabytes=700.0,
        meaning=by_meaning(peak_memory_megabytes=1200.0),
    )

    crossed = crossed_targets(measured)

    assert [(crossing.target.figure, crossing.target.most) for crossing in crossed] == [
        (SpeedFigure.status_p50_milliseconds, 50.0),
        (SpeedFigure.search_after_edit_p50_milliseconds, 250.0),
        (SpeedFigure.peak_memory_megabytes, 200.0),
        (SpeedFigure.resident_memory_megabytes, 200.0),
        (SpeedFigure.meaning_peak_memory_megabytes, 1000.0),
    ]


def test_a_repository_s_size_is_held_to_its_own_bounds_and_to_none_it_lacks():
    """Sentry's first answer is slower than the pinned corpora's bound allows, within its own;
    and a run by words alone has no figure by meaning to hold."""
    measured = speeds(
        chunks=SENTRY_CHUNKS,
        cold_first_answer_seconds=10.0,
        status_milliseconds=40.0,
        search_after_edit_milliseconds=200.0,
        peak_memory_megabytes=700.0,
        meaning=None,
    )

    crossed = crossed_targets(measured)

    assert [crossing.target.figure for crossing in crossed] == [
        SpeedFigure.peak_memory_megabytes,
        SpeedFigure.resident_memory_megabytes,
    ]
