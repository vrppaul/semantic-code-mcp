"""Tests for the scoring of the search-quality harness."""

from pathlib import Path

from evals.corpus import Answer
from evals.quality import RankScores, first_rank, relative_hit, score
from evals.server import SearchHit


def test_first_rank_counts_from_one():
    assert first_rank([False, True, True]) == 2


def test_first_rank_is_none_when_nothing_matched():
    assert first_rank([False, False]) is None


def test_score_counts_hits_and_averages_reciprocal_ranks_over_every_question():
    scores = score([1, 3, None, 2])

    assert scores == RankScores(hits_at_1=1, hits_at_3=3, mean_reciprocal_rank=0.458)


def test_relative_hit_names_the_file_from_the_corpus_root(tmp_path: Path):
    hit = SearchHit(
        file_path=str(tmp_path / "src" / "walk.rs"),
        name="scan",
        kind="function",
        line_start=1,
        line_end=9,
    )

    assert relative_hit(hit, tmp_path) == Answer(path="src/walk.rs", symbol="scan")
