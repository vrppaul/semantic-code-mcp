"""Tests for the harness: the scoring of search quality, and the edits of the speed run."""

from pathlib import Path

import pytest
from evals.corpus import Answer
from evals.quality import RankScores, first_rank, relative_hit, score
from evals.server import SearchHit
from evals.speed import edit_text

from semantic_code_mcp.app import build_chunker


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


PYTHON_SOURCE = '''"""A module."""


def first():
    return 1
'''
RUST_SOURCE = """/// A function.
fn first() -> u32 {
    1
}
"""
MARKDOWN_SOURCE = """# Guide

Intro.

## Install

Steps.
"""


@pytest.mark.parametrize(
    ("name", "source"),
    [
        pytest.param("module.py", PYTHON_SOURCE, id="python"),
        pytest.param("lib.rs", RUST_SOURCE, id="rust"),
        pytest.param("guide.md", MARKDOWN_SOURCE, id="markdown"),
    ],
)
def test_an_edit_of_the_speed_run_adds_one_chunk_and_changes_no_other(
    tmp_path: Path, name: str, source: str
):
    """So the search after it re-embeds one chunk once vectors are reused."""
    edited = tmp_path / name
    edited.write_text(source)
    before = build_chunker().chunk_file(tmp_path, Path(name))

    edited.write_text(source + edit_text(edited, 1))
    after = build_chunker().chunk_file(tmp_path, Path(name))

    assert [chunk.chunk_id for chunk in after][:-1] == [chunk.chunk_id for chunk in before]
