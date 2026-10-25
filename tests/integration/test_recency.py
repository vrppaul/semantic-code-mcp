"""Tests for the boost a recently edited file's hits get."""

import os
import time
from pathlib import Path

from semantic_code_mcp.domain.models import HIGHEST_SCORE, ChunkType, SearchResult
from semantic_code_mcp.services.search import (
    MAX_RECENCY_BOOST,
    ONE_WEEK_SECONDS,
    with_recency_boost,
)


def hit_in(file_path: Path, score: float) -> SearchResult:
    return SearchResult(
        file_path=str(file_path),
        line_start=1,
        line_end=1,
        content="x = 1",
        chunk_type=ChunkType.function,
        name="x",
        score=score,
    )


def test_a_file_edited_just_now_gets_nearly_the_whole_boost(tmp_path: Path):
    edited = tmp_path / "edited.py"
    edited.write_text("x = 1\n")
    now = time.time()
    os.utime(edited, (now, now))

    boosted = with_recency_boost(hit_in(edited, 0.5), now)

    assert boosted.score == 0.5 + MAX_RECENCY_BOOST


def test_the_boost_never_lifts_a_score_past_the_highest(tmp_path: Path):
    edited = tmp_path / "edited.py"
    edited.write_text("x = 1\n")
    now = time.time()
    os.utime(edited, (now, now))

    assert with_recency_boost(hit_in(edited, HIGHEST_SCORE), now).score == HIGHEST_SCORE


def test_a_file_older_than_a_week_gets_none(tmp_path: Path):
    old = tmp_path / "old.py"
    old.write_text("x = 1\n")
    now = time.time()
    os.utime(old, (now - 2 * ONE_WEEK_SECONDS, now - 2 * ONE_WEEK_SECONDS))

    assert with_recency_boost(hit_in(old, 0.5), now).score == 0.5
