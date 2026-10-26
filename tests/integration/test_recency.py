"""Tests for the boost a recently edited file's hits get."""

import os
import time
from pathlib import Path

from semantic_code_mcp.domain.models import (
    HIGHEST_SCORE,
    Chunk,
    ChunkKind,
    Language,
    SearchResult,
)
from semantic_code_mcp.services.search import (
    MAX_RECENCY_BOOST,
    ONE_WEEK_SECONDS,
    with_recency_boost,
)


def hit_in(path: str, score: float) -> SearchResult:
    chunk = Chunk(
        chunk_id=path,
        path=Path(path),
        language=Language.python,
        kind=ChunkKind.function,
        name="x",
        parent=None,
        line_start=1,
        line_end=1,
        content="x = 1",
    )
    return SearchResult(chunk=chunk, score=score)


def test_a_file_edited_just_now_gets_nearly_the_whole_boost(tmp_path: Path):
    edited = tmp_path / "edited.py"
    edited.write_text("x = 1\n")
    now = time.time()
    os.utime(edited, (now, now))

    boosted = with_recency_boost(hit_in("edited.py", 0.5), tmp_path, now)

    assert boosted.score == 0.5 + MAX_RECENCY_BOOST


def test_the_boost_never_lifts_a_score_past_the_highest(tmp_path: Path):
    edited = tmp_path / "edited.py"
    edited.write_text("x = 1\n")
    now = time.time()
    os.utime(edited, (now, now))

    assert (
        with_recency_boost(hit_in("edited.py", HIGHEST_SCORE), tmp_path, now).score == HIGHEST_SCORE
    )


def test_a_file_older_than_a_week_gets_none(tmp_path: Path):
    old = tmp_path / "old.py"
    old.write_text("x = 1\n")
    now = time.time()
    os.utime(old, (now - 2 * ONE_WEEK_SECONDS, now - 2 * ONE_WEEK_SECONDS))

    assert with_recency_boost(hit_in("old.py", 0.5), tmp_path, now).score == 0.5
