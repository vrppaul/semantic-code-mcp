"""Tests for the harness: the scoring of search quality, and the edits of the speed run."""

from pathlib import Path

import pytest
from evals.corpus import Answer, QuestionKind
from evals.quality import (
    KindScores,
    QueryRanks,
    RankScores,
    first_rank,
    relative_hit,
    sample_names,
    score,
    scores_by_kind,
)
from evals.server import SearchHit
from evals.speed import edit_text

from semantic_code_mcp.app import build_chunker
from semantic_code_mcp.domain.models import Chunk, ChunkKind, Language


def test_first_rank_counts_from_one():
    assert first_rank([False, True, True]) == 2


def test_first_rank_is_none_when_nothing_matched():
    assert first_rank([False, False]) is None


def test_score_counts_hits_and_averages_reciprocal_ranks_over_every_question():
    scores = score([1, 3, None, 2])

    assert scores == RankScores(hits_at_1=1, hits_at_3=3, mean_reciprocal_rank=0.458)


def ranks_of(kind: QuestionKind, file_rank: int | None, chunk_rank: int | None) -> QueryRanks:
    return QueryRanks(
        query="a question", kind=kind, file_rank=file_rank, chunk_rank=chunk_rank, hits=()
    )


def test_scores_are_kept_apart_for_each_kind_of_question():
    ranks = [
        ranks_of(QuestionKind.code, 1, 2),
        ranks_of(QuestionKind.documentation, 4, None),
        ranks_of(QuestionKind.code, None, None),
    ]

    assert scores_by_kind(ranks) == (
        KindScores(
            kind=QuestionKind.code,
            queries=2,
            file=RankScores(hits_at_1=1, hits_at_3=1, mean_reciprocal_rank=0.5),
            chunk=RankScores(hits_at_1=0, hits_at_3=1, mean_reciprocal_rank=0.25),
        ),
        KindScores(
            kind=QuestionKind.documentation,
            queries=1,
            file=RankScores(hits_at_1=0, hits_at_3=0, mean_reciprocal_rank=0.25),
            chunk=RankScores(hits_at_1=0, hits_at_3=0, mean_reciprocal_rank=0.0),
        ),
    )


def defined(name: str, path: str, kind: ChunkKind = ChunkKind.function) -> Chunk:
    return Chunk(
        chunk_id=f"{path}:{name}",
        path=Path(path),
        language=Language.python,
        kind=kind,
        name=name,
        parent=None,
        line_start=1,
        line_end=1,
        content=f"def {name}(): ...",
    )


def test_a_bare_name_is_asked_of_every_fifth_name_one_file_defines():
    """A name defined in two files, or that is no identifier, has no one chunk to find;
    each such name here sorts first, where the sample would take it."""
    chunks = [
        *(defined(f"name_{number:02}", "a.py") for number in range(12)),
        defined("name_03", "a.py", ChunkKind.method),
        defined("aaa", "a.py"),
        defined("aaa", "b.py"),
        defined("Display for Path", "a.py", ChunkKind.klass),
        defined("Install", "a.md", ChunkKind.section),
    ]

    assert sample_names(chunks) == [
        Answer(path="a.py", symbol="name_00"),
        Answer(path="a.py", symbol="name_05"),
        Answer(path="a.py", symbol="name_10"),
    ]


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
