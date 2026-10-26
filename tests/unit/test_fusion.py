"""Tests for fusing what the two channels of a search found into one ranking."""

from pathlib import Path

from hypothesis import given, strategies

from semantic_code_mcp.domain.models import (
    Candidate,
    Chunk,
    ChunkKind,
    Language,
    SearchCandidates,
)
from semantic_code_mcp.services.fusion import fuse

SIMILARITY = 0.5
# The chunks a channel can find, each at most once, in the order it ranks them
CHUNK_IDS = [f"chunk-{number}" for number in range(12)]
CHANNEL_HITS = strategies.lists(strategies.sampled_from(CHUNK_IDS), unique=True, max_size=8)
# A chunk none of CHUNK_IDS is, whose rank a test follows
FOLLOWED = "followed"


def candidate(chunk_id: str, similarity: float = SIMILARITY) -> Candidate:
    chunk = Chunk(
        chunk_id=chunk_id,
        path=Path("module.py"),
        language=Language.python,
        kind=ChunkKind.function,
        name=chunk_id,
        parent=None,
        line_start=1,
        line_end=1,
        content=f"def {chunk_id}(): ...",
    )
    return Candidate(chunk=chunk, similarity=similarity)


def fused_ids(by_vector: list[str], by_keyword: list[str]) -> list[str]:
    """The chunk ids in fused order, every candidate of the same similarity."""
    candidates = SearchCandidates(
        by_vector=[candidate(chunk_id) for chunk_id in by_vector],
        by_keyword=[candidate(chunk_id) for chunk_id in by_keyword],
    )
    return [hit.chunk.chunk_id for hit in fuse(candidates)]


@given(by_vector=CHANNEL_HITS, by_keyword=CHANNEL_HITS)
def test_fusion_ranks_each_chunk_either_channel_found_once(
    by_vector: list[str], by_keyword: list[str]
):
    assert sorted(fused_ids(by_vector, by_keyword)) == sorted({*by_vector, *by_keyword})


@given(by_vector=CHANNEL_HITS, by_keyword=CHANNEL_HITS, data=strategies.data())
def test_a_chunk_the_other_channel_finds_too_never_ranks_lower(
    by_vector: list[str], by_keyword: list[str], data: strategies.DataObject
):
    vector_position = data.draw(strategies.integers(0, len(by_vector)))
    keyword_position = data.draw(strategies.integers(0, len(by_keyword)))
    found_by_vector = [*by_vector[:vector_position], FOLLOWED, *by_vector[vector_position:]]
    found_by_both = [*by_keyword[:keyword_position], FOLLOWED, *by_keyword[keyword_position:]]

    rank_by_vector = fused_ids(found_by_vector, by_keyword).index(FOLLOWED)
    rank_by_both = fused_ids(found_by_vector, found_by_both).index(FOLLOWED)

    assert rank_by_both <= rank_by_vector


def test_a_chunk_both_channels_rank_second_comes_before_either_channel_s_first():
    assert fused_ids(["vector_first", "both"], ["keyword_first", "both"])[0] == "both"


def test_chunks_ranked_alike_come_closest_in_meaning_first():
    candidates = SearchCandidates(
        by_vector=[candidate("vector_first", similarity=0.4)],
        by_keyword=[candidate("keyword_first", similarity=0.6)],
    )

    assert [hit.chunk.chunk_id for hit in fuse(candidates)] == ["keyword_first", "vector_first"]
