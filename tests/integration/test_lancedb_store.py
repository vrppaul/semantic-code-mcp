"""Tests for the LanceDB store: what is written comes back, by vector and by words."""

from pathlib import Path

import pytest

from semantic_code_mcp.domain.models import Chunk, ChunkType, ChunkWithEmbedding
from semantic_code_mcp.storage.lancedb import LanceDBVectorStore
from tests.fakes import EMBEDDING_DIMENSION

TOWARD_QUERY = [0.9] * EMBEDDING_DIMENSION
QUERY = [1.0] * EMBEDDING_DIMENSION
AWAY_FROM_QUERY = [-0.9] * EMBEDDING_DIMENSION
ALTERNATING = [0.1, -0.1] * (EMBEDDING_DIMENSION // 2)
LIMIT = 10
VECTOR_WEIGHT = 0.5


def stored(name: str, file_path: str, content: str, embedding: list[float]) -> ChunkWithEmbedding:
    return ChunkWithEmbedding(
        chunk=Chunk(
            file_path=file_path,
            line_start=1,
            line_end=3,
            content=content,
            chunk_type=ChunkType.function,
            name=name,
        ),
        embedding=embedding,
    )


@pytest.fixture
def store(tmp_path: Path) -> LanceDBVectorStore:
    return LanceDBVectorStore(tmp_path / "index")


@pytest.fixture
def filled_store(store: LanceDBVectorStore) -> LanceDBVectorStore:
    store.add_chunks(
        [
            stored("near", "/project/near.py", "def near(): return elapsed_time()", TOWARD_QUERY),
            stored(
                "keyword",
                "/project/indexer.py",
                'log.debug("timing", duration_ms=round(time.time() - started))',
                ALTERNATING,
            ),
            stored("far", "/project/far.py", "def far(): return None", AWAY_FROM_QUERY),
        ]
    )
    return store


def test_vector_search_ranks_the_nearest_chunk_first(filled_store: LanceDBVectorStore):
    results = filled_store.search_vector(QUERY, LIMIT)

    assert [result.name for result in results] == ["near", "keyword", "far"]


def test_full_text_search_finds_an_exact_identifier(filled_store: LanceDBVectorStore):
    results = filled_store.search_full_text("duration_ms", LIMIT)

    assert [result.name for result in results] == ["keyword"]


def test_hybrid_search_returns_both_the_nearest_and_the_keyword_match(
    filled_store: LanceDBVectorStore,
):
    results = filled_store.search_hybrid(QUERY, "duration_ms", 2, VECTOR_WEIGHT)

    assert {result.name for result in results} == {"near", "keyword"}


def test_an_empty_store_finds_nothing(store: LanceDBVectorStore):
    assert store.search_hybrid(QUERY, "anything", LIMIT, VECTOR_WEIGHT) == []


def test_deleting_files_removes_only_their_chunks(filled_store: LanceDBVectorStore):
    filled_store.delete_by_files(["/project/near.py", "/project/it's quoted.py"])

    assert sorted(filled_store.statistics().indexed_files) == [
        "/project/far.py",
        "/project/indexer.py",
    ]


def test_statistics_count_files_and_chunks(filled_store: LanceDBVectorStore):
    statistics = filled_store.statistics()

    assert (sorted(statistics.indexed_files), statistics.chunks_count) == (
        ["/project/far.py", "/project/indexer.py", "/project/near.py"],
        3,
    )


def test_clear_empties_the_store(filled_store: LanceDBVectorStore):
    filled_store.clear()

    assert filled_store.statistics().chunks_count == 0
