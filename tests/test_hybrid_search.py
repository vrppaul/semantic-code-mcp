"""Tests for hybrid search (vector + full-text)."""

from pathlib import Path

import pytest

from semantic_code_mcp.models import Chunk, ChunkType, ChunkWithEmbedding
from semantic_code_mcp.storage.lancedb import VectorStore


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Create a temporary database path."""
    return tmp_path / "test_db"


@pytest.fixture
def store_with_chunks(temp_db_path: Path) -> VectorStore:
    """Create a store with sample chunks for testing hybrid search."""
    store = VectorStore(temp_db_path)

    # Create embeddings that are actually different (not parallel)
    # Using varied values to get different cosine similarities
    emb_low = [0.1 if i % 2 == 0 else -0.1 for i in range(384)]
    emb_high = [0.9 if i % 2 == 0 else 0.8 for i in range(384)]
    emb_mid = [0.5 if i % 3 == 0 else 0.3 for i in range(384)]

    chunks = [
        # Chunk with "duration_ms" - should be found by keyword search
        ChunkWithEmbedding(
            chunk=Chunk(
                file_path="/project/indexer.py",
                line_start=100,
                line_end=120,
                content='log.debug("timing", duration_ms=round(time.time() - t0))',
                chunk_type=ChunkType.METHOD,
                name="index",
            ),
            embedding=emb_low,  # Low similarity to query
        ),
        # Chunk semantically about timing but no "duration_ms" keyword
        ChunkWithEmbedding(
            chunk=Chunk(
                file_path="/project/utils.py",
                line_start=1,
                line_end=10,
                content="def measure_elapsed_time(): return time.perf_counter()",
                chunk_type=ChunkType.FUNCTION,
                name="measure_elapsed_time",
            ),
            embedding=emb_high,  # High similarity to query
        ),
        # Chunk with "log.debug" but different topic
        ChunkWithEmbedding(
            chunk=Chunk(
                file_path="/project/cache.py",
                line_start=50,
                line_end=60,
                content='log.debug("cache_saved", files_count=len(files))',
                chunk_type=ChunkType.METHOD,
                name="_save",
            ),
            embedding=emb_mid,
        ),
    ]

    store.add_chunks(chunks)
    return store


class TestHybridSearch:
    """Tests for hybrid search functionality."""

    def test_vector_only_search_misses_keyword_match(self, store_with_chunks: VectorStore):
        """Pure vector search may miss results with exact keyword matches."""
        # Query embedding similar to "measure_elapsed_time" chunk (emb_high pattern)
        query_embedding = [0.9 if i % 2 == 0 else 0.8 for i in range(384)]

        results = store_with_chunks.search(query_embedding, limit=2)

        # Vector search finds semantically similar, not keyword matches
        assert len(results) > 0
        # The "duration_ms" chunk has low embedding similarity, may not appear
        names = [r.name for r in results]
        assert "measure_elapsed_time" in names

    def test_fts_search_finds_exact_keyword(self, store_with_chunks: VectorStore):
        """Full-text search finds chunks containing exact keywords."""
        results = store_with_chunks.search_fts("duration_ms", limit=5)

        assert len(results) >= 1
        assert any("duration_ms" in r.content for r in results)

    def test_fts_search_returns_empty_for_no_match(self, store_with_chunks: VectorStore):
        """FTS returns empty list when no matches found."""
        results = store_with_chunks.search_fts("nonexistent_keyword_xyz", limit=5)
        assert results == []

    def test_hybrid_search_combines_both(self, store_with_chunks: VectorStore):
        """Hybrid search finds both semantic and keyword matches."""
        query_embedding = [
            0.9 if i % 2 == 0 else 0.8 for i in range(384)
        ]  # Similar to measure_elapsed_time

        results = store_with_chunks.search_hybrid(
            query_embedding=query_embedding,
            query_text="duration_ms",
            limit=5,
        )

        # Should find the keyword match in results
        assert "duration_ms" in " ".join(r.content for r in results)

    def test_hybrid_search_weight_adjustable(self, store_with_chunks: VectorStore):
        """Can adjust weight between vector and FTS search."""
        query_embedding = [0.9 if i % 2 == 0 else 0.8 for i in range(384)]

        # Heavy FTS weight should prioritize keyword matches
        results_fts_heavy = store_with_chunks.search_hybrid(
            query_embedding=query_embedding,
            query_text="duration_ms",
            limit=5,
            vector_weight=0.3,  # 30% vector, 70% FTS
        )

        # Heavy vector weight should prioritize semantic matches
        results_vec_heavy = store_with_chunks.search_hybrid(
            query_embedding=query_embedding,
            query_text="duration_ms",
            limit=5,
            vector_weight=0.9,  # 90% vector, 10% FTS
        )

        # Both should return results, but ordering may differ
        assert len(results_fts_heavy) > 0
        assert len(results_vec_heavy) > 0
