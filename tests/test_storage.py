"""Tests for LanceDB storage layer."""

from pathlib import Path

import numpy as np

from semantic_code_mcp.models import Chunk, ChunkType, ChunkWithEmbedding
from semantic_code_mcp.storage.lancedb import LanceDBConnection, LanceDBVectorStore


class TestLanceDBConnection:
    """Tests for LanceDBConnection (expensive, shared resource)."""

    def test_create_connection(self, temp_db_path: Path):
        """Can create a new connection."""
        conn = LanceDBConnection(temp_db_path)
        assert conn is not None
        assert conn.db is not None

    def test_connection_creates_table(self, temp_db_path: Path):
        """Connection creates chunks table on init."""
        conn = LanceDBConnection(temp_db_path)
        # Table property should work without error
        table = conn.table
        assert table is not None


class TestLanceDBVectorStore:
    """Tests for LanceDBVectorStore (per-request session)."""

    def test_create_store(self, lance_connection: LanceDBConnection):
        """Can create a new vector store with connection."""
        store = LanceDBVectorStore(lance_connection)
        assert store is not None
        assert store.connection is lance_connection

    def test_store_and_retrieve_chunk(
        self, vector_store: LanceDBVectorStore, sample_chunk: Chunk, sample_embedding: list[float]
    ):
        """Can store a chunk with embedding and retrieve it."""
        item = ChunkWithEmbedding(chunk=sample_chunk, embedding=sample_embedding)
        vector_store.add_chunks([item])

        results = vector_store.search(sample_embedding, limit=1)

        assert len(results) == 1
        assert results[0].file_path == sample_chunk.file_path
        assert results[0].name == sample_chunk.name

    def test_search_returns_similar_items(self, vector_store: LanceDBVectorStore):
        """Search returns items ranked by similarity."""
        chunk1 = Chunk(
            file_path="/a.py",
            line_start=1,
            line_end=5,
            content="def foo(): pass",
            chunk_type=ChunkType.FUNCTION,
            name="foo",
        )
        chunk2 = Chunk(
            file_path="/b.py",
            line_start=1,
            line_end=5,
            content="def bar(): pass",
            chunk_type=ChunkType.FUNCTION,
            name="bar",
        )

        # chunk1 embedding is similar to query, chunk2 is different
        query = [1.0] * 384
        embed1 = [0.9] * 384  # Similar to query
        embed2 = [-0.9] * 384  # Different from query

        vector_store.add_chunks(
            [
                ChunkWithEmbedding(chunk=chunk1, embedding=embed1),
                ChunkWithEmbedding(chunk=chunk2, embedding=embed2),
            ]
        )

        results = vector_store.search(query, limit=2)

        assert len(results) == 2
        assert results[0].name == "foo"  # More similar should be first
        assert results[0].score > results[1].score

    def test_search_respects_limit(self, vector_store: LanceDBVectorStore):
        """Search respects the limit parameter."""
        items = [
            ChunkWithEmbedding(
                chunk=Chunk(
                    file_path=f"/file{i}.py",
                    line_start=1,
                    line_end=5,
                    content=f"def func{i}(): pass",
                    chunk_type=ChunkType.FUNCTION,
                    name=f"func{i}",
                ),
                embedding=np.random.rand(384).tolist(),
            )
            for i in range(10)
        ]

        vector_store.add_chunks(items)

        results = vector_store.search([0.5] * 384, limit=3)
        assert len(results) == 3

    def test_delete_chunks_by_file(self, vector_store: LanceDBVectorStore):
        """Can delete all chunks for a specific file."""
        chunk1 = Chunk(
            file_path="/a.py",
            line_start=1,
            line_end=5,
            content="def foo(): pass",
            chunk_type=ChunkType.FUNCTION,
            name="foo",
        )
        chunk2 = Chunk(
            file_path="/b.py",
            line_start=1,
            line_end=5,
            content="def bar(): pass",
            chunk_type=ChunkType.FUNCTION,
            name="bar",
        )

        vector_store.add_chunks(
            [
                ChunkWithEmbedding(chunk=chunk1, embedding=[0.5] * 384),
                ChunkWithEmbedding(chunk=chunk2, embedding=[0.5] * 384),
            ]
        )

        vector_store.delete_by_file("/a.py")

        results = vector_store.search([0.5] * 384, limit=10)
        assert len(results) == 1
        assert results[0].file_path == "/b.py"

    def test_empty_store_returns_empty_results(self, vector_store: LanceDBVectorStore):
        """Search on empty store returns empty list."""
        results = vector_store.search([0.5] * 384, limit=10)
        assert results == []

    def test_get_all_file_paths(self, vector_store: LanceDBVectorStore):
        """Can get list of all indexed file paths."""
        chunks = [
            Chunk(
                file_path="/a.py",
                line_start=1,
                line_end=5,
                content="def foo(): pass",
                chunk_type=ChunkType.FUNCTION,
                name="foo",
            ),
            Chunk(
                file_path="/a.py",
                line_start=10,
                line_end=15,
                content="def bar(): pass",
                chunk_type=ChunkType.FUNCTION,
                name="bar",
            ),
            Chunk(
                file_path="/b.py",
                line_start=1,
                line_end=5,
                content="class Baz: pass",
                chunk_type=ChunkType.CLASS,
                name="Baz",
            ),
        ]

        vector_store.add_chunks(
            [
                ChunkWithEmbedding(chunk=chunks[0], embedding=[0.1] * 384),
                ChunkWithEmbedding(chunk=chunks[1], embedding=[0.2] * 384),
                ChunkWithEmbedding(chunk=chunks[2], embedding=[0.3] * 384),
            ]
        )

        file_paths = vector_store.get_indexed_files()
        assert set(file_paths) == {"/a.py", "/b.py"}

    def test_count_chunks(self, vector_store: LanceDBVectorStore):
        """Can count total chunks in store."""
        assert vector_store.count() == 0

        items = [
            ChunkWithEmbedding(
                chunk=Chunk(
                    file_path=f"/file{i}.py",
                    line_start=1,
                    line_end=5,
                    content=f"def func{i}(): pass",
                    chunk_type=ChunkType.FUNCTION,
                    name=f"func{i}",
                ),
                embedding=[0.5] * 384,
            )
            for i in range(5)
        ]
        vector_store.add_chunks(items)

        assert vector_store.count() == 5

    def test_clear_removes_all_chunks(self, vector_store: LanceDBVectorStore):
        """Clear removes all chunks from the store."""
        items = [
            ChunkWithEmbedding(
                chunk=Chunk(
                    file_path=f"/file{i}.py",
                    line_start=1,
                    line_end=5,
                    content=f"def func{i}(): pass",
                    chunk_type=ChunkType.FUNCTION,
                    name=f"func{i}",
                ),
                embedding=[0.5] * 384,
            )
            for i in range(5)
        ]
        vector_store.add_chunks(items)
        assert vector_store.count() == 5

        vector_store.clear()

        assert vector_store.count() == 0
        assert vector_store.get_indexed_files() == []

    def test_shared_connection_across_stores(self, lance_connection: LanceDBConnection):
        """Multiple stores share the same connection."""
        store1 = LanceDBVectorStore(lance_connection)
        store2 = LanceDBVectorStore(lance_connection)

        # Add through store1
        chunk = Chunk(
            file_path="/test.py",
            line_start=1,
            line_end=5,
            content="def test(): pass",
            chunk_type=ChunkType.FUNCTION,
            name="test",
        )
        store1.add_chunks([ChunkWithEmbedding(chunk=chunk, embedding=[0.5] * 384)])

        # Should be visible through store2
        assert store2.count() == 1
        results = store2.search([0.5] * 384, limit=1)
        assert len(results) == 1
        assert results[0].name == "test"
