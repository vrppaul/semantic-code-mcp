"""Tests for data models."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from semantic_code_mcp.models import Chunk, ChunkType, ChunkWithEmbedding, IndexStatus, SearchResult


class TestChunk:
    """Tests for Chunk model."""

    def test_create_function_chunk(self):
        """Chunk can be created with valid function data."""
        chunk = Chunk(
            file_path="/path/to/file.py",
            line_start=10,
            line_end=20,
            content="def hello():\n    pass",
            chunk_type=ChunkType.FUNCTION,
            name="hello",
        )
        assert chunk.file_path == "/path/to/file.py"
        assert chunk.line_start == 10
        assert chunk.line_end == 20
        assert chunk.content == "def hello():\n    pass"
        assert chunk.chunk_type == ChunkType.FUNCTION
        assert chunk.name == "hello"

    def test_create_class_chunk(self):
        """Chunk can be created with class type."""
        chunk = Chunk(
            file_path="/path/to/file.py",
            line_start=1,
            line_end=50,
            content="class Foo:\n    pass",
            chunk_type=ChunkType.CLASS,
            name="Foo",
        )
        assert chunk.chunk_type == ChunkType.CLASS

    def test_create_method_chunk(self):
        """Chunk can be created with method type."""
        chunk = Chunk(
            file_path="/path/to/file.py",
            line_start=5,
            line_end=10,
            content="def bar(self):\n    pass",
            chunk_type=ChunkType.METHOD,
            name="bar",
        )
        assert chunk.chunk_type == ChunkType.METHOD

    def test_line_end_must_be_gte_line_start(self):
        """line_end must be >= line_start."""
        with pytest.raises(ValidationError):
            Chunk(
                file_path="/path/to/file.py",
                line_start=20,
                line_end=10,  # Invalid: end < start
                content="def hello():\n    pass",
                chunk_type=ChunkType.FUNCTION,
                name="hello",
            )

    def test_line_numbers_must_be_positive(self):
        """Line numbers must be positive."""
        with pytest.raises(ValidationError):
            Chunk(
                file_path="/path/to/file.py",
                line_start=0,  # Invalid: must be >= 1
                line_end=10,
                content="def hello():\n    pass",
                chunk_type=ChunkType.FUNCTION,
                name="hello",
            )

    def test_chunk_to_dict(self):
        """Chunk serializes to dict for MCP responses."""
        chunk = Chunk(
            file_path="/path/to/file.py",
            line_start=10,
            line_end=20,
            content="def hello():\n    pass",
            chunk_type=ChunkType.FUNCTION,
            name="hello",
        )
        d = chunk.model_dump()
        assert d["file_path"] == "/path/to/file.py"
        assert d["line_start"] == 10
        assert d["chunk_type"] == "function"


class TestChunkWithEmbedding:
    """Tests for ChunkWithEmbedding model."""

    def test_create_chunk_with_embedding(self):
        """ChunkWithEmbedding pairs a chunk with its vector."""
        chunk = Chunk(
            file_path="/path/to/file.py",
            line_start=10,
            line_end=20,
            content="def hello():\n    pass",
            chunk_type=ChunkType.FUNCTION,
            name="hello",
        )
        embedding = [0.1, 0.2, 0.3]

        item = ChunkWithEmbedding(chunk=chunk, embedding=embedding)

        assert item.chunk == chunk
        assert item.embedding == embedding

    def test_chunk_with_embedding_to_dict(self):
        """ChunkWithEmbedding serializes correctly."""
        chunk = Chunk(
            file_path="/path/to/file.py",
            line_start=10,
            line_end=20,
            content="def hello():\n    pass",
            chunk_type=ChunkType.FUNCTION,
            name="hello",
        )
        item = ChunkWithEmbedding(chunk=chunk, embedding=[0.5] * 384)
        d = item.model_dump()

        assert "chunk" in d
        assert "embedding" in d
        assert len(d["embedding"]) == 384


class TestSearchResult:
    """Tests for SearchResult model."""

    def test_search_result_has_score(self):
        """SearchResult includes score field."""
        result = SearchResult(
            file_path="/path/to/file.py",
            line_start=10,
            line_end=20,
            content="def hello():\n    pass",
            chunk_type=ChunkType.FUNCTION,
            name="hello",
            score=0.95,
        )
        assert result.score == 0.95

    def test_score_must_be_between_0_and_1(self):
        """Score must be in [0, 1] range."""
        with pytest.raises(ValidationError):
            SearchResult(
                file_path="/path/to/file.py",
                line_start=10,
                line_end=20,
                content="def hello():\n    pass",
                chunk_type=ChunkType.FUNCTION,
                name="hello",
                score=1.5,  # Invalid: > 1
            )

    def test_search_result_to_dict(self):
        """SearchResult serializes to dict with score."""
        result = SearchResult(
            file_path="/path/to/file.py",
            line_start=10,
            line_end=20,
            content="def hello():\n    pass",
            chunk_type=ChunkType.FUNCTION,
            name="hello",
            score=0.85,
        )
        d = result.model_dump()
        assert d["score"] == 0.85
        assert d["file_path"] == "/path/to/file.py"


class TestIndexStatus:
    """Tests for IndexStatus model."""

    def test_create_indexed_status(self):
        """IndexStatus for an indexed codebase."""
        status = IndexStatus(
            is_indexed=True,
            last_updated=datetime(2024, 1, 24, 12, 0, 0),
            files_count=100,
            chunks_count=500,
            stale_files=[],
        )
        assert status.is_indexed is True
        assert status.files_count == 100
        assert status.chunks_count == 500
        assert status.stale_files == []

    def test_create_not_indexed_status(self):
        """IndexStatus for a non-indexed codebase."""
        status = IndexStatus(
            is_indexed=False,
            last_updated=None,
            files_count=0,
            chunks_count=0,
            stale_files=[],
        )
        assert status.is_indexed is False
        assert status.last_updated is None

    def test_status_with_stale_files(self):
        """IndexStatus can track stale files."""
        status = IndexStatus(
            is_indexed=True,
            last_updated=datetime(2024, 1, 24, 12, 0, 0),
            files_count=100,
            chunks_count=500,
            stale_files=["file1.py", "file2.py"],
        )
        assert len(status.stale_files) == 2
        assert "file1.py" in status.stale_files

    def test_index_status_to_dict(self):
        """IndexStatus serializes to dict for MCP responses."""
        status = IndexStatus(
            is_indexed=True,
            last_updated=datetime(2024, 1, 24, 12, 0, 0),
            files_count=100,
            chunks_count=500,
            stale_files=[],
        )
        d = status.model_dump()
        assert d["is_indexed"] is True
        assert d["files_count"] == 100
