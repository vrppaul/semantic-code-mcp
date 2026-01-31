"""Tests for CompositeChunker dispatcher."""

from pathlib import Path

import pytest

from semantic_code_mcp.chunkers.composite import CompositeChunker
from semantic_code_mcp.chunkers.markdown import MarkdownChunker
from semantic_code_mcp.chunkers.python import PythonChunker
from semantic_code_mcp.chunkers.rust import RustChunker
from semantic_code_mcp.models import ChunkType


class TestCompositeChunker:
    """Tests for CompositeChunker."""

    def test_routes_py_to_python_chunker(self, tmp_path: Path):
        """Routes .py files to PythonChunker."""
        file_path = tmp_path / "test.py"
        file_path.write_text("def hello(): pass\n")

        chunker = CompositeChunker([PythonChunker(), RustChunker()])
        chunks = chunker.chunk_file(str(file_path))

        assert len(chunks) == 1
        assert chunks[0].name == "hello"
        assert chunks[0].chunk_type == ChunkType.function

    def test_routes_rs_to_rust_chunker(self, tmp_path: Path):
        """Routes .rs files to RustChunker."""
        file_path = tmp_path / "lib.rs"
        file_path.write_text('fn greet() { println!("hi"); }\n')

        chunker = CompositeChunker([PythonChunker(), RustChunker()])
        chunks = chunker.chunk_file(str(file_path))

        assert len(chunks) == 1
        assert chunks[0].name == "greet"
        assert chunks[0].chunk_type == ChunkType.function

    def test_routes_md_to_markdown_chunker(self, tmp_path: Path):
        """Routes .md files to MarkdownChunker."""
        file_path = tmp_path / "README.md"
        file_path.write_text("# Hello\n\nWorld.\n")

        chunker = CompositeChunker([PythonChunker(), RustChunker(), MarkdownChunker()])
        chunks = chunker.chunk_file(str(file_path))

        assert len(chunks) == 1
        assert chunks[0].name == "Hello"
        assert chunks[0].chunk_type == ChunkType.section

    def test_unknown_extension_returns_empty(self, tmp_path: Path):
        """Unknown file extension returns empty list."""
        file_path = tmp_path / "data.txt"
        file_path.write_text("some text\n")

        chunker = CompositeChunker([PythonChunker(), RustChunker(), MarkdownChunker()])
        chunks = chunker.chunk_file(str(file_path))

        assert chunks == []

    def test_supported_extensions(self):
        """supported_extensions returns all registered extensions."""
        chunker = CompositeChunker([PythonChunker(), RustChunker(), MarkdownChunker()])
        exts = chunker.supported_extensions

        assert ".py" in exts
        assert ".rs" in exts
        assert ".md" in exts

    def test_no_chunkers_returns_empty(self, tmp_path: Path):
        """Empty chunker list returns empty for any file."""
        file_path = tmp_path / "test.py"
        file_path.write_text("def hello(): pass\n")

        chunker = CompositeChunker([])
        chunks = chunker.chunk_file(str(file_path))

        assert chunks == []

    def test_supported_extensions_empty(self):
        """Empty chunker list has no supported extensions."""
        chunker = CompositeChunker([])
        assert chunker.supported_extensions == []

    def test_collision_detection(self):
        """Raises ValueError when two chunkers claim the same extension."""
        with pytest.raises(ValueError, match=r"already registered"):
            CompositeChunker([PythonChunker(), PythonChunker()])
