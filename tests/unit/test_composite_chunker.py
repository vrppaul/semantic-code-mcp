"""Tests for CompositeChunker dispatcher."""

import tempfile
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from semantic_code_mcp.app import build_chunker
from semantic_code_mcp.chunkers.composite import CHUNKER_VERSION, CompositeChunker
from semantic_code_mcp.chunkers.markdown import MarkdownChunker
from semantic_code_mcp.chunkers.python import PythonChunker
from semantic_code_mcp.chunkers.rust import RustChunker
from semantic_code_mcp.domain.models import Chunk, ChunkKind, Language

# A few files of every language, and what each chunker version made of them
RECORDED_CHUNKS = Path(__file__).parent.parent / "data" / "chunker_outputs.json"
RECORD_INDENT = 2


class RecordedChunks(BaseModel):
    """Files, by name, and the chunks the chunkers made of them, by CHUNKER_VERSION."""

    model_config = ConfigDict(frozen=True)

    sources: dict[str, str]
    chunks: dict[int, list[Chunk]]


def chunks_of(sources: dict[str, str], project_root: Path) -> list[Chunk]:
    """What the chunkers the server ships make of the files, written into a project, in the
    order of their names."""
    for name, source in sources.items():
        (project_root / name).write_text(source)
    chunker = build_chunker()
    return [
        chunk for name in sorted(sources) for chunk in chunker.chunk_file(project_root, Path(name))
    ]


def record_chunks() -> None:
    """Record what the chunkers make of the files under the current CHUNKER_VERSION. Run it
    by hand from the repository's root, through `uv run python -c` and an import of this
    module. A version recorded before keeps what it made: bump the version rather than
    record over it."""
    recorded = RecordedChunks.model_validate_json(RECORDED_CHUNKS.read_text())
    with tempfile.TemporaryDirectory() as project_root:
        chunks = chunks_of(recorded.sources, Path(project_root))
    updated = RecordedChunks(
        sources=recorded.sources, chunks=recorded.chunks | {CHUNKER_VERSION: chunks}
    )
    RECORDED_CHUNKS.write_text(updated.model_dump_json(indent=RECORD_INDENT) + "\n")


def test_the_chunker_s_version_names_the_chunks_it_makes(tmp_path: Path):
    """An index holds the chunks of one CHUNKER_VERSION, so a change to what a file gives
    needs a new version, whose chunks `record_chunks` records; none recorded fails here."""
    recorded = RecordedChunks.model_validate_json(RECORDED_CHUNKS.read_text())

    chunks = chunks_of(recorded.sources, tmp_path)

    assert chunks == recorded.chunks.get(CHUNKER_VERSION)


class TestCompositeChunker:
    """Tests for CompositeChunker."""

    def test_routes_py_to_python_chunker(self, tmp_path: Path):
        """Routes .py files to PythonChunker."""
        file_path = tmp_path / "test.py"
        file_path.write_text("def hello(): pass\n")

        chunker = CompositeChunker([PythonChunker(), RustChunker()])
        chunks = chunker.chunk_file(file_path.parent, Path(file_path.name))

        assert [(chunk.name, chunk.kind, chunk.language) for chunk in chunks] == [
            ("hello", ChunkKind.function, Language.python)
        ]

    def test_routes_rs_to_rust_chunker(self, tmp_path: Path):
        """Routes .rs files to RustChunker."""
        file_path = tmp_path / "lib.rs"
        file_path.write_text('fn greet() { println!("hi"); }\n')

        chunker = CompositeChunker([PythonChunker(), RustChunker()])
        chunks = chunker.chunk_file(file_path.parent, Path(file_path.name))

        assert [(chunk.name, chunk.kind, chunk.language) for chunk in chunks] == [
            ("greet", ChunkKind.function, Language.rust)
        ]

    def test_routes_md_to_markdown_chunker(self, tmp_path: Path):
        """Routes .md files to MarkdownChunker."""
        file_path = tmp_path / "README.md"
        file_path.write_text("# Hello\n\nWorld.\n")

        chunker = CompositeChunker([PythonChunker(), RustChunker(), MarkdownChunker()])
        chunks = chunker.chunk_file(file_path.parent, Path(file_path.name))

        assert [(chunk.name, chunk.kind, chunk.language) for chunk in chunks] == [
            ("Hello", ChunkKind.section, Language.markdown)
        ]

    def test_unknown_extension_returns_empty(self, tmp_path: Path):
        """Unknown file extension returns empty list."""
        file_path = tmp_path / "data.txt"
        file_path.write_text("some text\n")

        chunker = CompositeChunker([PythonChunker(), RustChunker(), MarkdownChunker()])
        chunks = chunker.chunk_file(file_path.parent, Path(file_path.name))

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
        chunks = chunker.chunk_file(file_path.parent, Path(file_path.name))

        assert chunks == []

    def test_supported_extensions_empty(self):
        """Empty chunker list has no supported extensions."""
        chunker = CompositeChunker([])
        assert chunker.supported_extensions == []

    def test_collision_detection(self):
        """Raises ValueError when two chunkers claim the same extension."""
        with pytest.raises(ValueError, match=r"already registered"):
            CompositeChunker([PythonChunker(), PythonChunker()])
