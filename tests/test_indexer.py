"""Tests for indexer orchestration."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from semantic_code_mcp.config import Settings, get_index_path
from semantic_code_mcp.indexer.chunker import PythonChunker
from semantic_code_mcp.indexer.embedder import Embedder
from semantic_code_mcp.indexer.indexer import Indexer
from semantic_code_mcp.models import IndexResult
from semantic_code_mcp.storage.lancedb import LanceDBConnection, LanceDBVectorStore


async def run_index(indexer: Indexer, project_path: Path, force: bool = False) -> IndexResult:
    """Helper to run the atomic indexing pipeline and return result."""
    files = indexer.scan_files(project_path)
    plan = indexer.detect_changes(project_path, files, force=force)
    if not plan.has_work:
        return IndexResult(files_indexed=0, chunks_indexed=0, files_deleted=0)
    chunks = await indexer.chunk_files(plan.files_to_index)
    return await indexer.embed_and_store(project_path, plan, chunks)


class TestIndexer:
    """Tests for Indexer orchestration with injected dependencies."""

    @pytest.fixture
    def chunker(self) -> PythonChunker:
        return PythonChunker()

    @pytest.fixture
    def store(self, test_settings: Settings, sample_project: Path) -> LanceDBVectorStore:
        index_path = get_index_path(test_settings, sample_project)
        index_path.mkdir(parents=True, exist_ok=True)
        connection = LanceDBConnection(index_path)
        return LanceDBVectorStore(connection)

    @pytest.fixture
    def indexer(
        self,
        test_settings: Settings,
        embedder: Embedder,
        store: LanceDBVectorStore,
        chunker: PythonChunker,
        sample_project: Path,
    ) -> Indexer:
        cache_dir = get_index_path(test_settings, sample_project)
        return Indexer(
            settings=test_settings,
            embedder=embedder,
            store=store,
            chunker=chunker,
            cache_dir=cache_dir,
        )

    def test_create_indexer(
        self,
        test_settings: Settings,
        embedder: Embedder,
        store: LanceDBVectorStore,
        chunker: PythonChunker,
    ):
        """Can create an indexer instance with injected dependencies."""
        indexer = Indexer(
            settings=test_settings,
            embedder=embedder,
            store=store,
            chunker=chunker,
        )
        assert indexer is not None
        assert indexer.embedder is embedder
        assert indexer.store is store
        assert indexer.chunker is chunker

    def test_scan_finds_python_files(self, indexer: Indexer, sample_project: Path):
        """Scans directory and finds Python files."""
        files = indexer.scan_files(sample_project)

        assert len(files) == 2
        filenames = {Path(f).name for f in files}
        assert "main.py" in filenames
        assert "utils.py" in filenames

    def test_scan_ignores_non_python(self, indexer: Indexer, tmp_path: Path):
        """Ignores non-Python files."""
        project = tmp_path / "project2"
        project.mkdir()
        (project / "code.py").write_text("# Python")
        (project / "readme.md").write_text("# Readme")
        (project / "data.json").write_text("{}")

        files = indexer.scan_files(project)

        assert len(files) == 1
        assert "code.py" in files[0]

    def test_scan_respects_ignore_patterns(self, indexer: Indexer, tmp_path: Path):
        """Respects ignore patterns from settings."""
        project = tmp_path / "project3"
        project.mkdir()
        (project / "code.py").write_text("# Code")

        # Create ignored directories
        venv = project / ".venv"
        venv.mkdir()
        (venv / "lib.py").write_text("# Venv lib")

        pycache = project / "__pycache__"
        pycache.mkdir()
        (pycache / "code.cpython-311.pyc").write_text("# Compiled")

        files = indexer.scan_files(project)

        # Should only find code.py, not files in ignored dirs
        assert len(files) == 1
        assert "code.py" in files[0]

    def test_scan_respects_gitignore(self, indexer: Indexer, tmp_path: Path):
        """Respects .gitignore patterns."""
        project = tmp_path / "project4"
        project.mkdir()

        (project / ".gitignore").write_text("ignored/\n*.generated.py\n")
        (project / "code.py").write_text("# Code")
        (project / "model.generated.py").write_text("# Generated")

        ignored = project / "ignored"
        ignored.mkdir()
        (ignored / "secret.py").write_text("# Secret")

        files = indexer.scan_files(project)

        assert len(files) == 1
        assert "code.py" in files[0]

    def test_detect_changes_force(self, indexer: Indexer, sample_project: Path):
        """Force mode returns all files for indexing."""
        files = indexer.scan_files(sample_project)
        plan = indexer.detect_changes(sample_project, files, force=True)

        assert plan.files_to_index == files
        assert plan.files_to_delete == []
        assert plan.has_work

    def test_detect_changes_no_prior_index(self, indexer: Indexer, sample_project: Path):
        """All files are new when no prior index exists."""
        files = indexer.scan_files(sample_project)
        plan = indexer.detect_changes(sample_project, files, force=False)

        assert len(plan.files_to_index) == 2
        assert plan.has_work

    @pytest.mark.asyncio
    async def test_index_project_creates_chunks(self, indexer: Indexer, sample_project: Path):
        """Index creates chunks in vector store."""
        result = await run_index(indexer, sample_project)

        assert result.files_indexed > 0
        assert result.chunks_indexed > 0
        assert result.files_indexed == 2  # main.py and utils.py

    @pytest.mark.asyncio
    async def test_index_incremental_skips_unchanged(self, indexer: Indexer, sample_project: Path):
        """Incremental index skips unchanged files."""
        # First index
        await run_index(indexer, sample_project)

        # Second index without changes
        result2 = await run_index(indexer, sample_project, force=False)

        # No files should be re-indexed
        assert result2.files_indexed == 0
        assert result2.chunks_indexed == 0

    @pytest.mark.asyncio
    async def test_index_incremental_reindexes_changed(
        self, indexer: Indexer, sample_project: Path
    ):
        """Incremental index reindexes changed files."""
        # First index
        await run_index(indexer, sample_project)

        # Modify a file
        import asyncio

        await asyncio.sleep(0.01)  # Ensure mtime changes
        (sample_project / "main.py").write_text('''"""Modified module."""

def new_function():
    pass
''')

        # Second index
        result = await run_index(indexer, sample_project, force=False)

        # Only main.py should be re-indexed
        assert result.files_indexed == 1

    @pytest.mark.asyncio
    async def test_index_force_reindexes_all(self, indexer: Indexer, sample_project: Path):
        """Force index reindexes all files."""
        # First index
        await run_index(indexer, sample_project)

        # Force re-index
        result = await run_index(indexer, sample_project, force=True)

        # All files should be re-indexed
        assert result.files_indexed == 2

    @pytest.mark.asyncio
    async def test_index_handles_empty_project(
        self,
        test_settings: Settings,
        embedder: Embedder,
        chunker: PythonChunker,
        tmp_path: Path,
    ):
        """Handles project with no Python files."""
        empty_project = tmp_path / "empty"
        empty_project.mkdir()
        (empty_project / "readme.md").write_text("# Empty")

        index_path = get_index_path(test_settings, empty_project)
        index_path.mkdir(parents=True, exist_ok=True)
        connection = LanceDBConnection(index_path)
        store = LanceDBVectorStore(connection)

        indexer = Indexer(
            settings=test_settings,
            embedder=embedder,
            store=store,
            chunker=chunker,
            cache_dir=index_path,
        )
        result = await run_index(indexer, empty_project)

        assert result.files_indexed == 0
        assert result.chunks_indexed == 0

    @pytest.mark.asyncio
    async def test_index_handles_syntax_errors(
        self,
        test_settings: Settings,
        embedder: Embedder,
        chunker: PythonChunker,
        tmp_path: Path,
    ):
        """Handles files with syntax errors gracefully."""
        project = tmp_path / "project_with_errors"
        project.mkdir()

        (project / "valid.py").write_text("def foo(): pass")
        (project / "broken.py").write_text("def broken(: pass")  # Syntax error

        index_path = get_index_path(test_settings, project)
        index_path.mkdir(parents=True, exist_ok=True)
        connection = LanceDBConnection(index_path)
        store = LanceDBVectorStore(connection)

        indexer = Indexer(
            settings=test_settings,
            embedder=embedder,
            store=store,
            chunker=chunker,
            cache_dir=index_path,
        )
        result = await run_index(indexer, project)

        # Should index valid file, skip broken one
        assert result.files_indexed >= 1

    @pytest.mark.asyncio
    async def test_get_index_status(self, indexer: Indexer, sample_project: Path):
        """Can get index status for a project."""
        # Before indexing
        status = indexer.get_status(sample_project)
        assert status.is_indexed is False

        # After indexing
        await run_index(indexer, sample_project)
        status = indexer.get_status(sample_project)

        assert status.is_indexed is True
        assert status.files_count > 0
        assert status.chunks_count > 0

    @pytest.mark.asyncio
    async def test_index_removes_deleted_files(self, indexer: Indexer, sample_project: Path):
        """Removes chunks for deleted files on re-index."""
        # Initial index
        await run_index(indexer, sample_project)
        status1 = indexer.get_status(sample_project)

        # Delete a file
        (sample_project / "utils.py").unlink()

        # Re-index
        await run_index(indexer, sample_project, force=False)
        status2 = indexer.get_status(sample_project)

        # Should have fewer files/chunks
        assert status2.files_count < status1.files_count

    @pytest.mark.asyncio
    async def test_chunk_files_returns_chunks(self, indexer: Indexer, sample_project: Path):
        """chunk_files() returns extracted chunks from files."""
        files = indexer.scan_files(sample_project)
        chunks = await indexer.chunk_files(files)

        assert len(chunks) > 0
        assert all(c.content for c in chunks)


class TestIndexerWithMocks:
    """Tests for Indexer using mock dependencies for isolation."""

    @pytest.fixture
    def mock_chunker_with_data(self):
        from semantic_code_mcp.models import Chunk, ChunkType

        mock = MagicMock()
        mock.chunk_file.return_value = [
            Chunk(
                file_path="/test.py",
                line_start=1,
                line_end=5,
                content="def test(): pass",
                chunk_type=ChunkType.FUNCTION,
                name="test",
            )
        ]
        return mock

    def test_indexer_uses_injected_chunker(
        self,
        test_settings: Settings,
        mock_embedder,
        mock_store,
        mock_chunker_with_data,
        tmp_path: Path,
    ):
        """Indexer uses the injected chunker."""
        indexer = Indexer(
            settings=test_settings,
            embedder=mock_embedder,
            store=mock_store,
            chunker=mock_chunker_with_data,
        )

        # Create a test file
        project = tmp_path / "mock_project"
        project.mkdir()
        (project / "test.py").write_text("def test(): pass")

        # Scan files will find the file
        files = indexer.scan_files(project)
        assert len(files) == 1

        # Verify chunker attribute is set
        assert indexer.chunker is mock_chunker_with_data
