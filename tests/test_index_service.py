"""Tests for IndexService (scan, detect changes, chunk, full pipeline)."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from semantic_code_mcp.chunkers.composite import CompositeChunker
from semantic_code_mcp.chunkers.python import PythonChunker
from semantic_code_mcp.config import Settings, get_index_path
from semantic_code_mcp.embedder import Embedder
from semantic_code_mcp.indexer import Indexer
from semantic_code_mcp.services.index_service import IndexService
from semantic_code_mcp.storage.lancedb import LanceDBConnection, LanceDBVectorStore


class TestIndexServiceIntegration:
    """Integration tests using real embedder, LanceDB store, PythonChunker."""

    @pytest.fixture
    def chunker(self) -> CompositeChunker:
        return CompositeChunker([PythonChunker()])

    @pytest.fixture
    def store(self, test_settings: Settings, sample_project: Path) -> LanceDBVectorStore:
        index_path = get_index_path(test_settings, sample_project)
        index_path.mkdir(parents=True, exist_ok=True)
        connection = LanceDBConnection(index_path)
        return LanceDBVectorStore(connection)

    @pytest.fixture
    def index_service(
        self,
        test_settings: Settings,
        embedder: Embedder,
        store: LanceDBVectorStore,
        chunker: CompositeChunker,
        sample_project: Path,
    ) -> IndexService:
        cache_dir = get_index_path(test_settings, sample_project)
        indexer = Indexer(embedder=embedder, store=store)
        return IndexService(
            settings=test_settings,
            indexer=indexer,
            chunker=chunker,
            cache_dir=cache_dir,
        )

    def test_scan_finds_python_files(self, index_service: IndexService, sample_project: Path):
        """Scans directory and finds Python files."""
        files = index_service.scan_files(sample_project)

        assert len(files) == 2
        filenames = {Path(f).name for f in files}
        assert "main.py" in filenames
        assert "utils.py" in filenames

    def test_scan_ignores_non_python(self, index_service: IndexService, tmp_path: Path):
        """Ignores non-Python files."""
        project = tmp_path / "project2"
        project.mkdir()
        (project / "code.py").write_text("# Python")
        (project / "readme.md").write_text("# Readme")
        (project / "data.json").write_text("{}")

        files = index_service.scan_files(project)

        assert len(files) == 1
        assert "code.py" in files[0]

    def test_scan_respects_ignore_patterns(self, index_service: IndexService, tmp_path: Path):
        """Respects ignore patterns from settings."""
        project = tmp_path / "project3"
        project.mkdir()
        (project / "code.py").write_text("# Code")

        venv = project / ".venv"
        venv.mkdir()
        (venv / "lib.py").write_text("# Venv lib")

        pycache = project / "__pycache__"
        pycache.mkdir()
        (pycache / "code.cpython-311.pyc").write_text("# Compiled")

        files = index_service.scan_files(project)

        assert len(files) == 1
        assert "code.py" in files[0]

    def test_scan_respects_gitignore(self, index_service: IndexService, tmp_path: Path):
        """Respects .gitignore patterns."""
        project = tmp_path / "project4"
        project.mkdir()

        (project / ".gitignore").write_text("ignored/\n*.generated.py\n")
        (project / "code.py").write_text("# Code")
        (project / "model.generated.py").write_text("# Generated")

        ignored = project / "ignored"
        ignored.mkdir()
        (ignored / "secret.py").write_text("# Secret")

        files = index_service.scan_files(project)

        assert len(files) == 1
        assert "code.py" in files[0]

    def test_detect_changes_force(self, index_service: IndexService, sample_project: Path):
        """Force mode returns all files for indexing."""
        files = index_service.scan_files(sample_project)
        plan = index_service.detect_changes(sample_project, files, force=True)

        assert plan.files_to_index == files
        assert plan.files_to_delete == []
        assert plan.has_work

    def test_detect_changes_no_prior_index(self, index_service: IndexService, sample_project: Path):
        """All files are new when no prior index exists."""
        files = index_service.scan_files(sample_project)
        plan = index_service.detect_changes(sample_project, files, force=False)

        assert len(plan.files_to_index) == 2
        assert plan.has_work

    @pytest.mark.asyncio
    async def test_index_creates_chunks(self, index_service: IndexService, sample_project: Path):
        """Full index creates chunks in vector store."""
        result = await index_service.index(sample_project)

        assert result.files_indexed > 0
        assert result.chunks_indexed > 0
        assert result.files_indexed == 2

    @pytest.mark.asyncio
    async def test_index_incremental_skips_unchanged(
        self, index_service: IndexService, sample_project: Path
    ):
        """Incremental index skips unchanged files."""
        await index_service.index(sample_project)

        result2 = await index_service.index(sample_project, force=False)

        assert result2.files_indexed == 0
        assert result2.chunks_indexed == 0

    @pytest.mark.asyncio
    async def test_index_incremental_reindexes_changed(
        self, index_service: IndexService, sample_project: Path
    ):
        """Incremental index reindexes changed files."""
        await index_service.index(sample_project)

        import asyncio

        await asyncio.sleep(0.01)
        modified = '"""Modified module."""\n\ndef new_function():\n    pass\n'
        (sample_project / "main.py").write_text(modified)

        result = await index_service.index(sample_project, force=False)

        assert result.files_indexed == 1

    @pytest.mark.asyncio
    async def test_index_force_reindexes_all(
        self, index_service: IndexService, sample_project: Path
    ):
        """Force index reindexes all files."""
        await index_service.index(sample_project)

        result = await index_service.index(sample_project, force=True)

        assert result.files_indexed == 2

    @pytest.mark.asyncio
    async def test_index_handles_empty_project(
        self,
        test_settings: Settings,
        embedder: Embedder,
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

        indexer = Indexer(embedder=embedder, store=store)
        svc = IndexService(
            settings=test_settings,
            indexer=indexer,
            chunker=CompositeChunker([PythonChunker()]),
            cache_dir=index_path,
        )
        result = await svc.index(empty_project)

        assert result.files_indexed == 0
        assert result.chunks_indexed == 0

    @pytest.mark.asyncio
    async def test_index_handles_syntax_errors(
        self,
        test_settings: Settings,
        embedder: Embedder,
        tmp_path: Path,
    ):
        """Handles files with syntax errors gracefully."""
        project = tmp_path / "project_with_errors"
        project.mkdir()
        (project / "valid.py").write_text("def foo(): pass")
        (project / "broken.py").write_text("def broken(: pass")

        index_path = get_index_path(test_settings, project)
        index_path.mkdir(parents=True, exist_ok=True)
        connection = LanceDBConnection(index_path)
        store = LanceDBVectorStore(connection)

        indexer = Indexer(embedder=embedder, store=store)
        svc = IndexService(
            settings=test_settings,
            indexer=indexer,
            chunker=CompositeChunker([PythonChunker()]),
            cache_dir=index_path,
        )
        result = await svc.index(project)

        assert result.files_indexed >= 1

    @pytest.mark.asyncio
    async def test_get_index_status(self, index_service: IndexService, sample_project: Path):
        """Can get index status for a project."""
        status = index_service.get_status(sample_project)
        assert status.is_indexed is False

        await index_service.index(sample_project)
        status = index_service.get_status(sample_project)

        assert status.is_indexed is True
        assert status.files_count > 0
        assert status.chunks_count > 0

    @pytest.mark.asyncio
    async def test_index_removes_deleted_files(
        self, index_service: IndexService, sample_project: Path
    ):
        """Removes chunks for deleted files on re-index."""
        await index_service.index(sample_project)
        status1 = index_service.get_status(sample_project)

        (sample_project / "utils.py").unlink()

        await index_service.index(sample_project, force=False)
        status2 = index_service.get_status(sample_project)

        assert status2.files_count < status1.files_count

    @pytest.mark.asyncio
    async def test_chunk_files_returns_chunks(
        self, index_service: IndexService, sample_project: Path
    ):
        """chunk_files() returns extracted chunks from files."""
        files = index_service.scan_files(sample_project)
        chunks = await index_service.chunk_files(files)

        assert len(chunks) > 0
        assert all(c.content for c in chunks)


class TestIndexServiceWithMocks:
    """Tests for IndexService using mock Indexer."""

    @pytest.fixture
    def mock_indexer(self):
        indexer = MagicMock()
        indexer.clear_store.return_value = None
        indexer.get_store_stats.return_value = ([], 0)

        async def fake_embed_and_store(plan, chunks):
            pass

        indexer.embed_and_store = fake_embed_and_store
        return indexer

    @pytest.fixture
    def index_service(self, test_settings, mock_indexer, tmp_path) -> IndexService:
        return IndexService(
            settings=test_settings,
            indexer=mock_indexer,
            chunker=CompositeChunker([PythonChunker()]),
            cache_dir=tmp_path / "cache",
        )

    @pytest.mark.asyncio
    async def test_index_returns_result_with_duration(self, index_service, tmp_path):
        """index() returns IndexResult with duration_seconds set."""
        project = tmp_path / "proj"
        project.mkdir()
        (project / "a.py").write_text("def foo(): pass")

        result = await index_service.index(project, force=True)

        assert result.files_indexed == 1
        assert result.chunks_indexed == 1
        assert result.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_index_no_work_returns_zeros(self, index_service, tmp_path):
        """index() returns zeros when no work is needed."""
        project = tmp_path / "proj2"
        project.mkdir()
        # No .py files -> no work

        result = await index_service.index(project)

        assert result.files_indexed == 0
        assert result.chunks_indexed == 0
        assert result.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_index_calls_progress(self, index_service, tmp_path):
        """index() calls progress callback."""
        project = tmp_path / "proj3"
        project.mkdir()
        (project / "a.py").write_text("def foo(): pass")

        progress_calls: list[tuple[float, float, str]] = []

        async def track_progress(progress: float, total: float, message: str) -> None:
            progress_calls.append((progress, total, message))

        await index_service.index(project, on_progress=track_progress)

        assert len(progress_calls) >= 3
        assert "Scanning" in progress_calls[0][2]

    @pytest.mark.asyncio
    async def test_index_force_passes_through(self, index_service, mock_indexer, tmp_path):
        """force=True clears the store via indexer.clear_store()."""
        project = tmp_path / "proj4"
        project.mkdir()
        (project / "a.py").write_text("def foo(): pass")

        await index_service.index(project, force=True)

        mock_indexer.clear_store.assert_called_once()
