"""Tests for indexer orchestration."""

from pathlib import Path

import pytest

from semantic_code_mcp.config import Settings
from semantic_code_mcp.indexer.indexer import Indexer


class TestIndexer:
    """Tests for Indexer orchestration."""

    @pytest.fixture
    def settings(self, tmp_path: Path) -> Settings:
        """Create test settings with temp cache dir."""
        return Settings(cache_dir=tmp_path / "cache")

    @pytest.fixture
    def sample_project(self, tmp_path: Path) -> Path:
        """Create a sample Python project for testing."""
        project = tmp_path / "project"
        project.mkdir()

        # Create a simple Python file
        (project / "main.py").write_text('''"""Main module."""

def greet(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}!"

def farewell(name: str) -> str:
    """Say goodbye."""
    return f"Goodbye, {name}!"
''')

        # Create another file
        (project / "utils.py").write_text('''"""Utility functions."""

class Helper:
    """A helper class."""

    def assist(self):
        """Provide assistance."""
        pass
''')

        return project

    def test_create_indexer(self, settings: Settings):
        """Can create an indexer instance."""
        indexer = Indexer(settings)
        assert indexer is not None

    def test_scan_finds_python_files(self, settings: Settings, sample_project: Path):
        """Scans directory and finds Python files."""
        indexer = Indexer(settings)
        files = indexer.scan_files(sample_project)

        assert len(files) == 2
        filenames = {Path(f).name for f in files}
        assert "main.py" in filenames
        assert "utils.py" in filenames

    def test_scan_ignores_non_python(self, settings: Settings, tmp_path: Path):
        """Ignores non-Python files."""
        project = tmp_path / "project"
        project.mkdir()
        (project / "code.py").write_text("# Python")
        (project / "readme.md").write_text("# Readme")
        (project / "data.json").write_text("{}")

        indexer = Indexer(settings)
        files = indexer.scan_files(project)

        assert len(files) == 1
        assert "code.py" in files[0]

    def test_scan_respects_ignore_patterns(self, settings: Settings, tmp_path: Path):
        """Respects ignore patterns from settings."""
        project = tmp_path / "project"
        project.mkdir()
        (project / "code.py").write_text("# Code")

        # Create ignored directories
        venv = project / ".venv"
        venv.mkdir()
        (venv / "lib.py").write_text("# Venv lib")

        pycache = project / "__pycache__"
        pycache.mkdir()
        (pycache / "code.cpython-311.pyc").write_text("# Compiled")

        indexer = Indexer(settings)
        files = indexer.scan_files(project)

        # Should only find code.py, not files in ignored dirs
        assert len(files) == 1
        assert "code.py" in files[0]

    def test_scan_respects_gitignore(self, settings: Settings, tmp_path: Path):
        """Respects .gitignore patterns."""
        project = tmp_path / "project"
        project.mkdir()

        (project / ".gitignore").write_text("ignored/\n*.generated.py\n")
        (project / "code.py").write_text("# Code")
        (project / "model.generated.py").write_text("# Generated")

        ignored = project / "ignored"
        ignored.mkdir()
        (ignored / "secret.py").write_text("# Secret")

        # Settings with gitignore enabled (default)
        indexer = Indexer(settings)
        files = indexer.scan_files(project)

        assert len(files) == 1
        assert "code.py" in files[0]

    def test_index_project_creates_chunks(self, settings: Settings, sample_project: Path):
        """Index creates chunks in vector store."""
        indexer = Indexer(settings)
        result = indexer.index(sample_project)

        assert result.files_indexed > 0
        assert result.chunks_indexed > 0
        assert result.files_indexed == 2  # main.py and utils.py

    def test_index_incremental_skips_unchanged(self, settings: Settings, sample_project: Path):
        """Incremental index skips unchanged files."""
        indexer = Indexer(settings)

        # First index
        indexer.index(sample_project)

        # Second index without changes
        result2 = indexer.index(sample_project, force=False)

        # No files should be re-indexed
        assert result2.files_indexed == 0
        assert result2.chunks_indexed == 0

    def test_index_incremental_reindexes_changed(self, settings: Settings, sample_project: Path):
        """Incremental index reindexes changed files."""
        indexer = Indexer(settings)

        # First index
        indexer.index(sample_project)

        # Modify a file
        import time

        time.sleep(0.01)  # Ensure mtime changes
        (sample_project / "main.py").write_text('''"""Modified module."""

def new_function():
    pass
''')

        # Second index
        result = indexer.index(sample_project, force=False)

        # Only main.py should be re-indexed
        assert result.files_indexed == 1

    def test_index_force_reindexes_all(self, settings: Settings, sample_project: Path):
        """Force index reindexes all files."""
        indexer = Indexer(settings)

        # First index
        indexer.index(sample_project)

        # Force re-index
        result = indexer.index(sample_project, force=True)

        # All files should be re-indexed
        assert result.files_indexed == 2

    def test_index_handles_empty_project(self, settings: Settings, tmp_path: Path):
        """Handles project with no Python files."""
        empty_project = tmp_path / "empty"
        empty_project.mkdir()
        (empty_project / "readme.md").write_text("# Empty")

        indexer = Indexer(settings)
        result = indexer.index(empty_project)

        assert result.files_indexed == 0
        assert result.chunks_indexed == 0

    def test_index_handles_syntax_errors(self, settings: Settings, tmp_path: Path):
        """Handles files with syntax errors gracefully."""
        project = tmp_path / "project"
        project.mkdir()

        (project / "valid.py").write_text("def foo(): pass")
        (project / "broken.py").write_text("def broken(: pass")  # Syntax error

        indexer = Indexer(settings)
        result = indexer.index(project)

        # Should index valid file, skip broken one
        assert result.files_indexed >= 1

    def test_get_index_status(self, settings: Settings, sample_project: Path):
        """Can get index status for a project."""
        indexer = Indexer(settings)

        # Before indexing
        status = indexer.get_status(sample_project)
        assert status.is_indexed is False

        # After indexing
        indexer.index(sample_project)
        status = indexer.get_status(sample_project)

        assert status.is_indexed is True
        assert status.files_count > 0
        assert status.chunks_count > 0

    def test_index_result_has_timing(self, settings: Settings, sample_project: Path):
        """Index result includes timing information."""
        indexer = Indexer(settings)
        result = indexer.index(sample_project)

        assert result.duration_seconds > 0

    def test_index_removes_deleted_files(self, settings: Settings, sample_project: Path):
        """Removes chunks for deleted files on re-index."""
        indexer = Indexer(settings)

        # Initial index
        indexer.index(sample_project)
        status1 = indexer.get_status(sample_project)

        # Delete a file
        (sample_project / "utils.py").unlink()

        # Re-index
        indexer.index(sample_project, force=False)
        status2 = indexer.get_status(sample_project)

        # Should have fewer files/chunks
        assert status2.files_count < status1.files_count
