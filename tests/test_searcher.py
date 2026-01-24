"""Tests for semantic search functionality."""

from pathlib import Path

import pytest

from semantic_code_mcp.config import Settings
from semantic_code_mcp.indexer.indexer import Indexer
from semantic_code_mcp.models import SearchResult
from semantic_code_mcp.search.searcher import Searcher


class TestSearcher:
    """Tests for Searcher."""

    @pytest.fixture
    def settings(self, tmp_path: Path) -> Settings:
        """Create test settings with temp cache dir."""
        return Settings(cache_dir=tmp_path / "cache")

    @pytest.fixture
    def indexed_project(self, settings: Settings, tmp_path: Path) -> Path:
        """Create and index a sample project."""
        project = tmp_path / "project"
        project.mkdir()

        (project / "auth.py").write_text('''"""Authentication module."""

def login(username: str, password: str) -> bool:
    """Authenticate user with username and password.

    Validates credentials against the database.
    """
    return validate_credentials(username, password)

def logout(session_id: str) -> None:
    """End user session and invalidate token."""
    invalidate_session(session_id)
''')

        (project / "database.py").write_text('''"""Database operations."""

class DatabaseConnection:
    """Manages database connections."""

    def connect(self):
        """Establish connection to database."""
        pass

    def execute_query(self, sql: str):
        """Execute SQL query and return results."""
        pass
''')

        (project / "utils.py").write_text('''"""Utility functions."""

def format_date(date):
    """Format a date object to string."""
    return date.strftime("%Y-%m-%d")

def parse_json(text: str):
    """Parse JSON string into dictionary."""
    import json
    return json.loads(text)
''')

        # Index the project
        indexer = Indexer(settings)
        indexer.index(project)

        return project

    def test_create_searcher(self, settings: Settings):
        """Can create a searcher instance."""
        searcher = Searcher(settings)
        assert searcher is not None

    def test_search_returns_results(self, settings: Settings, indexed_project: Path):
        """Search returns relevant results."""
        searcher = Searcher(settings)
        results = searcher.search(indexed_project, "user authentication")

        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_results_have_scores(self, settings: Settings, indexed_project: Path):
        """Search results include relevance scores."""
        searcher = Searcher(settings)
        results = searcher.search(indexed_project, "database connection")

        assert all(hasattr(r, "score") for r in results)
        assert all(0 <= r.score <= 1 for r in results)

    def test_search_results_sorted_by_relevance(self, settings: Settings, indexed_project: Path):
        """Results are sorted by relevance (highest first)."""
        searcher = Searcher(settings)
        results = searcher.search(indexed_project, "login")

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_limit_respected(self, settings: Settings, indexed_project: Path):
        """Search respects the limit parameter."""
        searcher = Searcher(settings)
        results = searcher.search(indexed_project, "function", limit=2)

        assert len(results) <= 2

    def test_search_finds_semantically_relevant(self, settings: Settings, indexed_project: Path):
        """Finds semantically relevant results even without exact match."""
        searcher = Searcher(settings)

        # Search for "credentials" - should find login function
        results = searcher.search(indexed_project, "validate credentials")

        # The login function mentions credentials
        file_paths = [r.file_path for r in results]
        assert any("auth.py" in fp for fp in file_paths)

    def test_search_empty_index_returns_empty(self, settings: Settings, tmp_path: Path):
        """Search on non-indexed project returns empty."""
        empty_project = tmp_path / "empty_project"
        empty_project.mkdir()

        searcher = Searcher(settings)
        results = searcher.search(empty_project, "anything")

        assert results == []

    def test_search_auto_reindexes_stale_files(self, settings: Settings, indexed_project: Path):
        """Search auto-reindexes stale files before searching."""
        searcher = Searcher(settings)

        # First search
        results1 = searcher.search(indexed_project, "new feature")
        new_feature_found = any("new_feature" in r.name for r in results1)
        assert not new_feature_found  # Doesn't exist yet

        # Add a new file
        import time

        time.sleep(0.01)
        (indexed_project / "new_module.py").write_text('''"""New module."""

def new_feature():
    """A brand new feature implementation."""
    pass
''')

        # Search again - should auto-reindex and find new code
        results2 = searcher.search(indexed_project, "new feature")

        # Should find the new function
        names = [r.name for r in results2]
        assert "new_feature" in names

    def test_search_result_contains_content(self, settings: Settings, indexed_project: Path):
        """Search results include the actual code content."""
        searcher = Searcher(settings)
        results = searcher.search(indexed_project, "login")

        assert len(results) > 0
        # Find the login result
        login_result = next((r for r in results if r.name == "login"), None)
        assert login_result is not None
        assert "def login" in login_result.content

    def test_search_result_contains_location(self, settings: Settings, indexed_project: Path):
        """Search results include file path and line numbers."""
        searcher = Searcher(settings)
        results = searcher.search(indexed_project, "login")

        login_result = next((r for r in results if r.name == "login"), None)
        assert login_result is not None
        assert "auth.py" in login_result.file_path
        assert login_result.line_start > 0
        assert login_result.line_end >= login_result.line_start
