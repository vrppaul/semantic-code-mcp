"""Fixtures shared by the tests."""

from pathlib import Path

import pytest

from semantic_code_mcp.config import Settings
from semantic_code_mcp.embedding import SentenceTransformerEmbedder

MODEL_NAME = "all-MiniLM-L6-v2"

GREETING_MODULE = '''"""Main module."""

def greet(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}!"

def farewell(name: str) -> str:
    """Say goodbye."""
    return f"Goodbye, {name}!"
'''

HELPER_MODULE = '''"""Utility functions."""

class Helper:
    """A helper class."""

    def assist(self):
        """Provide assistance."""
        pass
'''


@pytest.fixture(scope="session")
def embedder() -> SentenceTransformerEmbedder:
    """The real embedding model, loaded once for the whole session on first use."""
    return SentenceTransformerEmbedder(MODEL_NAME)


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Settings whose indexes live in the test's temporary directory."""
    return Settings(cache_dir=tmp_path / "cache")


@pytest.fixture
def sample_project(tmp_path: Path) -> Path:
    """A project of two Python files: two functions, and a class with one method."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text(GREETING_MODULE)
    (project / "utils.py").write_text(HELPER_MODULE)
    return project
