"""Fixtures shared by the tests."""

import os
from pathlib import Path

import pytest

from semantic_code_mcp.config import Settings
from semantic_code_mcp.embedding.embedder import OnnxEmbedder
from tests.stores import STORE_CLASSES, StoreClass

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


@pytest.fixture(autouse=True)
def git_without_the_machine_s_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Git as a fresh install has it: no user or system setting changes what it answers."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


@pytest.fixture(scope="session")
def embedder() -> OnnxEmbedder:
    """The real embedding model, loaded once for the whole session on first use."""
    return OnnxEmbedder(MODEL_NAME)


@pytest.fixture(params=STORE_CLASSES)
def store_class(request: pytest.FixtureRequest) -> StoreClass:
    """Each store in turn, as the class that builds one on an index's directory."""
    return request.param


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Settings whose indexes live in the test's temporary directory."""
    return Settings(cache_dir=tmp_path / "cache")


@pytest.fixture
def store_settings(tmp_path: Path, store_class: StoreClass) -> Settings:
    """Settings whose indexes live in the test's temporary directory, in each store in turn."""
    return Settings(cache_dir=tmp_path / "cache", index_store=store_class.LAYOUT.store_kind)


@pytest.fixture
def sample_project(tmp_path: Path) -> Path:
    """A project of two Python files: two functions, and a class with one method."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text(GREETING_MODULE)
    (project / "utils.py").write_text(HELPER_MODULE)
    return project
