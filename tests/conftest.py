"""Pytest configuration and fixtures."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sentence_transformers import SentenceTransformer

from semantic_code_mcp.config import Settings
from semantic_code_mcp.indexer.embedder import Embedder
from semantic_code_mcp.models import Chunk, ChunkType
from semantic_code_mcp.protocols import ChunkerProtocol, EmbedderProtocol, VectorStoreProtocol
from semantic_code_mcp.storage.lancedb import LanceDBConnection, LanceDBVectorStore

# Shared model fixtures (session-scoped to avoid repeated loading)


@pytest.fixture(scope="session")
def model() -> SentenceTransformer:
    """Load the embedding model once for the entire test session."""
    return SentenceTransformer("all-MiniLM-L6-v2")


@pytest.fixture(scope="session")
def embedder(model: SentenceTransformer) -> Embedder:
    """Create an embedder with the session-scoped model."""
    return Embedder(model)


# Settings fixtures


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Create test settings with temp cache dir."""
    return Settings(cache_dir=tmp_path / "cache")


# Protocol mock fixtures


@pytest.fixture
def mock_connection(tmp_path: Path) -> LanceDBConnection:
    """Create a real LanceDB connection for testing."""
    db_path = tmp_path / "test.lance"
    return LanceDBConnection(db_path)


@pytest.fixture
def mock_store() -> VectorStoreProtocol:
    """Create a mock VectorStore implementing the protocol."""
    mock = MagicMock(spec=VectorStoreProtocol)
    mock.count.return_value = 0
    mock.search_hybrid.return_value = []
    mock.get_indexed_files.return_value = []
    return mock


@pytest.fixture
def mock_embedder() -> EmbedderProtocol:
    """Create a mock Embedder implementing the protocol."""
    mock = MagicMock(spec=EmbedderProtocol)
    mock.embed_text.return_value = [0.1] * 384
    mock.embed_batch.return_value = [[0.1] * 384]
    return mock


@pytest.fixture
def mock_chunker() -> ChunkerProtocol:
    """Create a mock Chunker implementing the protocol."""
    mock = MagicMock(spec=ChunkerProtocol)
    mock.chunk_file.return_value = []
    return mock


# Real component fixtures for integration tests


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Create a temporary directory for the database."""
    return tmp_path / "test.lance"


@pytest.fixture
def lance_connection(temp_db_path: Path) -> LanceDBConnection:
    """Create a LanceDB connection for testing."""
    return LanceDBConnection(temp_db_path)


@pytest.fixture
def vector_store(lance_connection: LanceDBConnection) -> LanceDBVectorStore:
    """Create a VectorStore instance for testing."""
    return LanceDBVectorStore(lance_connection)


@pytest.fixture
def sample_chunk() -> Chunk:
    """Create a sample chunk for testing."""
    return Chunk(
        file_path="/path/to/file.py",
        line_start=10,
        line_end=20,
        content="def hello():\n    return 'world'",
        chunk_type=ChunkType.FUNCTION,
        name="hello",
    )


@pytest.fixture
def sample_embedding() -> list[float]:
    """Create a sample 384-dim embedding (MiniLM size)."""
    import numpy as np

    return np.random.rand(384).astype(np.float32).tolist()


# Sample project fixtures


@pytest.fixture
def sample_project(tmp_path: Path) -> Path:
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
