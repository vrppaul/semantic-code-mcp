"""Dependency injection container.

Shares expensive resources (model, DB connections) across requests.
Lazy: nothing is loaded until first use. Configurable: call configure()
to override default settings (e.g. in tests).
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from semantic_code_mcp.chunkers.base import BaseTreeSitterChunker
from semantic_code_mcp.chunkers.composite import CompositeChunker
from semantic_code_mcp.chunkers.python import PythonChunker
from semantic_code_mcp.chunkers.rust import RustChunker
from semantic_code_mcp.config import Settings, get_index_path, get_settings
from semantic_code_mcp.embedder import Embedder
from semantic_code_mcp.indexer import Indexer
from semantic_code_mcp.services.index_service import IndexService
from semantic_code_mcp.services.search_service import SearchService
from semantic_code_mcp.storage.lancedb import LanceDBConnection, LanceDBVectorStore

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

log = structlog.get_logger()


class Container:
    """Shares expensive resources, creates fresh lightweight instances per request.

    Caching strategy:
    - Model: session-scoped (expensive to load, stateless)
    - Embedder: session-scoped (wraps model, stateless)
    - Connections: per-project (DB handle)
    - Stores: per-project (wraps connection, lightweight)
    - Chunker, Indexer, services: created fresh (cheap, stateless)
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._connections: dict[str, LanceDBConnection] = {}
        self._stores: dict[str, LanceDBVectorStore] = {}

    @cached_property
    def model(self) -> SentenceTransformer:
        """Lazy-load the SentenceTransformer model on first access."""
        # Lazy: sentence-transformers pulls in torch (~4s); defer until first use
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        log.info("loading_embedding_model", model=self.settings.embedding_model)
        model = SentenceTransformer(self.settings.embedding_model)
        log.info("embedding_model_loaded", model=self.settings.embedding_model)
        return model

    @cached_property
    def embedder(self) -> Embedder:
        """Lazy-load the embedder (wraps the shared model)."""
        return Embedder(self.model)

    def _get_connection(self, project_path: Path) -> LanceDBConnection:
        index_path = get_index_path(self.settings, project_path)
        key = str(index_path)
        if key not in self._connections:
            index_path.mkdir(parents=True, exist_ok=True)
            self._connections[key] = LanceDBConnection(index_path)
        return self._connections[key]

    def get_store(self, project_path: Path) -> LanceDBVectorStore:
        """Get or create a cached vector store for a project."""
        key = str(get_index_path(self.settings, project_path))
        if key not in self._stores:
            self._stores[key] = LanceDBVectorStore(self._get_connection(project_path))
        return self._stores[key]

    def get_chunkers(self) -> list[BaseTreeSitterChunker]:
        """All language-specific chunkers. Add new languages here."""
        return [PythonChunker(), RustChunker()]

    def create_chunker(self) -> CompositeChunker:
        """Create a CompositeChunker from all registered language chunkers."""
        return CompositeChunker(self.get_chunkers())

    def create_index_service(self, project_path: Path) -> IndexService:
        """Create an IndexService wired to cached store/embedder."""
        indexer = Indexer(embedder=self.embedder, store=self.get_store(project_path))
        return IndexService(
            settings=self.settings,
            indexer=indexer,
            chunker=self.create_chunker(),
            cache_dir=get_index_path(self.settings, project_path),
        )

    def create_search_service(self, project_path: Path) -> SearchService:
        """Create a SearchService sharing store/embedder with its IndexService."""
        return SearchService(
            store=self.get_store(project_path),
            embedder=self.embedder,
            index_service=self.create_index_service(project_path),
        )


# --- Global container lifecycle ---

_container: Container | None = None


def configure(settings: Settings) -> Container:
    """Initialize the global container with explicit settings (e.g. tests)."""
    global _container
    _container = Container(settings)
    return _container


def get_container() -> Container:
    """Get the global container, auto-configuring with default Settings if needed."""
    global _container
    if _container is None:
        _container = Container(get_settings())
    return _container
