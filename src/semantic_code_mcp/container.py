"""Dependency injection container.

Shares expensive resources (model, DB connections) across requests.
Lazy: nothing is loaded until first use. Configurable: call configure()
to override default settings (e.g. in tests).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from semantic_code_mcp.config import Settings, get_index_path, get_settings
from semantic_code_mcp.indexer.chunker import PythonChunker
from semantic_code_mcp.indexer.embedder import Embedder
from semantic_code_mcp.indexer.indexer import Indexer
from semantic_code_mcp.services.index_service import IndexService
from semantic_code_mcp.services.search_service import SearchService
from semantic_code_mcp.storage.lancedb import LanceDBConnection, LanceDBVectorStore

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

log = structlog.get_logger()


class Container:
    """Shares expensive resources, creates fresh lightweight instances per request."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: SentenceTransformer | None = None
        self._connections: dict[str, LanceDBConnection] = {}

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the SentenceTransformer model on first access."""
        if self._model is None:
            # Lazy: sentence-transformers pulls in torch (~4s); defer until first use
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            log.info("loading_embedding_model", model=self.settings.embedding_model)
            self._model = SentenceTransformer(self.settings.embedding_model)
            log.info("embedding_model_loaded", model=self.settings.embedding_model)
        return self._model

    def _get_connection(self, project_path: Path) -> LanceDBConnection:
        index_path = get_index_path(self.settings, project_path)
        key = str(index_path)
        if key not in self._connections:
            index_path.mkdir(parents=True, exist_ok=True)
            self._connections[key] = LanceDBConnection(index_path)
        return self._connections[key]

    def create_store(self, project_path: Path) -> LanceDBVectorStore:
        return LanceDBVectorStore(self._get_connection(project_path))

    def create_embedder(self) -> Embedder:
        return Embedder(self.model)

    def create_chunker(self) -> PythonChunker:
        return PythonChunker()

    def create_indexer(self, project_path: Path) -> Indexer:
        store = self.create_store(project_path)
        cache_dir = get_index_path(self.settings, project_path)
        return Indexer(
            settings=self.settings,
            embedder=self.create_embedder(),
            store=store,
            chunker=self.create_chunker(),
            cache_dir=cache_dir,
        )

    def create_index_service(self, project_path: Path) -> IndexService:
        return IndexService(self.create_indexer(project_path))

    def create_search_service(self, project_path: Path) -> SearchService:
        indexer = self.create_indexer(project_path)
        index_service = IndexService(indexer)
        return SearchService(
            store=indexer.store,
            embedder=indexer.embedder,
            index_service=index_service,
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
