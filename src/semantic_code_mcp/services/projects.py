"""The services of each project the server has been asked about, built once and kept."""

import asyncio
from pathlib import Path

from semantic_code_mcp.chunkers.composite import CompositeChunker
from semantic_code_mcp.chunkers.markdown import MarkdownChunker
from semantic_code_mcp.chunkers.python import PythonChunker
from semantic_code_mcp.chunkers.rust import RustChunker
from semantic_code_mcp.config import Settings, get_index_path
from semantic_code_mcp.discovery import SourceDiscovery
from semantic_code_mcp.domain.interfaces import Chunker, Embedder
from semantic_code_mcp.services.indexing import IndexingService
from semantic_code_mcp.services.search import SearchService
from semantic_code_mcp.storage.lancedb import LanceDBVectorStore


def build_chunker() -> CompositeChunker:
    """Every language the server indexes. Add a language here."""
    return CompositeChunker([PythonChunker(), RustChunker(), MarkdownChunker()])


class ProjectServices:
    """The indexing and search of one project, over one store."""

    def __init__(self, indexing: IndexingService, search: SearchService) -> None:
        self.indexing = indexing
        self.search = search


class ProjectRegistry:
    """Builds each project's services on first use and keeps them for the session.

    Keyed by index directory, so a project reached by two spellings shares one store,
    lock and cache as long as callers resolve its path first (the tools do). The
    embedding model is shared by every project.
    """

    def __init__(self, settings: Settings, embedder: Embedder, chunker: Chunker) -> None:
        self._settings = settings
        self._embedder = embedder
        self._chunker = chunker
        self._discovery = SourceDiscovery(
            extensions=chunker.supported_extensions,
            ignore_patterns=settings.ignore_patterns,
            use_gitignore=settings.use_gitignore,
            git_timeout_seconds=settings.git_timeout_seconds,
        )
        self._services_by_index: dict[Path, ProjectServices] = {}
        self._building = asyncio.Lock()

    async def services_for(self, project_root: Path) -> ProjectServices:
        """The services of a project, built on its first request.

        Opening the store touches the disk, so it happens in a worker thread; the
        registry itself is only ever changed on the event loop.
        """
        index_path = get_index_path(self._settings, project_root)
        # Held across the build, so two first requests for one project build it once
        async with self._building:
            services = self._services_by_index.get(index_path)
            if services is None:
                services = await asyncio.to_thread(self._build, index_path)
                self._services_by_index[index_path] = services
        return services

    def _build(self, index_path: Path) -> ProjectServices:
        index_path.mkdir(parents=True, exist_ok=True)
        store = LanceDBVectorStore(index_path)
        indexing = IndexingService(
            discovery=self._discovery,
            chunker=self._chunker,
            embedder=self._embedder,
            store=store,
            cache_directory=index_path,
        )
        return ProjectServices(
            indexing=indexing,
            search=SearchService(store=store, embedder=self._embedder, indexing=indexing),
        )
