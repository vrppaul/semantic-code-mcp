"""The services of each project the server has been asked about, built once and kept."""

from collections.abc import Callable
from pathlib import Path

from semantic_code_mcp.config import Settings, get_index_path
from semantic_code_mcp.discovery import SourceDiscovery
from semantic_code_mcp.domain.interfaces import Chunker, Embedder, IndexStore
from semantic_code_mcp.services.indexing import IndexingService
from semantic_code_mcp.services.search import SearchService

# Opens the store of the index kept in a directory; the composition root says which store
type StoreOpener = Callable[[Path], IndexStore]


class ProjectServices:
    """The indexing and search of one project, over one store."""

    def __init__(self, indexing: IndexingService, search: SearchService) -> None:
        self.indexing = indexing
        self.search = search


class ProjectRegistry:
    """Builds each project's services on first use and keeps them for the session.

    Keyed by index directory, which every spelling of a project's path shares, so the
    project has one store and one set of locks. Building touches no disk — a store opens
    its connection when first used, in a worker thread — so a lookup and its insert run
    with no await between them, and two first requests cannot build a project twice.
    The embedding model is shared by every project.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        embedder: Embedder,
        chunker: Chunker,
        discovery: SourceDiscovery,
        open_store: StoreOpener,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._chunker = chunker
        self._discovery = discovery
        self._open_store = open_store
        self._services_by_index: dict[Path, ProjectServices] = {}

    def services_for(self, project_root: Path) -> ProjectServices:
        """The services of a project, built on its first request."""
        index_path = get_index_path(self._settings, project_root)
        services = self._services_by_index.get(index_path)
        if services is None:
            services = self._build(index_path)
            self._services_by_index[index_path] = services
        return services

    def _build(self, index_path: Path) -> ProjectServices:
        store = self._open_store(index_path)
        indexing = IndexingService(
            discovery=self._discovery,
            chunker=self._chunker,
            embedder=self._embedder,
            store=store,
        )
        return ProjectServices(
            indexing=indexing,
            search=SearchService(store=store, embedder=self._embedder, indexing=indexing),
        )
