"""The services of each project the server has been asked about, built once and kept."""

from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from semantic_code_mcp.config import Settings, get_index_path
from semantic_code_mcp.discovery import SourceDiscovery
from semantic_code_mcp.domain.interfaces import Chunker, Embedder, IndexStore
from semantic_code_mcp.domain.models import IndexIdentity, VectorIdentity
from semantic_code_mcp.services.indexing import IndexingService
from semantic_code_mcp.services.search import SearchService, SimilarityFloors
from semantic_code_mcp.services.vectors import VectorService

# Opens the store of a project's index of one identity, kept under the project's index
# directory; the composition root says which store
type StoreOpener = Callable[[Path, IndexIdentity], IndexStore]


class ProjectLocation(BaseModel):
    """Where a directory a tool was asked about lies: the root of its project, which keeps
    the one index, the directory under that root, None when it is the root, and whether a
    git repository names the root a project."""

    model_config = ConfigDict(frozen=True)

    root: Path
    directory: Path | None
    is_repository: bool

    @property
    def absolute_directory(self) -> Path:
        """The directory itself, from the filesystem's root: the project's root, or the
        directory under it."""
        return self.root if self.directory is None else self.root / self.directory


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
    The chunker names the identity of every index this server keeps, and the embedding
    model, shared by every project, that of the vectors it keeps in them.
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

    def locate(self, directory: Path) -> ProjectLocation:
        """Where a directory lies: in the git work tree around it, whose root is its
        project's, so a repository keeps one index whichever directory a tool names. Asks
        git: called off the event loop."""
        root = self._discovery.project_root_of(directory)
        below = None if directory == root.path else directory.relative_to(root.path)
        return ProjectLocation(root=root.path, directory=below, is_repository=root.is_repository)

    def services_for(self, project_root: Path) -> ProjectServices:
        """The services of a project, built on its first request."""
        index_path = get_index_path(self._settings, project_root)
        services = self._services_by_index.get(index_path)
        if services is None:
            services = self._build(index_path)
            self._services_by_index[index_path] = services
        return services

    def _build(self, index_path: Path) -> ProjectServices:
        store = self._open_store(index_path, IndexIdentity(chunker_version=self._chunker.version))
        model = vector_identity(self._embedder)
        indexing = IndexingService(
            discovery=self._discovery, chunker=self._chunker, store=store, model=model
        )
        return ProjectServices(
            indexing=indexing,
            search=SearchService(
                store=store,
                embedder=self._embedder,
                model=model,
                indexing=indexing,
                vectors=VectorService(embedder=self._embedder, store=store, model=model),
                floors=similarity_floors(self._settings),
                mode=self._settings.search_mode,
            ),
        )


def vector_identity(embedder: Embedder) -> VectorIdentity:
    """What makes the vectors an embedder gives."""
    return VectorIdentity(
        model_name=embedder.model_name,
        model_revision=embedder.revision,
        embedder_version=embedder.version,
    )


def similarity_floors(settings: Settings) -> SimilarityFloors:
    """The floors a search holds its hits to, as the settings give them."""
    return SimilarityFloors(
        meaning_only=settings.min_similarity,
        words_match=settings.min_similarity_when_words_match,
    )
