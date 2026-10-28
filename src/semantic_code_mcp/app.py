"""The composition root: builds the server and everything its tools share."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import partial
from importlib.metadata import version
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from semantic_code_mcp.chunkers.composite import CompositeChunker
from semantic_code_mcp.chunkers.markdown import MarkdownChunker
from semantic_code_mcp.chunkers.python import PythonChunker
from semantic_code_mcp.chunkers.rust import RustChunker
from semantic_code_mcp.config import Settings
from semantic_code_mcp.discovery import SourceDiscovery
from semantic_code_mcp.domain.interfaces import Embedder, IndexStore
from semantic_code_mcp.domain.models import IndexIdentity, StoreKind
from semantic_code_mcp.server.instructions import SERVER_INSTRUCTIONS
from semantic_code_mcp.server.profiling import ProfilingMiddleware
from semantic_code_mcp.server.tools import index_codebase, index_status, search_code
from semantic_code_mcp.services.projects import ProjectRegistry, StoreOpener
from semantic_code_mcp.storage.index_directories import IndexDirectory, IndexLayout
from semantic_code_mcp.storage.lancedb import LanceDBIndexStore
from semantic_code_mcp.storage.sqlite import SqliteIndexStore

SERVER_NAME = "semantic-code-mcp"
PACKAGE_NAME = "semantic-code-mcp"
# How long a write waits while another server process writes the same index; a write
# takes seconds at most, embedding having happened before it
WRITE_LOCK_TIMEOUT_SECONDS = 60.0
# The key of a tool's `_meta` that asks Claude Code to load the tool with the prompt, not
# behind its tool search: there, an agent spends a turn looking the tool up, and on a
# repository it did not know it took the search in 10 of 15 sessions against 15 of 15
# with the tool loaded (ADR 008, pilots 5 and 6)
ALWAYS_LOAD_KEY = "anthropic/alwaysLoad"
# The level MCPServer sets Python's own logging to, with a handler on stderr: the records
# of the libraries below it, as httpx's for each model file the hub fetches, would land
# among the server's JSON lines
LIBRARY_LOG_LEVEL = "WARNING"


def create_app(settings: Settings, embedder: Embedder) -> MCPServer[ProjectRegistry]:
    """The MCP server with its tools, sharing one registry of projects.

    Args:
        settings: Application settings.
        embedder: The embedder every project shares.
    """
    middleware = [ProfilingMiddleware(settings.profiles_dir)] if settings.profile else []
    server = MCPServer(
        SERVER_NAME,
        instructions=SERVER_INSTRUCTIONS,
        version=version(PACKAGE_NAME),
        lifespan=partial(project_registry_lifespan, settings=settings, embedder=embedder),
        middleware=middleware,
        log_level=LIBRARY_LOG_LEVEL,
    )
    server.add_tool(search_code, meta={ALWAYS_LOAD_KEY: True})
    # Indexing runs inside every search, so an agent seldom needs these: they wait behind
    # the client's tool search rather than fill every request
    for tool in (index_codebase, index_status):
        server.add_tool(tool)
    return server


@asynccontextmanager
async def project_registry_lifespan(
    _server: MCPServer[ProjectRegistry], *, settings: Settings, embedder: Embedder
) -> AsyncGenerator[ProjectRegistry]:
    """The state every request of a session shares: the projects and their services.

    The lifespan protocol hands in the server; the registry does not need it.
    """
    yield build_registry(settings, embedder)


def build_registry(settings: Settings, embedder: Embedder) -> ProjectRegistry:
    """The registry of projects, over the parts the server is built from."""
    chunker = build_chunker()
    return ProjectRegistry(
        settings=settings,
        embedder=embedder,
        chunker=chunker,
        discovery=SourceDiscovery(
            extensions=chunker.supported_extensions,
            ignore_patterns=settings.ignore_patterns,
            use_gitignore=settings.use_gitignore,
            git_timeout_seconds=settings.git_timeout_seconds,
            home_directory=Path.home(),
        ),
        open_store=store_opener(settings),
    )


def build_chunker() -> CompositeChunker:
    """Every language the server indexes. Add a language here."""
    return CompositeChunker([PythonChunker(), RustChunker(), MarkdownChunker()])


def store_opener(settings: Settings) -> StoreOpener:
    """What opens each index of a project in the store the settings pick."""
    return partial(open_index_store, store_kind=settings.index_store)


def open_index_store(
    project_directory: Path, identity: IndexIdentity, *, store_kind: StoreKind
) -> IndexStore:
    """The store of a project's index of one identity, of the kind given, built on the
    index's own directory under the project's index directory."""
    match store_kind:
        case StoreKind.sqlite:
            return SqliteIndexStore(
                index_directory(project_directory, identity, SqliteIndexStore.LAYOUT)
            )
        case StoreKind.lancedb:
            return LanceDBIndexStore(
                index_directory(project_directory, identity, LanceDBIndexStore.LAYOUT)
            )


def index_directory(
    project_directory: Path, identity: IndexIdentity, layout: IndexLayout
) -> IndexDirectory:
    """The directory of a project's index of one identity in a store's layout."""
    return IndexDirectory(
        project_directory, identity, layout, lock_timeout_seconds=WRITE_LOCK_TIMEOUT_SECONDS
    )
