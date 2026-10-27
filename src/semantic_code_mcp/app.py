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
from semantic_code_mcp.domain.interfaces import Embedder
from semantic_code_mcp.domain.models import IndexIdentity
from semantic_code_mcp.server.instructions import SERVER_INSTRUCTIONS
from semantic_code_mcp.server.profiling import ProfilingMiddleware
from semantic_code_mcp.server.tools import index_codebase, index_status, search_code
from semantic_code_mcp.services.projects import ProjectRegistry
from semantic_code_mcp.storage.lancedb import LanceDBIndexStore

SERVER_NAME = "semantic-code-mcp"
PACKAGE_NAME = "semantic-code-mcp"
# How long a write waits while another server process writes the same index; a write
# takes seconds at most, embedding having happened before it
WRITE_LOCK_TIMEOUT_SECONDS = 60.0
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
    for tool in (search_code, index_codebase, index_status):
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
        open_store=open_index_store,
    )


def build_chunker() -> CompositeChunker:
    """Every language the server indexes. Add a language here."""
    return CompositeChunker([PythonChunker(), RustChunker(), MarkdownChunker()])


def open_index_store(project_directory: Path, identity: IndexIdentity) -> LanceDBIndexStore:
    """The store of a project's index of one identity, under the project's index directory."""
    return LanceDBIndexStore(
        project_directory, identity, lock_timeout_seconds=WRITE_LOCK_TIMEOUT_SECONDS
    )
