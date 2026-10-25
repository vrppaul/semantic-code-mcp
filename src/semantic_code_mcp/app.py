"""The composition root: builds the server and everything its tools share."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import partial
from importlib.metadata import version

from mcp.server.mcpserver import MCPServer

from semantic_code_mcp.config import Settings
from semantic_code_mcp.domain.interfaces import Embedder
from semantic_code_mcp.server.profiling import ProfilingMiddleware
from semantic_code_mcp.server.tools import index_codebase, index_status, search_code
from semantic_code_mcp.services.projects import ProjectRegistry, build_chunker

SERVER_NAME = "semantic-code-mcp"
PACKAGE_NAME = "semantic-code-mcp"


def create_app(settings: Settings, embedder: Embedder) -> MCPServer[ProjectRegistry]:
    """The MCP server with its tools, sharing one registry of projects.

    Args:
        settings: Application settings.
        embedder: The embedder every project shares.
    """
    middleware = [ProfilingMiddleware(settings.profiles_dir)] if settings.profile else []
    server = MCPServer(
        SERVER_NAME,
        version=version(PACKAGE_NAME),
        lifespan=partial(project_registry_lifespan, settings=settings, embedder=embedder),
        middleware=middleware,
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
    yield ProjectRegistry(settings=settings, embedder=embedder, chunker=build_chunker())
