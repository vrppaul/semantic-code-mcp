"""Application factory — the composition root.

Creates and wires settings, logging, container, and the MCP server.
"""

from mcp.server.fastmcp import FastMCP

from semantic_code_mcp.config import get_settings
from semantic_code_mcp.container import configure as configure_container
from semantic_code_mcp.logging import configure_logging
from semantic_code_mcp.profiling import configure_profiling


def create_app() -> FastMCP:
    """Create the fully-configured MCP application."""
    settings = get_settings()
    configure_logging(debug=settings.debug)
    configure_profiling(enabled=settings.profile)
    configure_container(settings)

    # Import tools after bootstrap so they can use get_container()
    from semantic_code_mcp.server import mcp  # noqa: PLC0415

    return mcp
