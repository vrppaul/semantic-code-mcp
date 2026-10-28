"""Command-line entry point: read the settings, start the server over stdio.

A setting that needs an optional extra the install lacks stops the server before it runs,
with one line naming the extra, rather than a client's bare "failed to connect" over a
traceback.
"""

import sys
from typing import assert_never

import structlog
from mcp.server.mcpserver import MCPServer

from semantic_code_mcp.app import create_app
from semantic_code_mcp.config import Settings
from semantic_code_mcp.domain.models import SearchMode
from semantic_code_mcp.embedding.embedder import OnnxEmbedder, model_loader
from semantic_code_mcp.extras import ExtraMissingError
from semantic_code_mcp.logging import configure_logging
from semantic_code_mcp.services.projects import ProjectRegistry

log = structlog.get_logger()

# The exit code of a server whose settings need an extra the install lacks
EXIT_EXTRA_MISSING = 1


def main() -> None:
    """Run the MCP server over stdio until the client hangs up or Ctrl+C."""
    settings = Settings()
    configure_logging(debug=settings.debug)
    try:
        server = build_server(settings)
    except ExtraMissingError as error:
        # One line naming the extra: its traceback would bury what to install
        log.error("extra_missing", reason=str(error))  # noqa: TRY400
        sys.exit(EXIT_EXTRA_MISSING)
    log.info("starting_mcp_server", name=server.name)
    try:
        server.run()
    except KeyboardInterrupt:
        log.info("mcp_server_interrupted")


def build_server(settings: Settings) -> MCPServer[ProjectRegistry]:
    """The server the settings describe, over the ONNX embedder, each extra they need looked
    for first."""
    require_libraries_of(settings.search_mode)
    return create_app(settings, OnnxEmbedder(settings.embedding_model))


def require_libraries_of(search_mode: SearchMode) -> None:
    """The libraries the embedder needs in a search mode, looked for now: a server set to
    search by meaning loads its model on its first search."""
    match search_mode:
        case SearchMode.words:
            return
        case SearchMode.words_and_meaning:
            model_loader()
        case unhandled:
            assert_never(unhandled)
