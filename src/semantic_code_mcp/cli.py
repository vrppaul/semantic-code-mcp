"""Command-line entry point: read the settings, start the server over stdio."""

import structlog

from semantic_code_mcp.app import create_app
from semantic_code_mcp.config import Settings
from semantic_code_mcp.embedding.embedder import OnnxEmbedder
from semantic_code_mcp.logging import configure_logging

log = structlog.get_logger()


def main() -> None:
    """Run the MCP server over stdio until the client hangs up or Ctrl+C."""
    settings = Settings()
    configure_logging(debug=settings.debug)
    server = create_app(settings, OnnxEmbedder(settings.embedding_model))
    log.info("starting_mcp_server", name=server.name)
    try:
        server.run()
    except KeyboardInterrupt:
        log.info("mcp_server_interrupted")
