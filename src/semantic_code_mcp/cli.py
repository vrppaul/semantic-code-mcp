"""Command-line interface for semantic-code-mcp."""

import structlog

from semantic_code_mcp.app import create_app

log = structlog.get_logger()


def main() -> None:
    """Entry point for the MCP server."""
    app = create_app()
    log.info("starting_mcp_server", name=app.name)
    app.run()
