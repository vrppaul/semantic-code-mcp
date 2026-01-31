"""Command-line interface for semantic-code-mcp."""

import signal
import sys

import structlog

from semantic_code_mcp.app import create_app

log = structlog.get_logger()


def main() -> None:
    """Entry point for the MCP server."""
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    app = create_app()
    log.info("starting_mcp_server", name=app.name)
    app.run()
