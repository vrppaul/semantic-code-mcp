"""Command-line interface for semantic-code-mcp."""

from semantic_code_mcp.server import run_server


def main() -> None:
    """Entry point for the MCP server."""
    run_server()
