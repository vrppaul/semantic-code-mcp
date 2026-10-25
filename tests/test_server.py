"""End-to-end tests: the MCP tools driven through a client, as an agent calls them."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from mcp import ClientSession
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult

from semantic_code_mcp.app import create_app
from semantic_code_mcp.config import Settings, configure_settings

GREETING_MODULE = '''def greet(name: str) -> str:
    """Greet a person by their name."""
    return f"Hello, {name}!"
'''

CONFIG_MODULE = '''import json


def load_config(path: str) -> dict:
    """Read the application configuration from a JSON file."""
    with open(path) as config_file:
        return json.load(config_file)
'''

RETRY_FUNCTION = '''

def retry_with_backoff(operation, attempts: int):
    """Call an operation again after a growing pause each time it fails."""
    return operation()
'''


@asynccontextmanager
async def connected_client(cache_dir: Path) -> AsyncGenerator[ClientSession]:
    """A client session to a freshly built server whose indexes live in cache_dir."""
    configure_settings(Settings(cache_dir=cache_dir))
    app = create_app()
    async with create_connected_server_and_client_session(app) as client:
        yield client


def tool_payload(result: CallToolResult) -> dict[str, Any]:
    """The structured answer of a tool. FastMCP wraps a union return type in `result`."""
    assert result.structuredContent is not None
    return result.structuredContent["result"]


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "greeting.py").write_text(GREETING_MODULE)
    (root / "config.py").write_text(CONFIG_MODULE)
    return root


async def test_search_returns_the_function_the_query_describes(project: Path, tmp_path: Path):
    async with connected_client(tmp_path / "cache") as client:
        result = await client.call_tool(
            "search_code", {"query": "say hello to someone", "project_path": str(project)}
        )

    first_hit = tool_payload(result)["results"][0]
    assert (Path(first_hit["file_path"]).name, first_hit["name"]) == ("greeting.py", "greet")


async def test_index_codebase_reports_what_it_indexed(project: Path, tmp_path: Path):
    async with connected_client(tmp_path / "cache") as client:
        result = await client.call_tool("index_codebase", {"project_path": str(project)})

    report = tool_payload(result)
    assert (report["files_indexed"], report["chunks_indexed"]) == (2, 2)


async def test_index_status_counts_the_indexed_project(project: Path, tmp_path: Path):
    async with connected_client(tmp_path / "cache") as client:
        await client.call_tool("index_codebase", {"project_path": str(project)})
        result = await client.call_tool("index_status", {"project_path": str(project)})

    status = tool_payload(result)
    assert (status["is_indexed"], status["files_count"], status["chunks_count"]) == (True, 2, 2)


async def test_search_sees_a_function_added_after_indexing(project: Path, tmp_path: Path):
    async with connected_client(tmp_path / "cache") as client:
        await client.call_tool("index_codebase", {"project_path": str(project)})
        with (project / "config.py").open("a") as config_module:
            config_module.write(RETRY_FUNCTION)
        result = await client.call_tool(
            "search_code",
            {"query": "try again after waiting longer", "project_path": str(project)},
        )

    first_hit = tool_payload(result)["results"][0]
    assert first_hit["name"] == "retry_with_backoff"


@pytest.mark.xfail(
    strict=True,
    reason="errors come back as a successful result until ToolError (ADR 006, D10, stage 4)",
)
async def test_search_in_a_missing_project_is_an_error(tmp_path: Path):
    async with connected_client(tmp_path / "cache") as client:
        result = await client.call_tool(
            "search_code", {"query": "anything", "project_path": str(tmp_path / "missing")}
        )

    assert result.isError
