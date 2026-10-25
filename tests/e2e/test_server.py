"""End-to-end tests: the MCP tools driven through a client, as an agent calls them."""

from pathlib import Path

import pytest
from mcp import Client
from mcp.types import CallToolResult

from semantic_code_mcp.app import create_app
from semantic_code_mcp.config import Settings
from semantic_code_mcp.embedding import SentenceTransformerEmbedder
from semantic_code_mcp.server.responses import (
    IndexCodebaseResponse,
    IndexStatusResponse,
    SearchCodeResponse,
)

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


def structured(result: CallToolResult) -> object:
    assert not result.is_error, result.content
    return result.structured_content


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "greeting.py").write_text(GREETING_MODULE)
    (root / "config.py").write_text(CONFIG_MODULE)
    return root


@pytest.fixture
def client(test_settings: Settings, embedder: SentenceTransformerEmbedder) -> Client:
    return Client(create_app(test_settings, embedder=embedder))


async def test_search_returns_the_function_the_query_describes(project: Path, client: Client):
    async with client:
        result = await client.call_tool(
            "search_code", {"query": "say hello to someone", "project_path": str(project)}
        )

    first_hit = SearchCodeResponse.model_validate(structured(result)).results[0]
    assert (Path(first_hit.file_path).name, first_hit.name) == ("greeting.py", "greet")


async def test_search_sends_debug_details_only_when_asked(project: Path, client: Client):
    async with client:
        plain = await client.call_tool(
            "search_code", {"query": "greeting", "project_path": str(project)}
        )
        detailed = await client.call_tool(
            "search_code",
            {"query": "greeting", "project_path": str(project), "include_debug": True},
        )

    assert SearchCodeResponse.model_validate(structured(plain)).debug is None
    assert SearchCodeResponse.model_validate(structured(detailed)).debug is not None


async def test_index_codebase_reports_what_it_indexed(project: Path, client: Client):
    async with client:
        result = await client.call_tool("index_codebase", {"project_path": str(project)})

    report = IndexCodebaseResponse.model_validate(structured(result))
    assert (report.files_indexed, report.chunks_indexed) == (2, 2)


async def test_index_status_counts_the_indexed_project(project: Path, client: Client):
    async with client:
        await client.call_tool("index_codebase", {"project_path": str(project)})
        result = await client.call_tool("index_status", {"project_path": str(project)})

    status = IndexStatusResponse.model_validate(structured(result))
    assert (status.is_indexed, status.files_count, status.chunks_count) == (True, 2, 2)


async def test_search_sees_a_function_added_after_indexing(project: Path, client: Client):
    async with client:
        await client.call_tool("index_codebase", {"project_path": str(project)})
        with (project / "config.py").open("a") as config_module:
            config_module.write(RETRY_FUNCTION)
        result = await client.call_tool(
            "search_code",
            {"query": "try again after waiting longer", "project_path": str(project)},
        )

    first_hit = SearchCodeResponse.model_validate(structured(result)).results[0]
    assert first_hit.name == "retry_with_backoff"


async def test_search_in_a_missing_project_is_an_error_naming_the_path(
    tmp_path: Path, client: Client
):
    missing = tmp_path / "missing"
    async with client:
        result = await client.call_tool(
            "search_code", {"query": "anything", "project_path": str(missing)}
        )

    assert result.is_error
    assert str(missing) in str(result.content)


async def test_a_limit_over_the_maximum_is_refused(project: Path, client: Client):
    async with client:
        result = await client.call_tool(
            "search_code", {"query": "anything", "project_path": str(project), "limit": 500}
        )

    assert result.is_error
