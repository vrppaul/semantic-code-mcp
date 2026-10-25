"""End-to-end tests: the MCP tools driven through a client, as an agent calls them."""

import asyncio
import os
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

# An mtime well after any the test wrote, so the change is seen on any file system
LATER_MTIME_SECONDS = 4_000_000_000

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
    # "legacy" goes through JSON-RPC framing, as a real client does
    return Client(create_app(test_settings, embedder=embedder), mode="legacy")


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
        config_module = project / "config.py"
        config_module.write_text(config_module.read_text() + RETRY_FUNCTION)
        os.utime(config_module, (LATER_MTIME_SECONDS, LATER_MTIME_SECONDS))
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


async def test_a_failure_inside_a_tool_reaches_the_agent_with_its_cause(
    project: Path, tmp_path: Path, embedder: SentenceTransformerEmbedder
):
    """An index directory that cannot be made: the agent must hear why, not just that."""
    blocking_file = tmp_path / "not-a-directory"
    blocking_file.write_text("")
    client = Client(create_app(Settings(cache_dir=blocking_file), embedder=embedder), mode="legacy")

    async with client:
        result = await client.call_tool("index_codebase", {"project_path": str(project)})

    assert result.is_error
    assert "not-a-directory" in str(result.content)


async def test_a_relative_project_path_is_refused(client: Client):
    async with client:
        result = await client.call_tool("index_status", {"project_path": "relative/project"})

    assert result.is_error
    assert "absolute" in str(result.content)


async def test_two_spellings_of_one_project_share_its_local_index(
    project: Path, tmp_path: Path, embedder: SentenceTransformerEmbedder
):
    settings = Settings(cache_dir=tmp_path / "cache", local_index=True)
    client = Client(create_app(settings, embedder=embedder), mode="legacy")
    roundabout = project / ".." / project.name

    async with client:
        await asyncio.gather(
            client.call_tool("index_codebase", {"project_path": str(roundabout)}),
            client.call_tool("index_codebase", {"project_path": str(project)}),
        )
        result = await client.call_tool("index_status", {"project_path": str(project)})

    # One lock for both spellings: the two runs do not store the chunks twice
    assert IndexStatusResponse.model_validate(structured(result)).chunks_count == 2
