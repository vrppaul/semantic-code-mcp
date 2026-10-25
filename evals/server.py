"""The server as an agent meets it: a process started over stdio, spoken to through its tools.

Measuring through the tools, not the package's modules, keeps the harness valid across a
rebuild of everything behind them. This module alone knows the tools' names and the fields
of their answers the harness reads.
"""

import os
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TextIO

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult
from pydantic import BaseModel, ConfigDict

SETTINGS_PREFIX = "SEMANTIC_CODE_MCP_"
CACHE_DIR_VARIABLE = f"{SETTINGS_PREFIX}CACHE_DIR"
SERVER_MODULE = "semantic_code_mcp"
# Long enough for a full index of the largest corpus with a cold model download
READ_TIMEOUT_SECONDS = 900.0


class SearchHit(BaseModel):
    """One result of `search_code`, as far as the harness reads it."""

    model_config = ConfigDict(frozen=True)

    file_path: str
    name: str
    chunk_type: str
    line_start: int
    line_end: int


class SearchAnswer(BaseModel):
    """The answer of `search_code`."""

    model_config = ConfigDict(frozen=True)

    results: tuple[SearchHit, ...]


class IndexReport(BaseModel):
    """The answer of `index_codebase`."""

    model_config = ConfigDict(frozen=True)

    files_indexed: int
    chunks_indexed: int
    duration_seconds: float


class StatusReport(BaseModel):
    """The answer of `index_status`, as far as the harness reads it."""

    model_config = ConfigDict(frozen=True)

    files_count: int


class ServerConnection:
    """A client session to one server process, with the tools the harness calls."""

    def __init__(self, session: ClientSession) -> None:
        self._session = session

    async def index(self, project: Path) -> IndexReport:
        """Index a project from scratch or incrementally, as `index_codebase` decides."""
        result = await self._session.call_tool("index_codebase", {"project_path": str(project)})
        return IndexReport.model_validate(tool_payload(result))

    async def status(self, project: Path) -> StatusReport:
        """Ask how far a project's index is, as every search does before it searches."""
        result = await self._session.call_tool("index_status", {"project_path": str(project)})
        return StatusReport.model_validate(tool_payload(result))

    async def search(self, project: Path, query: str, limit: int) -> SearchAnswer:
        """Search a project as an agent would."""
        arguments = {"query": query, "project_path": str(project), "limit": limit}
        result = await self._session.call_tool("search_code", arguments)
        return SearchAnswer.model_validate(tool_payload(result))


@asynccontextmanager
async def connect(index_directory: Path, server_log: TextIO) -> AsyncGenerator[ServerConnection]:
    """Start a server whose indexes live in index_directory, its log written to server_log.

    Settings of the caller's environment (`SEMANTIC_CODE_MCP_*`) are left out, so a
    measurement depends on the defaults and the index directory alone.
    """
    environment = {
        name: value for name, value in os.environ.items() if not name.startswith(SETTINGS_PREFIX)
    }
    environment[CACHE_DIR_VARIABLE] = str(index_directory)
    parameters = StdioServerParameters(
        command=sys.executable, args=["-m", SERVER_MODULE], env=environment
    )
    async with (
        stdio_client(parameters, errlog=server_log) as (read_stream, write_stream),
        ClientSession(
            read_stream, write_stream, read_timeout_seconds=READ_TIMEOUT_SECONDS
        ) as session,
    ):
        await session.initialize()
        yield ServerConnection(session)


def tool_payload(result: CallToolResult) -> object:
    """The structured answer of a tool call, or an error naming what the tool said."""
    if result.is_error or result.structured_content is None:
        message = f"tool call failed: {result.content}"
        raise ToolCallFailedError(message)
    return result.structured_content


class ToolCallFailedError(RuntimeError):
    """A tool answered with an error or without structured content."""
