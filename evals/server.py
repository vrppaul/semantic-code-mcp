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

from semantic_code_mcp.domain.models import SearchMode, StoreKind

SETTINGS_PREFIX = "SEMANTIC_CODE_MCP_"
CACHE_DIR_VARIABLE = f"{SETTINGS_PREFIX}CACHE_DIR"
INDEX_STORE_VARIABLE = f"{SETTINGS_PREFIX}INDEX_STORE"
SEARCH_MODE_VARIABLE = f"{SETTINGS_PREFIX}SEARCH_MODE"
SERVER_MODULE = "semantic_code_mcp"
# Long enough for a full index of the largest corpus with a cold model download: `agents`
# indexes corpora kept outside the repository, and one of Sentry's 87,694 chunks took about
# an hour, where the fifteen minutes this was cut its first index short
READ_TIMEOUT_SECONDS = 3 * 3600.0


class SearchHit(BaseModel):
    """One result of `search_code`, as far as the harness reads it."""

    model_config = ConfigDict(frozen=True)

    file_path: str
    name: str
    kind: str
    parent: str | None
    line_start: int
    line_end: int


class SearchNarrowing(BaseModel):
    """What a narrowed search keeps to, as `search_code`'s parameters name it."""

    model_config = ConfigDict(frozen=True)

    chunk_kinds: tuple[str, ...]
    exclude_glob: str


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

    async def search(
        self,
        project: Path,
        query: str,
        limit: int,
        narrowing: SearchNarrowing | None = None,
    ) -> SearchAnswer:
        """Search a project as an agent would, narrowed where a narrowing is given, ranked as
        the server's setting says."""
        arguments: dict[str, object] = {
            "query": query,
            "project_path": str(project),
            "limit": limit,
        }
        if narrowing is not None:
            arguments |= {
                "chunk_kinds": list(narrowing.chunk_kinds),
                "exclude_glob": narrowing.exclude_glob,
            }
        result = await self._session.call_tool("search_code", arguments)
        return SearchAnswer.model_validate(tool_payload(result))


class ServerOptions(BaseModel):
    """What a run sets of the servers it starts: the store that keeps their indexes and how
    their searches rank, each the server's default where a run names none."""

    model_config = ConfigDict(frozen=True)

    index_store: StoreKind | None
    search_mode: SearchMode | None


# The servers of a run that sets nothing: the default store and the default search
DEFAULT_OPTIONS = ServerOptions(index_store=None, search_mode=None)


class ServerStarter:
    """Starts the server processes of a run, their logs written to one file, as the run's
    options set them.

    Settings of the caller's environment (`SEMANTIC_CODE_MCP_*`) are left out, so a
    measurement depends on the defaults, the options and the index directory alone.
    """

    def __init__(self, server_log: TextIO, options: ServerOptions) -> None:
        self._server_log = server_log
        self._options = options

    def searching(self, search_mode: SearchMode) -> ServerStarter:
        """A starter of the same run whose servers search in another mode, their logs written
        to the same file."""
        return ServerStarter(
            self._server_log,
            ServerOptions(index_store=self._options.index_store, search_mode=search_mode),
        )

    def environment(self, index_directory: Path) -> dict[str, str]:
        """The environment each server of this starter runs in, its indexes in
        index_directory."""
        return server_environment(index_directory, self._options)

    @asynccontextmanager
    async def connect(self, index_directory: Path) -> AsyncGenerator[ServerConnection]:
        """Start a server whose indexes live in index_directory."""
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", SERVER_MODULE],
            env=self.environment(index_directory),
        )
        async with (
            stdio_client(parameters, errlog=self._server_log) as (read_stream, write_stream),
            ClientSession(
                read_stream, write_stream, read_timeout_seconds=READ_TIMEOUT_SECONDS
            ) as session,
        ):
            await session.initialize()
            yield ServerConnection(session)


def server_environment(index_directory: Path, options: ServerOptions) -> dict[str, str]:
    """The environment a server of the harness runs in: the caller's without its settings,
    the index directory, and the store and the search mode a run names, where it names
    them."""
    environment = environment_without_settings()
    environment[CACHE_DIR_VARIABLE] = str(index_directory)
    if options.index_store is not None:
        environment[INDEX_STORE_VARIABLE] = options.index_store
    if options.search_mode is not None:
        environment[SEARCH_MODE_VARIABLE] = options.search_mode
    return environment


def environment_without_settings() -> dict[str, str]:
    """The caller's environment without the server's settings (`SEMANTIC_CODE_MCP_*`), so a
    server started in it takes its defaults."""
    return {
        name: value for name, value in os.environ.items() if not name.startswith(SETTINGS_PREFIX)
    }


def tool_payload(result: CallToolResult) -> object:
    """The structured answer of a tool call, or an error naming what the tool said."""
    if result.is_error or result.structured_content is None:
        message = f"tool call failed: {result.content}"
        raise ToolCallFailedError(message)
    return result.structured_content


class ToolCallFailedError(RuntimeError):
    """A tool answered with an error or without structured content."""
