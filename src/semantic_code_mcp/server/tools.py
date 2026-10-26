"""The MCP tools: each checks its project path, hands the work to the project's services
and returns what the presenters make of the report.

A tool fails with a `ToolError`, whose message reaches the agent; anything else that
escapes is a crash, and the agent hears only that the tool failed.
"""

import asyncio
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import structlog
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from semantic_code_mcp.domain.models import ChunkKind, Language, SearchQuery, SearchScope
from semantic_code_mcp.server.presenters import present_index, present_search, present_status
from semantic_code_mcp.server.responses import (
    IndexCodebaseResponse,
    IndexStatusResponse,
    SearchCodeResponse,
)
from semantic_code_mcp.services.progress import ProgressReporter
from semantic_code_mcp.services.projects import (
    ProjectLocation,
    ProjectRegistry,
    ProjectServices,
)

log = structlog.get_logger()

DEFAULT_RESULTS = 10
MAX_RESULTS = 50
# What an agent reads of the search's narrowing parameters in the tool's input schema
KINDS_DESCRIPTION = (
    "Only chunks of these kinds; Rust's structs, enums, traits and impls are classes."
)
LANGUAGES_DESCRIPTION = "Only chunks of files in these languages."
PATH_GLOB_DESCRIPTION = (
    "Only files this glob names, as in a .gitignore: without a slash a name at any depth"
    " (`*.rs`), with one a path from the directory project_path names (`src/**/*.py`); `*` stays"
    " within a directory, `**` crosses any number, and a directory takes what it holds."
)
# What a glob may hold and still name no path: blanks and slashes alone
GLOB_BLANKS = " /"
EXCLUDE_GLOB_DESCRIPTION = "No file this glob names, read as path_glob is (`tests`, `*_test.py`)."


# A tool's parameters are its input schema, which an agent fills in flat (ADR 006, D10)
async def search_code(  # noqa: PLR0913
    *,
    query: str,
    project_path: str,
    ctx: Context[ProjectRegistry],
    limit: Annotated[int, Field(ge=1, le=MAX_RESULTS)] = DEFAULT_RESULTS,
    chunk_kinds: Annotated[list[ChunkKind] | None, Field(description=KINDS_DESCRIPTION)] = None,
    languages: Annotated[list[Language] | None, Field(description=LANGUAGES_DESCRIPTION)] = None,
    path_glob: Annotated[str | None, Field(description=PATH_GLOB_DESCRIPTION)] = None,
    exclude_glob: Annotated[str | None, Field(description=EXCLUDE_GLOB_DESCRIPTION)] = None,
    include_debug: bool = False,
) -> SearchCodeResponse:
    """Search for code semantically similar to the query.

    Finds code by meaning, not just text matching. Use this when you want to find
    code related to a concept without knowing exact variable/function names; for an
    exact identifier or string, Grep is faster and complete.

    Examples:
    - "authentication logic" - finds login, session handling, token validation
    - "error handling for API calls" - finds try/except blocks, error responses
    - "database connection setup" - finds connection pooling, ORM initialization

    Indexes Python, Rust and Markdown files. Automatically indexes the project if not
    already indexed, and re-indexes any files that have changed since the last search.

    Args:
        query: Natural language description of what you're looking for.
        project_path: Absolute path to the project, or to a directory inside its git
            repository, which keeps the search to that directory.
        limit: Maximum number of results to return (default 10, at most 50).
        chunk_kinds: Only chunks of these kinds.
        languages: Only chunks of files in these languages.
        path_glob: Only files this glob names, from the project root.
        exclude_glob: No file this glob names.
        include_debug: Also return timings and counts of the search.

    Returns:
        Matching chunks, best first by meaning and words together, with the file's absolute
        path, line range, name, kind (module, function, method, class, section), the class,
        impl or heading it belongs to, content, and its similarity in meaning to the query
        alone, which a hit its words found can show low.
    """
    with failures_as_tool_errors():
        location = await located_project(ctx, project_path)
        services = project_services(ctx, location.root)
        scope = SearchScope(
            kinds=tuple(chunk_kinds or ()),
            languages=tuple(languages or ()),
            path_glob=given_glob(path_glob),
            exclude_glob=given_glob(exclude_glob),
            directory=location.directory,
        )
        report = await services.search.search(
            SearchQuery(text=query, limit=limit, scope=scope),
            location.root,
            progress=ProgressReporter(ctx.report_progress),
        )
        return present_search(report, location.root, include_debug=include_debug)


async def index_codebase(
    *,
    project_path: str,
    ctx: Context[ProjectRegistry],
    force: bool = False,
) -> IndexCodebaseResponse:
    """Index a codebase for semantic search.

    Splits Python, Rust and Markdown files into functions, classes, methods, module
    docstrings and document sections, embeds them, and stores them for search.
    `search_code` indexes on its own; call this only to re-index everything.

    Use force=True to re-index everything even if files haven't changed.
    Otherwise, only new and modified files are indexed (incremental).

    Args:
        project_path: Absolute path to the project; a directory inside a git repository
            indexes the whole repository.
        force: If True, re-index all files regardless of changes.

    Returns:
        Statistics about the indexing operation.
    """
    with failures_as_tool_errors():
        location = await located_project(ctx, project_path)
        services = project_services(ctx, location.root)
        report = await services.indexing.index(
            location.root, force=force, progress=ProgressReporter(ctx.report_progress)
        )
        return present_index(report)


async def index_status(
    *,
    project_path: str,
    ctx: Context[ProjectRegistry],
) -> IndexStatusResponse:
    """Get the index status for a project.

    Returns whether the project is indexed, when it was last updated, how many files
    and chunks are indexed, how many files changed since, and what built the index (each
    model and chunker version keeps an index of its own).

    Note: search_code automatically re-indexes stale files before searching,
    so there is no need to check or act on staleness manually.

    Args:
        project_path: Absolute path to the project; a directory inside a git repository
            reports the whole repository's index.

    Returns:
        Index status including files count and chunks count.
    """
    with failures_as_tool_errors():
        location = await located_project(ctx, project_path)
        services = project_services(ctx, location.root)
        return present_status(await services.indexing.status(location.root))


async def located_project(ctx: Context[ProjectRegistry], project_path: str) -> ProjectLocation:
    """Where the directory a tool was asked about lies: its project's root and the
    directory under it, git asked in a worker thread."""
    directory = existing_directory(project_path)
    registry = ctx.request_context.lifespan_context
    return await asyncio.to_thread(registry.locate, directory)


def project_services(ctx: Context[ProjectRegistry], project_root: Path) -> ProjectServices:
    """The services of a project, from the registry the session shares."""
    return ctx.request_context.lifespan_context.services_for(project_root)


@contextmanager
def failures_as_tool_errors() -> Generator[None]:
    """Turn what the parts raise when the disk, git, LanceDB or the model fail into tool
    errors that carry their cause, which mcp would otherwise drop."""
    try:
        yield
    except (OSError, RuntimeError, ValueError) as error:
        log.exception("tool_failed")
        message = f"{type(error).__name__}: {error}"
        raise ToolError(message) from error


def given_glob(glob: str | None) -> str | None:
    """A glob as a search reads it: None for one that names no path, as an empty string
    an agent fills an optional argument with, or a slash alone, which would otherwise
    narrow the search to nothing."""
    if glob is None or not glob.strip(GLOB_BLANKS):
        return None
    return glob


def existing_directory(project_path: str) -> Path:
    """The path a tool was given, resolved, as a directory that exists; or a tool error
    saying why not."""
    path = Path(project_path)
    if not path.is_absolute():
        message = f"Project path must be absolute: {project_path}"
        raise ToolError(message)
    if not path.exists():
        message = f"Project path does not exist: {project_path}"
        raise ToolError(message)
    if not path.is_dir():
        message = f"Project path is not a directory: {project_path}"
        raise ToolError(message)
    return path.resolve()
