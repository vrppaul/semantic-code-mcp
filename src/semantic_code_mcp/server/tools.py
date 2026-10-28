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
from mcp.server.mcpserver import Context, ListRoots, Resolve
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ListRootsResult
from pydantic import Field

from semantic_code_mcp.domain.globs import UnreadableGlobError, read_glob
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
    " below the directory project_path names (`*.rs`, and `tests/`, as a trailing slash"
    " does not count), with one before a name a path from that directory (`src/**/*.py`,"
    " `/tests`); `*` stays within a directory, `**` crosses any number, even inside a name"
    " (`src**.py`, where a .gitignore reads it as `*`), `?` is one"
    " character, and a directory takes what it holds. No character classes (`[ab]`),"
    " braces, escapes, a leading `!`, `..` or more than 1000 characters: each is refused"
    " with the reason."
)
EXCLUDE_GLOB_DESCRIPTION = "No file this glob names, read as path_glob is (`tests`, `*_test.py`)."
PROJECT_PATH_DESCRIPTION = (
    "Absolute path to the project, or to a directory inside its git repository. Left out,"
    " the client's workspace root: the directory Claude Code was started in, which must lie in"
    " a git repository."
)
# A tool's own parameter that names the project, which the client's roots stand in for
PROJECT_PATH_PARAMETER = "project_path"
# How a refusal lists the roots a client named
ROOTS_SEPARATOR = ", "


# A tool's parameters are its input schema, which an agent fills in flat (ADR 006, D10)
async def search_code(  # noqa: PLR0913
    *,
    query: str,
    ctx: Context[ProjectRegistry],
    roots: Annotated[ListRootsResult, Resolve(client_roots)],
    project_path: Annotated[str | None, Field(description=PROJECT_PATH_DESCRIPTION)] = None,
    limit: Annotated[int, Field(ge=1, le=MAX_RESULTS)] = DEFAULT_RESULTS,
    chunk_kinds: Annotated[list[ChunkKind] | None, Field(description=KINDS_DESCRIPTION)] = None,
    languages: Annotated[list[Language] | None, Field(description=LANGUAGES_DESCRIPTION)] = None,
    path_glob: Annotated[str | None, Field(description=PATH_GLOB_DESCRIPTION)] = None,
    exclude_glob: Annotated[str | None, Field(description=EXCLUDE_GLOB_DESCRIPTION)] = None,
    include_debug: bool = False,
) -> SearchCodeResponse:
    """Find the functions, classes, methods and document sections of a project by the words
    their code uses.

    It matches the words of code: write the identifiers and terms the code would use, and
    on a miss search again with others. A query that is a name, or a qualified one, brings
    the chunks of that name first:
    - "retry backoff delay attempts"
    - "parse_args config options"
    - "Stack.push"

    For every place a name is used, Grep: this returns only the best matches.

    Indexes Python, Rust and Markdown files. Automatically indexes the project if not
    already indexed, and re-indexes any files that have changed since the last search.

    Args:
        query: The words to find, as code would write them, or a name.
        project_path: Absolute path to the project, or to a directory inside its git
            repository, which keeps the search to that directory; left out, the client's
            workspace root, which must lie in a git repository.
        limit: Maximum number of results to return.
        chunk_kinds: Only chunks of these kinds.
        languages: Only chunks of files in these languages.
        path_glob: Only files this glob names, read from the directory searched.
        exclude_glob: No file this glob names, read as path_glob is.
        include_debug: Also return timings and counts of the search.

    Returns:
        The root of the project whose index answered and the directory the search kept to;
        then matching chunks, best first, with the file's absolute path, line range, name,
        kind (module, function, method, class, section), the class, impl or heading it
        belongs to, and content.
    """
    with failures_as_tool_errors():
        # Read before git is asked anything, so a glob refused costs nothing
        included = given_glob(path_glob)
        excluded = given_glob(exclude_glob)
        location = await located_project(ctx, project_path, roots)
        services = project_services(ctx, location.root)
        registry = ctx.request_context.lifespan_context
        scope = SearchScope(
            kinds=tuple(chunk_kinds or ()),
            languages=tuple(languages or ()),
            path_glob=included,
            exclude_glob=excluded,
            directory=location.directory,
        )
        report = await services.search.search(
            SearchQuery(
                text=query,
                limit=limit,
                scope=scope,
                mode=registry.search_mode,
            ),
            location.root,
            progress=ProgressReporter(ctx.report_progress),
        )
        return present_search(report, location, include_debug=include_debug)


async def index_codebase(
    *,
    ctx: Context[ProjectRegistry],
    roots: Annotated[ListRootsResult, Resolve(client_roots)],
    project_path: Annotated[str | None, Field(description=PROJECT_PATH_DESCRIPTION)] = None,
    force: bool = False,
) -> IndexCodebaseResponse:
    """Index a codebase's words for search.

    Splits Python, Rust and Markdown files into functions, classes, methods, module
    docstrings and document sections, and stores them with their words; a server set to
    search by meaning too embeds them. `search_code` indexes on its own; call this only to
    re-index everything.

    With `force: true`, it re-indexes everything even if files haven't changed; a model
    whose weights nothing pins, as a local path, has its vectors dropped too, so a server
    set to search by meaning embeds every chunk again. Otherwise, only new and modified
    files are indexed (incremental).

    Args:
        project_path: Absolute path to the project; a directory inside a git repository
            indexes the whole repository, unless the repository ignores it or holds the
            user's home directory. Left out, the client's workspace root, which must lie in
            a git repository.
        force: Re-index all files regardless of changes.

    Returns:
        The root of the project indexed, and what the run indexed and deleted.
    """
    with failures_as_tool_errors():
        location = await located_project(ctx, project_path, roots)
        services = project_services(ctx, location.root)
        report = await services.indexing.index(
            location.root, force=force, progress=ProgressReporter(ctx.report_progress)
        )
        return present_index(report, location.root)


async def index_status(
    *,
    ctx: Context[ProjectRegistry],
    roots: Annotated[ListRootsResult, Resolve(client_roots)],
    project_path: Annotated[str | None, Field(description=PROJECT_PATH_DESCRIPTION)] = None,
) -> IndexStatusResponse:
    """Get the index status for a project.

    Returns whether the project is indexed, when it was last updated, how many of its
    Python, Rust and Markdown files and their chunks are indexed, how many chunks have the
    model's vectors, how many files changed since, and what built the index (each chunker
    version keeps an index of its own, and each model's vectors are kept apart in it).

    Note: search_code automatically re-indexes stale files before searching,
    so there is no need to check or act on staleness manually.

    Args:
        project_path: Absolute path to the project; a directory inside a git repository
            reports the whole repository's index, unless the repository ignores it or holds
            the user's home directory. Left out, the client's workspace root, which must
            lie in a git repository.

    Returns:
        The root of the project whose index this is, and its status, including the
        counts of files and chunks.
    """
    with failures_as_tool_errors():
        location = await located_project(ctx, project_path, roots)
        services = project_services(ctx, location.root)
        status = await services.indexing.status(location.root)
        return present_status(status, location.root)


async def client_roots(
    ctx: Context[ProjectRegistry], project_path: str | None
) -> ListRoots | ListRootsResult:
    """The client's workspace roots, asked of it only where a tool names no project_path;
    a client that declares no roots is told to name one."""
    if given_path(project_path) is not None:
        return ListRootsResult(roots=[])
    capabilities = ctx.client_capabilities
    if capabilities is None or capabilities.roots is None:
        message = f"Give {PROJECT_PATH_PARAMETER}: this client names no workspace root"
        raise ToolError(message)
    return ListRoots()


async def located_project(
    ctx: Context[ProjectRegistry], project_path: str | None, roots: ListRootsResult
) -> ProjectLocation:
    """Where the directory a tool was asked about lies, git asked in a worker thread: the
    path it was given, else the client's one workspace root, taken only inside a repository
    git names a project, as a session started in the home directory or a folder of projects
    would index them all as one; or a tool error saying why neither will do."""
    registry = ctx.request_context.lifespan_context
    given = given_path(project_path)
    if given is not None:
        return await asyncio.to_thread(registry.locate, existing_directory(given))
    root = workspace_root(roots)
    location = await asyncio.to_thread(registry.locate, root)
    if not location.is_repository:
        message = (
            f"Give {PROJECT_PATH_PARAMETER}: no git repository names the client's workspace "
            f"root, {root}, a project, as it lies in none, or its repository ignores it, or that "
            "repository's root holds the home directory; a search would index every file below "
            "it as one project"
        )
        raise ToolError(message)
    return location


def given_path(project_path: str | None) -> str | None:
    """The project_path a tool was given; None where it was left out or left blank, as an
    agent fills an optional string it has nothing for."""
    if project_path is None or not project_path.strip():
        return None
    return project_path


def workspace_root(roots: ListRootsResult) -> Path:
    """The client's one workspace root, as a directory here; or a tool error saying why the
    client names none."""
    match roots.roots:
        case [root]:
            return workspace_directory(str(root.uri))
        case _:
            uris = ROOTS_SEPARATOR.join(str(root.uri) for root in roots.roots)
            message = (
                f"Give {PROJECT_PATH_PARAMETER}: the client names {len(roots.roots)} workspace "
                f"roots, not one ({uris})"
            )
            raise ToolError(message)


def workspace_directory(uri: str) -> Path:
    """The directory of a workspace root's URI, resolved as a given path is; a tool error
    asking for project_path where it is none: a root on another machine, or no directory."""
    try:
        root = Path.from_uri(uri)
    except ValueError as error:
        message = f"Give {PROJECT_PATH_PARAMETER}: the client's workspace root, {uri}, is not here"
        raise ToolError(message) from error
    if not root.is_dir():
        message = (
            f"Give {PROJECT_PATH_PARAMETER}: the client's workspace root, {root}, is no directory"
        )
        raise ToolError(message)
    # A root reached through a link is the directory it links to, as a given path is
    return root.resolve()


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
    """A glob as a search reads it, or a tool error that tells the agent why a search
    cannot read it."""
    try:
        return read_glob(glob)
    except UnreadableGlobError as error:
        raise ToolError(str(error)) from error


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
