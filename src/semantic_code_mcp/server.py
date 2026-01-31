"""FastMCP tool definitions."""

import time
from pathlib import Path

import structlog
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession

from semantic_code_mcp.container import get_container
from semantic_code_mcp.models import (
    ErrorResponse,
    FormattedSearchResult,
    IndexCodebaseResponse,
    IndexResultSummary,
    IndexStatusResponse,
    IndexStatusSummary,
    SearchDebugInfo,
    SearchResponse,
    SearchStats,
    SearchTimings,
)
from semantic_code_mcp.profiling import profile_async

log = structlog.get_logger()

mcp = FastMCP("semantic-code-mcp")


@mcp.tool()
@profile_async("search_code")
async def search_code(
    query: str,
    project_path: str,
    ctx: Context[ServerSession, None],
    limit: int = 10,
) -> SearchResponse | ErrorResponse:
    """Search for code semantically similar to the query.

    Finds code by meaning, not just text matching. Use this when you want to find
    code related to a concept without knowing exact variable/function names.

    Examples:
    - "authentication logic" - finds login, session handling, token validation
    - "error handling for API calls" - finds try/except blocks, error responses
    - "database connection setup" - finds connection pooling, ORM initialization

    Automatically indexes the project if not already indexed, and re-indexes
    any files that have changed since the last search.

    Args:
        query: Natural language description of what you're looking for.
        project_path: Absolute path to the project root directory.
        limit: Maximum number of results to return (default 10).

    Returns:
        List of matching code chunks with file path, line numbers, content, and score.
    """
    total_start = time.perf_counter()

    await ctx.info(f"Searching for: {query}")

    path = Path(project_path)
    if not path.exists():
        await ctx.warning(f"Project path does not exist: {project_path}")
        return ErrorResponse(error=f"Path does not exist: {project_path}")

    # Delegate to search service
    container = get_container()
    search_service = container.create_search_service(path)
    outcome = await search_service.search(query, path, limit, on_progress=ctx.report_progress)

    total_ms = round((time.perf_counter() - total_start) * 1000, 1)
    await ctx.info(f"Found {len(outcome.results)} results in {total_ms}ms")

    # Transform domain -> response
    indexing_ms = round(outcome.index_result.duration_seconds * 1000, 1)
    timings = SearchTimings(
        embedding_ms=outcome.embedding_ms,
        search_ms=outcome.search_ms,
        total_ms=total_ms,
        indexing_ms=indexing_ms if indexing_ms > 0 else None,
    )

    index_result_summary = IndexResultSummary(
        files_indexed=outcome.index_result.files_indexed,
        chunks_indexed=outcome.index_result.chunks_indexed,
    )

    was_stale = outcome.index_result.files_indexed > 0

    # Get live index status for debug info
    indexer = container.create_indexer(path)
    status = indexer.get_status(path)

    debug = SearchDebugInfo(
        timings=timings,
        stats=SearchStats.from_outcome(outcome),
        index_status=IndexStatusSummary(
            files_count=status.files_count,
            chunks_count=status.chunks_count,
            was_stale=was_stale,
        ),
        index_result=index_result_summary,
    )

    return SearchResponse(
        results=[FormattedSearchResult.from_domain(r) for r in outcome.results],
        debug=debug,
    )


@mcp.tool()
@profile_async("index_codebase")
async def index_codebase(
    project_path: str,
    ctx: Context[ServerSession, None],
    force: bool = False,
) -> IndexCodebaseResponse | ErrorResponse:
    """Index a codebase for semantic search.

    Scans Python files, extracts functions/classes/methods, generates embeddings,
    and stores them for fast semantic search.

    Use force=True to re-index everything even if files haven't changed.
    Otherwise, only new and modified files are indexed (incremental).

    Args:
        project_path: Absolute path to the project root directory.
        force: If True, re-index all files regardless of changes.

    Returns:
        Statistics about the indexing operation.
    """
    await ctx.info(f"Indexing: {project_path}")

    path = Path(project_path)
    if not path.exists():
        await ctx.warning(f"Project path does not exist: {project_path}")
        return ErrorResponse(error=f"Path does not exist: {project_path}")

    container = get_container()
    index_service = container.create_index_service(path)
    result = await index_service.index(path, force=force, on_progress=ctx.report_progress)

    await ctx.info(
        f"Indexed {result.files_indexed} files, {result.chunks_indexed} chunks "
        f"in {result.duration_seconds:.2f}s"
    )

    return IndexCodebaseResponse(
        files_indexed=result.files_indexed,
        chunks_indexed=result.chunks_indexed,
        files_deleted=result.files_deleted,
        duration_seconds=result.duration_seconds,
    )


@mcp.tool()
@profile_async("index_status")
async def index_status(
    project_path: str,
    ctx: Context[ServerSession, None],
) -> IndexStatusResponse | ErrorResponse:
    """Get the index status for a project.

    Returns information about whether the project is indexed, when it was last
    updated, and how many files and chunks are indexed.

    Note: search_code automatically re-indexes stale files before searching,
    so there is no need to check or act on staleness manually.

    Args:
        project_path: Absolute path to the project root directory.

    Returns:
        Index status including files count and chunks count.
    """
    path = Path(project_path)
    if not path.exists():
        await ctx.warning(f"Project path does not exist: {project_path}")
        return ErrorResponse(error=f"Path does not exist: {project_path}")

    container = get_container()
    indexer = container.create_indexer(path)
    status = indexer.get_status(path)

    return IndexStatusResponse(
        is_indexed=status.is_indexed,
        last_updated=status.last_updated.isoformat() if status.last_updated else None,
        files_count=status.files_count,
        chunks_count=status.chunks_count,
    )
