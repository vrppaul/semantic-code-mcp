"""FastMCP server and tool definitions."""

import asyncio
import time
from pathlib import Path

import structlog
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession

from semantic_code_mcp.config import Settings, get_index_path
from semantic_code_mcp.indexer.embedder import Embedder
from semantic_code_mcp.indexer.indexer import Indexer
from semantic_code_mcp.models import IndexProgress, IndexResult
from semantic_code_mcp.storage.lancedb import VectorStore

log = structlog.get_logger()

# Create settings and components
settings = Settings()
indexer = Indexer(settings)
embedder = Embedder(settings)

# Create MCP server
mcp = FastMCP("semantic-code-mcp")


async def _do_index_with_progress(
    ctx: Context[ServerSession, None],
    path: Path,
    force: bool,
) -> IndexResult:
    """Run indexing with progress updates."""
    result: IndexResult | None = None

    async for update in indexer.index_async(path, force=force):
        if isinstance(update, IndexProgress):
            await ctx.report_progress(
                progress=update.percent,
                total=100,
                message=update.message,
            )
        else:
            # It's the final IndexResult
            result = update

    if result is None:
        raise RuntimeError("Indexer did not return a result")

    return result


@mcp.tool()
async def search_code(
    query: str,
    project_path: str,
    ctx: Context[ServerSession, None],
    limit: int = 10,
) -> dict:
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
    timings: dict[str, float] = {}

    await ctx.info(f"Searching for: {query}")

    path = Path(project_path)
    if not path.exists():
        await ctx.warning(f"Project path does not exist: {project_path}")
        return {"error": f"Path does not exist: {project_path}"}

    index_path = get_index_path(settings, path)

    # Check if we need to index
    t0 = time.perf_counter()
    status = indexer.get_status(path)
    timings["status_check_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    index_result = None
    if not status.is_indexed:
        await ctx.info("Index not found, performing full indexing...")
        t0 = time.perf_counter()
        index_result = await _do_index_with_progress(ctx, path, force=False)
        timings["indexing_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    elif status.stale_files:
        await ctx.info(f"Re-indexing {len(status.stale_files)} stale files")
        t0 = time.perf_counter()
        index_result = await _do_index_with_progress(ctx, path, force=False)
        timings["indexing_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    await ctx.report_progress(progress=90, total=100, message="Searching...")

    # Embed the query
    t0 = time.perf_counter()
    query_embedding = await asyncio.to_thread(embedder.embed_text, query)
    timings["embedding_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # Search the vector store
    t0 = time.perf_counter()
    store = VectorStore(index_path)
    results = await asyncio.to_thread(store.search, query_embedding, limit)
    timings["search_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    timings["total_ms"] = round((time.perf_counter() - total_start) * 1000, 1)

    await ctx.report_progress(progress=100, total=100, message="Complete")
    await ctx.info(f"Found {len(results)} results in {timings['total_ms']}ms")

    # Estimate tokens in results (1 token ≈ 4 chars)
    total_content = "\n".join(r.content for r in results)
    tokens_estimate = len(total_content) // 4

    # Build debug info
    debug = {
        "timings": timings,
        "stats": {
            "results_count": len(results),
            "unique_files": len({r.file_path for r in results}),
            "tokens_estimate": tokens_estimate,
            "model_was_loaded": embedder.is_loaded,
        },
        "index_status": {
            "files_count": status.files_count,
            "chunks_count": status.chunks_count,
            "was_stale": len(status.stale_files) > 0 if status.is_indexed else None,
        },
    }
    if index_result:
        debug["index_result"] = {
            "files_indexed": index_result.files_indexed,
            "chunks_indexed": index_result.chunks_indexed,
        }

    # Convert results to dicts for MCP response
    return {
        "result": [
            {
                "file_path": r.file_path,
                "line_start": r.line_start,
                "line_end": r.line_end,
                "name": r.name,
                "chunk_type": r.chunk_type.value,
                "content": r.content,
                "score": round(r.score, 3),
            }
            for r in results
        ],
        "debug": debug,
    }


@mcp.tool()
async def index_codebase(
    project_path: str,
    ctx: Context[ServerSession, None],
    force: bool = False,
) -> dict:
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
        return {"error": f"Path does not exist: {project_path}"}

    # Perform indexing with progress updates
    result = await _do_index_with_progress(ctx, path, force=force)

    await ctx.info(
        f"Indexed {result.files_indexed} files, {result.chunks_indexed} chunks "
        f"in {result.duration_seconds:.2f}s"
    )

    return {
        "files_indexed": result.files_indexed,
        "chunks_indexed": result.chunks_indexed,
        "files_deleted": result.files_deleted,
        "duration_seconds": round(result.duration_seconds, 2),
    }


@mcp.tool()
async def index_status(
    project_path: str,
    ctx: Context[ServerSession, None],
) -> dict:
    """Get the index status for a project.

    Returns information about whether the project is indexed, when it was last
    updated, how many files and chunks are indexed, and which files have changed
    since the last index.

    Args:
        project_path: Absolute path to the project root directory.

    Returns:
        Index status including files count, chunks count, and stale files list.
    """
    path = Path(project_path)
    if not path.exists():
        await ctx.warning(f"Project path does not exist: {project_path}")
        return {"error": f"Path does not exist: {project_path}"}

    status = indexer.get_status(path)

    return {
        "is_indexed": status.is_indexed,
        "last_updated": status.last_updated.isoformat() if status.last_updated else None,
        "files_count": status.files_count,
        "chunks_count": status.chunks_count,
        "stale_files_count": len(status.stale_files),
        "stale_files": status.stale_files[:20],  # Limit to first 20
    }


def run_server() -> None:
    """Run the MCP server."""
    log.info("starting_mcp_server", name=mcp.name)
    mcp.run()
