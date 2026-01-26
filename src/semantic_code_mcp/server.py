"""FastMCP server and tool definitions."""

import asyncio
import time
from collections import defaultdict
from pathlib import Path

import structlog
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession

from semantic_code_mcp.config import Settings, get_index_path
from semantic_code_mcp.indexer.embedder import Embedder
from semantic_code_mcp.indexer.indexer import Indexer
from semantic_code_mcp.logging import configure_logging
from semantic_code_mcp.models import IndexProgress, IndexResult
from semantic_code_mcp.profiling import configure_profiling, profile_async
from semantic_code_mcp.storage.lancedb import VectorStore

# Create settings and configure logging
settings = Settings()
configure_logging(debug=settings.debug)
configure_profiling(enabled=settings.profile)

log = structlog.get_logger()

indexer = Indexer(settings)
embedder = Embedder(settings)

# Pre-load embedding model to avoid cold start penalty on first search
# This adds ~2s to server startup but makes first search much faster
embedder.load()

# Create MCP server
mcp = FastMCP("semantic-code-mcp")


async def _do_index_with_progress(
    ctx: Context[ServerSession, None],
    path: Path,
    force: bool,
) -> IndexResult:
    """Run indexing with progress updates."""
    result: IndexResult | None = None

    async for update in indexer.index(path, force=force):
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
@profile_async("search_code")
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

    # Search the vector store using hybrid search (vector + full-text)
    t0 = time.perf_counter()
    store = VectorStore(index_path)

    # Use hybrid search to combine semantic and keyword matching
    # This ensures exact keyword matches surface even with low semantic similarity
    raw_results = await asyncio.to_thread(
        store.search_hybrid,
        query_embedding,
        query,  # Use original query for FTS
        limit * 2,
        0.5,  # 50% vector, 50% FTS weight
    )

    # Filter low-confidence results (score < 0.3 is essentially noise)
    filtered = [r for r in raw_results if r.score >= 0.3]

    # Apply recency boost only (keyword boost is handled by hybrid FTS)
    boosted = []
    for r in filtered:
        try:
            mtime = Path(r.file_path).stat().st_mtime
        except OSError:
            mtime = None

        # Small recency boost for recently modified files
        recency_boost = 0.0
        if mtime is not None:
            now = time.time()
            age_seconds = now - mtime
            one_week = 7 * 24 * 60 * 60
            if age_seconds < one_week:
                recency_boost = 0.05 * (1 - age_seconds / one_week)

        boosted_score = min(1.0, r.score + recency_boost)
        boosted.append((r, boosted_score))

    # Re-sort by boosted score and take limit
    boosted.sort(key=lambda x: x[1], reverse=True)
    filtered = [r for r, _ in boosted[:limit]]

    # Group by file, order files by best score, chunks by score within file
    by_file: dict[str, list] = defaultdict(list)
    for r in filtered:
        by_file[r.file_path].append(r)
    # Sort files by their best chunk's score (descending)
    sorted_files = sorted(by_file.keys(), key=lambda f: by_file[f][0].score, reverse=True)
    # Flatten back, chunks within each file already ordered by score from vector search
    results = []
    for f in sorted_files:
        results.extend(by_file[f])

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
            "filtered_out": len(raw_results) - len(filtered),
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
    max_lines = 50
    formatted_results = []
    for r in results:
        content = r.content
        lines = content.split("\n")
        truncated = False
        if len(lines) > max_lines:
            content = "\n".join(lines[:max_lines]) + "\n... (truncated)"
            truncated = True
        formatted_results.append(
            {
                "file_path": r.file_path,
                "line_start": r.line_start,
                "line_end": r.line_end,
                "name": r.name,
                "chunk_type": r.chunk_type.value,
                "content": content,
                "score": round(r.score, 3),
                "truncated": truncated,
            }
        )

    return {
        "result": formatted_results,
        "debug": debug,
    }


@mcp.tool()
@profile_async("index_codebase")
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
@profile_async("index_status")
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
