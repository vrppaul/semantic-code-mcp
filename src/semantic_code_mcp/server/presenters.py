"""From what the services report to what the tools answer."""

from semantic_code_mcp.domain.models import IndexResult, IndexStatus, SearchReport, SearchResult
from semantic_code_mcp.server.responses import (
    IndexCodebaseResponse,
    IndexingSummaryView,
    IndexStatusResponse,
    SearchCodeResponse,
    SearchDebugView,
    SearchResultView,
    SearchStatsView,
    SearchTimingsView,
)

# A hit's content is cut to this many lines; the agent reads the file for the rest
MAX_RESULT_LINES = 50
TRUNCATION_MARK = "\n... (truncated)"
SCORE_DECIMALS = 3
# A rough count of tokens in English text and code
CHARACTERS_PER_TOKEN = 4
MILLISECONDS_PER_SECOND = 1000
MILLISECOND_DECIMALS = 1


def present_search(report: SearchReport, *, include_debug: bool) -> SearchCodeResponse:
    """The answer of `search_code`, with timings and counts when asked for."""
    results = [present_hit(result) for result in report.results]
    return SearchCodeResponse(
        results=results,
        debug=present_search_debug(report, results) if include_debug else None,
    )


def present_hit(result: SearchResult) -> SearchResultView:
    """One hit, its content cut to MAX_RESULT_LINES."""
    lines = result.content.split("\n")
    truncated = len(lines) > MAX_RESULT_LINES
    content = "\n".join(lines[:MAX_RESULT_LINES]) + TRUNCATION_MARK if truncated else result.content
    return SearchResultView(
        file_path=result.file_path,
        line_start=result.line_start,
        line_end=result.line_end,
        name=result.name,
        chunk_type=result.chunk_type,
        content=content,
        score=round(result.score, SCORE_DECIMALS),
        truncated=truncated,
    )


def present_search_debug(report: SearchReport, results: list[SearchResultView]) -> SearchDebugView:
    """The timings and counts behind one search."""
    indexing = report.index_result
    return SearchDebugView(
        timings=SearchTimingsView(
            embedding_ms=report.timings.embedding_ms,
            search_ms=report.timings.search_ms,
            total_ms=report.timings.total_ms,
            indexing_ms=None
            if indexing is None
            else round(indexing.duration_seconds * MILLISECONDS_PER_SECOND, MILLISECOND_DECIMALS),
        ),
        stats=SearchStatsView(
            results_count=len(results),
            filtered_out=report.filtered_count,
            unique_files=len({result.file_path for result in results}),
            tokens_estimate=sum(len(result.content) for result in results) // CHARACTERS_PER_TOKEN,
        ),
        indexing=None
        if indexing is None
        else IndexingSummaryView(
            files_indexed=indexing.files_indexed, chunks_indexed=indexing.chunks_indexed
        ),
    )


def present_index(result: IndexResult) -> IndexCodebaseResponse:
    """The answer of `index_codebase`."""
    return IndexCodebaseResponse(
        files_indexed=result.files_indexed,
        chunks_indexed=result.chunks_indexed,
        files_deleted=result.files_deleted,
        duration_seconds=result.duration_seconds,
    )


def present_status(status: IndexStatus) -> IndexStatusResponse:
    """The answer of `index_status`."""
    return IndexStatusResponse(
        is_indexed=status.is_indexed,
        last_updated=None if status.last_updated is None else status.last_updated.isoformat(),
        files_count=status.files_count,
        chunks_count=status.chunks_count,
        stale_files_count=len(status.stale_files),
    )
