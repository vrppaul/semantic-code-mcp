"""API response models for MCP tool return types."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from semantic_code_mcp.models.domain import SearchResult

if TYPE_CHECKING:
    from semantic_code_mcp.services.search_service import SearchOutcome


class SearchTimings(BaseModel):
    """Timing breakdown for search operation."""

    embedding_ms: float
    search_ms: float
    total_ms: float
    indexing_ms: float | None = None


class SearchStats(BaseModel):
    """Statistics about search results."""

    results_count: int
    filtered_out: int
    unique_files: int
    tokens_estimate: int

    @classmethod
    def from_outcome(cls, outcome: SearchOutcome) -> SearchStats:
        total_content = "\n".join(r.content for r in outcome.results)
        tokens_estimate = len(total_content) // 4

        return cls(
            results_count=len(outcome.results),
            filtered_out=outcome.filtered_count,
            unique_files=len({r.file_path for r in outcome.results}),
            tokens_estimate=tokens_estimate,
        )


class IndexStatusSummary(BaseModel):
    """Summary of index status for search debug info."""

    files_count: int
    chunks_count: int
    was_stale: bool | None = None


class IndexResultSummary(BaseModel):
    """Summary of indexing result for search debug info."""

    files_indexed: int
    chunks_indexed: int


class SearchDebugInfo(BaseModel):
    """Debug information included in search response."""

    timings: SearchTimings
    stats: SearchStats
    index_status: IndexStatusSummary
    index_result: IndexResultSummary


class FormattedSearchResult(BaseModel):
    """A search result formatted for response."""

    file_path: str
    line_start: int
    line_end: int
    name: str
    chunk_type: str
    content: str
    score: float
    truncated: bool = False

    @classmethod
    def from_domain(
        cls,
        result: SearchResult,
        max_lines: int = 50,
    ) -> FormattedSearchResult:
        content = result.content
        lines = content.split("\n")
        truncated = len(lines) > max_lines
        if truncated:
            content = "\n".join(lines[:max_lines]) + "\n... (truncated)"

        return cls(
            file_path=result.file_path,
            line_start=result.line_start,
            line_end=result.line_end,
            name=result.name,
            chunk_type=result.chunk_type,
            content=content,
            score=round(result.score, 3),
            truncated=truncated,
        )


class SearchResponse(BaseModel):
    """Complete search response with results and debug info."""

    results: list[FormattedSearchResult]
    debug: SearchDebugInfo


class IndexCodebaseResponse(BaseModel):
    """Response from index_codebase tool."""

    files_indexed: int
    chunks_indexed: int
    files_deleted: int
    duration_seconds: float


class IndexStatusResponse(BaseModel):
    """Response from index_status tool."""

    is_indexed: bool
    last_updated: str | None
    files_count: int
    chunks_count: int


class ErrorResponse(BaseModel):
    """Error response for tool failures."""

    error: str
