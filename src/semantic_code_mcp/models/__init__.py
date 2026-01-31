"""Domain models and API response types."""

from semantic_code_mcp.models.domain import (
    Chunk,
    ChunkType,
    ChunkWithEmbedding,
    FileChanges,
    IndexResult,
    IndexStatus,
    ScanPlan,
    SearchResult,
)
from semantic_code_mcp.models.responses import (
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

__all__ = [
    # Domain
    "Chunk",
    "ChunkType",
    "ChunkWithEmbedding",
    # Responses
    "ErrorResponse",
    "FileChanges",
    "FormattedSearchResult",
    "IndexCodebaseResponse",
    "IndexResult",
    "IndexResultSummary",
    "IndexStatus",
    "IndexStatusResponse",
    "IndexStatusSummary",
    "ScanPlan",
    "SearchDebugInfo",
    "SearchResponse",
    "SearchResult",
    "SearchStats",
    "SearchTimings",
]
