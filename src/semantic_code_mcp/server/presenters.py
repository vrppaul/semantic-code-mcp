"""From what the services report to what the tools answer."""

from pathlib import Path

from semantic_code_mcp.domain.models import (
    IndexManifest,
    IndexReport,
    IndexStatus,
    SearchHit,
    SearchReport,
)
from semantic_code_mcp.server.responses import (
    IndexCodebaseResponse,
    IndexingSummaryView,
    IndexManifestView,
    IndexStatusResponse,
    SearchCodeResponse,
    SearchDebugView,
    SearchResultView,
    SearchStatsView,
    SearchTimingsView,
)
from semantic_code_mcp.services.timing import MILLISECOND_DECIMALS, MILLISECONDS_PER_SECOND

# A hit's content is cut to this many lines; the agent reads the file for the rest
MAX_RESULT_LINES = 50
TRUNCATION_MARK = "\n... (truncated)"
SIMILARITY_DECIMALS = 3
# A rough count of tokens in English text and code
CHARACTERS_PER_TOKEN = 4


def present_search(
    report: SearchReport, project_root: Path, *, include_debug: bool
) -> SearchCodeResponse:
    """The answer of `search_code`, with timings and counts when asked for."""
    results = [present_hit(hit, project_root) for hit in report.results]
    return SearchCodeResponse(
        results=results,
        debug=present_search_debug(report, results) if include_debug else None,
    )


def present_hit(hit: SearchHit, project_root: Path) -> SearchResultView:
    """One hit, its file by its absolute path, its content cut to MAX_RESULT_LINES."""
    chunk = hit.chunk
    lines = chunk.content.split("\n")
    truncated = len(lines) > MAX_RESULT_LINES
    content = "\n".join(lines[:MAX_RESULT_LINES]) + TRUNCATION_MARK if truncated else chunk.content
    return SearchResultView(
        file_path=str(project_root / chunk.path),
        line_start=chunk.line_start,
        line_end=chunk.line_end,
        name=chunk.name,
        kind=chunk.kind,
        parent=chunk.parent,
        content=content,
        similarity=round(hit.similarity, SIMILARITY_DECIMALS),
        truncated=truncated,
    )


def present_search_debug(report: SearchReport, results: list[SearchResultView]) -> SearchDebugView:
    """The timings and counts behind one search."""
    indexing = report.index_report
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
            files_indexed=indexing.files_indexed,
            chunks_indexed=indexing.chunks_indexed,
        ),
    )


def present_index(report: IndexReport, project_root: Path) -> IndexCodebaseResponse:
    """The answer of `index_codebase` about the project at the root given."""
    return IndexCodebaseResponse(
        project_root=str(project_root),
        files_indexed=report.files_indexed,
        chunks_indexed=report.chunks_indexed,
        files_deleted=report.files_deleted,
        duration_seconds=report.duration_seconds,
    )


def present_status(status: IndexStatus, project_root: Path) -> IndexStatusResponse:
    """The answer of `index_status` about the project at the root given."""
    return IndexStatusResponse(
        project_root=str(project_root),
        is_indexed=status.is_indexed,
        last_updated=None if status.manifest is None else status.manifest.indexed_at.isoformat(),
        files_count=status.files_count,
        chunks_count=status.chunks_count,
        stale_files_count=status.stale_files_count,
        manifest=None if status.manifest is None else present_manifest(status.manifest),
    )


def present_manifest(manifest: IndexManifest) -> IndexManifestView:
    """What built an index, as `index_status` tells it."""
    return IndexManifestView(
        model_name=manifest.identity.model_name,
        model_revision=manifest.identity.model_revision,
        embedder_version=manifest.identity.embedder_version,
        chunker_version=manifest.identity.chunker_version,
        schema_version=manifest.schema_version,
    )
