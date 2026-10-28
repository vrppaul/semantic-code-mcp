"""From what the services report to what the tools answer."""

from pathlib import Path

from semantic_code_mcp.domain.models import (
    Chunk,
    IndexManifest,
    IndexReport,
    IndexStatus,
    SearchReport,
    VectorIdentity,
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
from semantic_code_mcp.services.projects import ProjectLocation
from semantic_code_mcp.services.timing import MILLISECOND_DECIMALS, MILLISECONDS_PER_SECOND

# A hit's content is cut to this many lines; the agent reads the file for the rest
MAX_RESULT_LINES = 50
TRUNCATION_MARK = "\n... (truncated)"
# What ends a line of a chunk's content, which the cut counts by
LINE_SEPARATOR = "\n"
# A rough count of tokens in English text and code
CHARACTERS_PER_TOKEN = 4


def present_search(
    report: SearchReport, location: ProjectLocation, *, include_debug: bool
) -> SearchCodeResponse:
    """The answer of `search_code` asked at a location, with timings and counts when asked
    for."""
    results = [present_hit(chunk, location.root) for chunk in report.results]
    return SearchCodeResponse(
        project_root=str(location.root),
        searched_directory=str(location.absolute_directory),
        results=results,
        debug=present_search_debug(report, results) if include_debug else None,
    )


def present_hit(chunk: Chunk, project_root: Path) -> SearchResultView:
    """One hit, its file by its absolute path, its content cut to MAX_RESULT_LINES."""
    lines = chunk.content.split(LINE_SEPARATOR)
    truncated = len(lines) > MAX_RESULT_LINES
    content = (
        LINE_SEPARATOR.join(lines[:MAX_RESULT_LINES]) + TRUNCATION_MARK
        if truncated
        else chunk.content
    )
    return SearchResultView(
        file_path=str(project_root / chunk.path),
        line_start=chunk.line_start,
        line_end=chunk.line_end,
        name=chunk.name,
        kind=chunk.kind,
        parent=chunk.parent,
        content=content,
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
        embedded_chunks_count=status.chunks_with_vectors_count,
        stale_files_count=status.stale_files_count,
        manifest=None
        if status.manifest is None
        else present_manifest(status.manifest, status.model),
    )


def present_manifest(manifest: IndexManifest, model: VectorIdentity) -> IndexManifestView:
    """What built an index, and the model whose vectors in it the status counts, as
    `index_status` tells them."""
    return IndexManifestView(
        model_name=model.model_name,
        model_revision=model.model_revision,
        embedder_version=model.embedder_version,
        chunker_version=manifest.identity.chunker_version,
        schema_version=manifest.schema_version,
    )
