"""Tests for what the tools answer, made from what the services report."""

from datetime import UTC, datetime
from pathlib import Path

from semantic_code_mcp.domain.models import (
    Chunk,
    ChunkKind,
    IndexIdentity,
    IndexManifest,
    IndexStatus,
    Language,
    SearchHit,
    SearchReport,
    SearchTimings,
)
from semantic_code_mcp.server.presenters import (
    MAX_RESULT_LINES,
    present_hit,
    present_search,
    present_status,
)
from semantic_code_mcp.services.projects import ProjectLocation

TIMINGS = SearchTimings(embedding_ms=1.0, search_ms=2.0, total_ms=3.0)
PROJECT_ROOT = Path("/project")
AT_THE_ROOT = ProjectLocation(root=PROJECT_ROOT, directory=None)


def hit_of(content: str) -> SearchHit:
    chunk = Chunk(
        chunk_id="module-method",
        path=Path("src/module.py"),
        language=Language.python,
        kind=ChunkKind.method,
        name="method",
        parent="Module",
        line_start=1,
        line_end=content.count("\n") + 1,
        content=content,
    )
    return SearchHit(
        chunk=chunk, similarity=0.5, vector_rank=1, keyword_rank=None, fused_score=1 / 61
    )


def test_a_hit_names_its_file_by_its_absolute_path_and_says_what_it_is_part_of():
    view = present_hit(hit_of("def method(self): pass"), PROJECT_ROOT)

    assert (view.file_path, view.kind, view.parent) == (
        str(PROJECT_ROOT / "src" / "module.py"),
        "method",
        "Module",
    )


def test_a_long_hit_is_cut_to_the_line_limit_and_says_so():
    long_content = "\n".join(f"line_{number} = {number}" for number in range(MAX_RESULT_LINES + 10))

    view = present_hit(hit_of(long_content), PROJECT_ROOT)

    assert view.truncated
    assert (
        view.content.split("\n")[:MAX_RESULT_LINES] == long_content.split("\n")[:MAX_RESULT_LINES]
    )


def test_a_short_hit_is_whole():
    view = present_hit(hit_of("def function():\n    return 1"), PROJECT_ROOT)

    assert (view.truncated, view.content) == (False, "def function():\n    return 1")


def test_debug_details_only_when_asked():
    report = SearchReport(
        results=[hit_of("x = 1")], filtered_count=0, timings=TIMINGS, index_report=None
    )

    assert present_search(report, AT_THE_ROOT, include_debug=False).debug is None
    assert present_search(report, AT_THE_ROOT, include_debug=True).debug is not None


def test_a_status_names_the_model_its_revision_and_the_embedder_that_built_the_index():
    identity = IndexIdentity(
        model_name="model", model_revision="revision", embedder_version=2, chunker_version=3
    )
    manifest = IndexManifest(
        identity=identity, schema_version=2, indexed_at=datetime(2026, 1, 1, tzinfo=UTC)
    )
    status = IndexStatus(files_count=1, chunks_count=1, stale_files_count=0, manifest=manifest)

    view = present_status(status, PROJECT_ROOT).manifest

    assert (view.model_name, view.model_revision, view.embedder_version) == ("model", "revision", 2)
