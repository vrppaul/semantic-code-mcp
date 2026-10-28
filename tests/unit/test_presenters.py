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
    SearchReport,
    SearchTimings,
    StoreKind,
    VectorIdentity,
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
AT_THE_ROOT = ProjectLocation(root=PROJECT_ROOT, directory=None, is_repository=True)


def hit_of(content: str) -> Chunk:
    return Chunk(
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


def test_a_status_names_the_chunker_that_built_the_index_and_the_model_of_its_vectors():
    manifest = IndexManifest(
        identity=IndexIdentity(chunker_version=3),
        store_kind=StoreKind.sqlite,
        schema_version=2,
        indexed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    status = IndexStatus(
        files_count=1,
        chunks_count=2,
        model=VectorIdentity(model_name="model", model_revision="revision", embedder_version=4),
        chunks_with_vectors_count=1,
        stale_files_count=0,
        manifest=manifest,
    )

    view = present_status(status, PROJECT_ROOT)

    assert (
        view.manifest.model_name,
        view.manifest.model_revision,
        view.manifest.embedder_version,
        view.manifest.chunker_version,
        view.embedded_chunks_count,
    ) == ("model", "revision", 4, 3, 1)
