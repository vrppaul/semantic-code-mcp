"""Tests for planning an index run: which files to index, which to drop, when to rebuild."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from semantic_code_mcp.domain.models import (
    IndexCounts,
    IndexDelta,
    IndexIdentity,
    IndexManifest,
    RebuildReason,
    SourceFile,
)
from semantic_code_mcp.services.indexing import plan_delta, rebuild_reason

UNCHANGED = SourceFile(path=Path("/project/unchanged.py"), mtime_ns=100, size_bytes=10)
TOUCHED = SourceFile(path=Path("/project/touched.py"), mtime_ns=200, size_bytes=10)
RESIZED = SourceFile(path=Path("/project/resized.py"), mtime_ns=100, size_bytes=20)
NEW = SourceFile(path=Path("/project/new.py"), mtime_ns=100, size_bytes=10)
GONE = SourceFile(path=Path("/project/gone.py"), mtime_ns=100, size_bytes=10)
ON_DISK = [UNCHANGED, TOUCHED, RESIZED, NEW]
INDEXED = [
    UNCHANGED,
    SourceFile(path=TOUCHED.path, mtime_ns=100, size_bytes=10),
    SourceFile(path=RESIZED.path, mtime_ns=100, size_bytes=10),
    GONE,
]
IDENTITY = IndexIdentity(model_name="model", chunker_version=1, schema_version=1)
WRITTEN_AT = datetime(2026, 10, 25, tzinfo=UTC)
EMPTY = IndexCounts(files=0, chunks=0)
FULL = IndexCounts(files=2, chunks=6)
CHUNKS_WITHOUT_RECORDS = IndexCounts(files=0, chunks=6)
RECORDS_WITHOUT_CHUNKS = IndexCounts(files=2, chunks=0)


def built_from(model_name: str, chunker_version: int, schema_version: int) -> IndexManifest:
    return IndexManifest(
        identity=IndexIdentity(
            model_name=model_name, chunker_version=chunker_version, schema_version=schema_version
        ),
        indexed_at=WRITTEN_AT,
    )


def test_a_run_indexes_what_is_new_or_changed_and_drops_what_is_gone():
    delta = plan_delta(ON_DISK, INDEXED, every_file=False)

    assert delta == IndexDelta(to_index=[TOUCHED, RESIZED, NEW], to_delete=[GONE.path])


def test_a_forced_run_indexes_every_file_and_drops_what_is_gone():
    delta = plan_delta(ON_DISK, INDEXED, every_file=True)

    assert delta == IndexDelta(to_index=ON_DISK, to_delete=[GONE.path])


@pytest.mark.parametrize(
    ("manifest", "counts", "reason"),
    [
        pytest.param(None, EMPTY, None, id="a new store"),
        pytest.param(None, FULL, RebuildReason.unversioned, id="data without a manifest"),
        pytest.param(built_from("model", 1, 1), FULL, None, id="built from the same"),
        pytest.param(
            built_from("model", 1, 1),
            CHUNKS_WITHOUT_RECORDS,
            RebuildReason.incomplete,
            id="chunks without records",
        ),
        pytest.param(built_from("model", 1, 1), EMPTY, None, id="every file deleted"),
        pytest.param(built_from("model", 1, 1), RECORDS_WITHOUT_CHUNKS, None, id="no chunk"),
        pytest.param(
            built_from("other", 1, 1),
            CHUNKS_WITHOUT_RECORDS,
            RebuildReason.model_changed,
            id="another model's chunks without records",
        ),
        pytest.param(
            built_from("model", 1, 0), FULL, RebuildReason.schema_changed, id="another schema"
        ),
        pytest.param(
            built_from("other", 1, 1), FULL, RebuildReason.model_changed, id="another model"
        ),
        pytest.param(
            built_from("model", 0, 1), FULL, RebuildReason.chunker_changed, id="another chunker"
        ),
    ],
)
def test_an_index_is_rebuilt_when_what_built_it_changed(
    manifest: IndexManifest | None, counts: IndexCounts, reason: RebuildReason | None
):
    assert rebuild_reason(manifest, counts=counts, expected=IDENTITY) == reason
