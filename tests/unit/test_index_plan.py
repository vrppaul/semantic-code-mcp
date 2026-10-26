"""Tests for planning an index run: which files to index, which to drop, when to rebuild."""

from pathlib import Path

import pytest

from semantic_code_mcp.domain.models import IndexCounts, IndexDelta, RebuildReason, SourceFile
from semantic_code_mcp.services.indexing import plan_delta, rebuild_reason

UNCHANGED = SourceFile(path=Path("unchanged.py"), mtime_ns=100, size_bytes=10)
TOUCHED = SourceFile(path=Path("touched.py"), mtime_ns=200, size_bytes=10)
RESIZED = SourceFile(path=Path("resized.py"), mtime_ns=100, size_bytes=20)
NEW = SourceFile(path=Path("new.py"), mtime_ns=100, size_bytes=10)
GONE = SourceFile(path=Path("gone.py"), mtime_ns=100, size_bytes=10)
ON_DISK = [UNCHANGED, TOUCHED, RESIZED, NEW]
INDEXED = [
    UNCHANGED,
    SourceFile(path=TOUCHED.path, mtime_ns=100, size_bytes=10),
    SourceFile(path=RESIZED.path, mtime_ns=100, size_bytes=10),
    GONE,
]
EMPTY = IndexCounts(files=0, chunks=0)
FULL = IndexCounts(files=2, chunks=6)
CHUNKS_WITHOUT_RECORDS = IndexCounts(files=0, chunks=6)
RECORDS_WITHOUT_CHUNKS = IndexCounts(files=2, chunks=0)


def test_a_run_indexes_what_is_new_or_changed_and_drops_what_is_gone():
    delta = plan_delta(ON_DISK, INDEXED, every_file=False)

    assert delta == IndexDelta(to_index=[TOUCHED, RESIZED, NEW], to_delete=[GONE.path])


def test_a_forced_run_indexes_every_file_and_drops_what_is_gone():
    delta = plan_delta(ON_DISK, INDEXED, every_file=True)

    assert delta == IndexDelta(to_index=ON_DISK, to_delete=[GONE.path])


@pytest.mark.parametrize(
    ("counts", "reason"),
    [
        pytest.param(EMPTY, None, id="a new store, or every file deleted"),
        pytest.param(FULL, None, id="chunks and records"),
        pytest.param(RECORDS_WITHOUT_CHUNKS, None, id="files that gave no chunk"),
        pytest.param(CHUNKS_WITHOUT_RECORDS, RebuildReason.incomplete, id="chunks without records"),
    ],
)
def test_an_index_whose_whole_write_stopped_midway_is_rebuilt(
    counts: IndexCounts, reason: RebuildReason | None
):
    assert rebuild_reason(counts) == reason
