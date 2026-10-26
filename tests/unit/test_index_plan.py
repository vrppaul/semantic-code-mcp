"""Tests for planning an index run: which files to index, which to drop."""

from pathlib import Path

from semantic_code_mcp.domain.models import IndexDelta, SourceFile
from semantic_code_mcp.services.indexing import plan_delta

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
# The files the chunks of the index come from, as the records name them
CHUNKED = {file.path for file in INDEXED}
# A write that stopped between its chunks and its records left a file's chunks unrecorded
ORPHAN = Path("orphan.py")


def test_a_run_indexes_what_is_new_or_changed_and_drops_what_is_gone():
    delta = plan_delta(ON_DISK, INDEXED, CHUNKED, every_file=False)

    assert delta == IndexDelta(to_index=[TOUCHED, RESIZED, NEW], to_delete=[GONE.path])


def test_a_forced_run_indexes_every_file_and_drops_what_is_gone():
    delta = plan_delta(ON_DISK, INDEXED, CHUNKED, every_file=True)

    assert delta == IndexDelta(to_index=ON_DISK, to_delete=[GONE.path])


def test_a_run_drops_the_chunks_of_a_file_gone_before_it_was_recorded():
    delta = plan_delta(ON_DISK, INDEXED, {*CHUNKED, ORPHAN}, every_file=False)

    assert delta.to_delete == [GONE.path, ORPHAN]
