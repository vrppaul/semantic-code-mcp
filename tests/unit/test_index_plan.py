"""Tests for planning an index run: which files to index, which to drop."""

from pathlib import Path

from semantic_code_mcp.domain.models import FileStamp, IndexDelta, IndexedFiles, SourceFile
from semantic_code_mcp.services.indexing import plan_delta

FIRST = FileStamp(mtime_ns=100, size_bytes=10)
TOUCHED = FileStamp(mtime_ns=200, size_bytes=10)
RESIZED = FileStamp(mtime_ns=100, size_bytes=20)
ON_DISK = {"unchanged.py": FIRST, "touched.py": TOUCHED, "resized.py": RESIZED, "new.py": FIRST}
RECORDS = {"unchanged.py": FIRST, "touched.py": FIRST, "resized.py": FIRST, "gone.py": FIRST}
INDEXED = IndexedFiles(records=RECORDS, unrecorded_paths=[])
GONE = Path("gone.py")
# A write that stopped between its chunks and its records left a file's chunks unrecorded
ORPHAN = "orphan.py"


def test_a_run_indexes_what_is_new_or_changed_and_drops_what_is_gone():
    delta = plan_delta(ON_DISK, INDEXED, every_file=False)

    assert delta == IndexDelta(
        to_index=[
            SourceFile(path=Path("touched.py"), stamp=TOUCHED),
            SourceFile(path=Path("resized.py"), stamp=RESIZED),
            SourceFile(path=Path("new.py"), stamp=FIRST),
        ],
        to_delete=[GONE],
    )


def test_a_forced_run_indexes_every_file_and_drops_what_is_gone():
    delta = plan_delta(ON_DISK, INDEXED, every_file=True)

    assert delta == IndexDelta(
        to_index=[
            SourceFile(path=Path("unchanged.py"), stamp=FIRST),
            SourceFile(path=Path("touched.py"), stamp=TOUCHED),
            SourceFile(path=Path("resized.py"), stamp=RESIZED),
            SourceFile(path=Path("new.py"), stamp=FIRST),
        ],
        to_delete=[GONE],
    )


def test_a_run_drops_the_chunks_of_a_file_gone_before_it_was_recorded():
    indexed = IndexedFiles(records=RECORDS, unrecorded_paths=[ORPHAN])

    delta = plan_delta(ON_DISK, indexed, every_file=False)

    assert delta.to_delete == [GONE, Path(ORPHAN)]
