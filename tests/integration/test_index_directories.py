"""Tests of an index's directory, whatever store keeps the index in it: where an index of an
identity and a layout lies, its write lock and manifest, the stamp of its use, and the sweep
of the project's indexes nobody used for a week, 0.4's leftovers among them."""

import os
from collections.abc import Callable
from pathlib import Path

import pytest
from filelock import FileLock, Timeout

from semantic_code_mcp.domain.models import IndexIdentity, StoreKind
from semantic_code_mcp.storage.index_directories import (
    LEGACY_ENTRIES,
    MANIFEST_FILE,
    REMOVING_PREFIX,
    USE_MARK_INTERVAL,
    USE_STAMP_FILE,
    IndexDirectory,
    IndexLayout,
)
from tests.stores import IDENTITY, LOCK_TIMEOUT_SECONDS, OTHER_IDENTITY

LAYOUT = IndexLayout(store_kind=StoreKind.sqlite, schema_version=1)
ANOTHER_STORE = IndexLayout(store_kind=StoreKind.lancedb, schema_version=1)
ANOTHER_SCHEMA = IndexLayout(store_kind=StoreKind.sqlite, schema_version=2)
# What a store keeps in its index's directory, as far as these tests care
TABLE_FILE = "table"
# Long enough to tell a write that waits from one that does not, short for a test
SHORT_LOCK_TIMEOUT_SECONDS = 0.2
# A time long before any test runs: an index last used then is long unused
LONG_AGO_SECONDS = 1_000_000_000


class FailedWriteError(RuntimeError):
    """A store's write that failed inside the directory's hold."""


def directory_of(
    project_directory: Path,
    identity: IndexIdentity = IDENTITY,
    layout: IndexLayout = LAYOUT,
    lock_timeout_seconds: float = LOCK_TIMEOUT_SECONDS,
) -> IndexDirectory:
    return IndexDirectory(
        project_directory, identity, layout, lock_timeout_seconds=lock_timeout_seconds
    )


def write_into(directory: IndexDirectory) -> None:
    """One write of a store into the directory: a file of its tables."""
    with directory.writing():
        (directory.path / TABLE_FILE).write_text("rows")


@pytest.mark.parametrize(
    ("identity", "layout"),
    [
        pytest.param(OTHER_IDENTITY, LAYOUT, id="another identity"),
        pytest.param(IDENTITY, ANOTHER_STORE, id="another store"),
        pytest.param(IDENTITY, ANOTHER_SCHEMA, id="another schema"),
    ],
)
def test_an_index_of_another_identity_or_layout_lies_in_a_directory_of_its_own(
    tmp_path: Path, identity: IndexIdentity, layout: IndexLayout
):
    """Two models, two stores or two versions of a schema on one project never meet each
    other's files."""
    own = directory_of(tmp_path)

    other = directory_of(tmp_path, identity, layout)

    assert other.path != own.path


def test_a_write_records_what_built_the_index(tmp_path: Path):
    directory = directory_of(tmp_path)
    before = directory.manifest()

    write_into(directory)

    after = directory.manifest()
    assert (before, after.identity, after.schema_version) == (
        None,
        IDENTITY,
        LAYOUT.schema_version,
    )


def test_a_write_that_fails_records_nothing(tmp_path: Path):
    directory = directory_of(tmp_path)

    with pytest.raises(FailedWriteError), directory.writing():
        raise FailedWriteError

    assert directory.manifest() is None


def test_a_write_waits_while_another_process_writes(tmp_path: Path):
    directory = directory_of(tmp_path, lock_timeout_seconds=SHORT_LOCK_TIMEOUT_SECONDS)

    with FileLock(directory.write_lock), pytest.raises(Timeout):
        write_into(directory)

    assert not directory.exists()


def write_bytes_that_are_not_utf8(manifest: Path) -> None:
    manifest.write_bytes(b'{"identity": "\xff"}')


def make_a_directory(manifest: Path) -> None:
    manifest.mkdir()


@pytest.mark.parametrize(
    "spoil",
    [
        pytest.param(write_bytes_that_are_not_utf8, id="not UTF-8"),
        pytest.param(make_a_directory, id="a directory"),
    ],
)
def test_a_manifest_that_cannot_be_read_counts_as_none(
    tmp_path: Path, spoil: Callable[[Path], None]
):
    directory = directory_of(tmp_path)
    write_into(directory)
    manifest = directory.path / MANIFEST_FILE
    manifest.unlink()

    spoil(manifest)

    assert directory.manifest() is None


def leave_an_index_of_another_identity(project_directory: Path) -> list[str]:
    """Another model's index of the project; the names it made in the project's directory."""
    other = directory_of(project_directory, OTHER_IDENTITY)
    write_into(other)
    return [other.path.name]


def leave_an_index_of_another_store(project_directory: Path) -> list[str]:
    """The index another store kept of the project, for this identity."""
    other = directory_of(project_directory, IDENTITY, ANOTHER_STORE)
    write_into(other)
    return [other.path.name]


def leave_an_index_of_version_0_4(project_directory: Path) -> list[str]:
    """What 0.4 kept in the project's directory: a chunks table and a record of mtimes."""
    table, records = (project_directory / name for name in LEGACY_ENTRIES)
    table.mkdir(parents=True)
    (table / TABLE_FILE).write_text("rows")
    records.write_text("{}")
    return list(LEGACY_ENTRIES)


LEFT_INDEXES = [
    pytest.param(leave_an_index_of_another_identity, id="another identity"),
    pytest.param(leave_an_index_of_another_store, id="another store"),
    pytest.param(leave_an_index_of_version_0_4, id="version 0.4"),
]


@pytest.mark.parametrize("leave_index", LEFT_INDEXES)
def test_an_index_used_within_the_week_is_left_alone(
    tmp_path: Path, leave_index: Callable[[Path], list[str]]
):
    """Another session may still use it; a store reads and writes its own index only."""
    left = leave_index(tmp_path)

    write_into(directory_of(tmp_path))

    assert set(left) <= set(names_in(tmp_path))


def mark_used(directory: IndexDirectory) -> None:
    directory.note_use()


@pytest.mark.parametrize("leave_index", LEFT_INDEXES)
@pytest.mark.parametrize(
    "use_own_index",
    [pytest.param(write_into, id="written"), pytest.param(mark_used, id="read")],
)
def test_an_index_nobody_used_for_a_week_is_removed_when_another_is_used(
    tmp_path: Path,
    leave_index: Callable[[Path], list[str]],
    use_own_index: Callable[[IndexDirectory], None],
):
    """A week after both were last used, a session of this index reads or writes it."""
    own = directory_of(tmp_path)
    write_into(own)
    left = leave_index(tmp_path)
    set_mtime_of_tree(tmp_path, LONG_AGO_SECONDS)

    use_own_index(own)

    assert set(names_in(tmp_path)).isdisjoint(left)


def test_an_index_only_read_within_the_week_is_left_alone(tmp_path: Path):
    """A session that searches a project nobody edits keeps its index, unwritten for weeks."""
    left = leave_an_index_of_another_identity(tmp_path)
    directory_of(tmp_path, OTHER_IDENTITY).note_use()
    set_mtime_of_tree(tmp_path, LONG_AGO_SECONDS)
    directory_of(tmp_path, OTHER_IDENTITY).note_use()

    write_into(directory_of(tmp_path))

    assert set(left) <= set(names_in(tmp_path))


def test_an_index_is_marked_again_once_its_stamp_is_older_than_the_interval(tmp_path: Path):
    """However long the machine slept since a session last looked: the stamp tells."""
    directory = directory_of(tmp_path)
    write_into(directory)
    stamp = directory.path / USE_STAMP_FILE
    long_before = stamp.stat().st_mtime - 2 * USE_MARK_INTERVAL.total_seconds()
    os.utime(stamp, (long_before, long_before))

    directory.note_use()

    assert stamp.stat().st_mtime > long_before + USE_MARK_INTERVAL.total_seconds()


def test_an_unused_index_that_another_process_is_writing_is_left_alone(tmp_path: Path):
    other = directory_of(tmp_path, OTHER_IDENTITY)
    write_into(other)
    set_mtime_of_tree(tmp_path, LONG_AGO_SECONDS)

    with FileLock(other.write_lock):
        write_into(directory_of(tmp_path))

    assert other.exists()


def test_a_removal_stopped_midway_is_finished_by_the_next_use(tmp_path: Path):
    """An index is renamed before it is deleted; a crash between the two leaves the name."""
    half_removed = tmp_path / f"{REMOVING_PREFIX}schema-1-0123456789abcdef"
    half_removed.mkdir(parents=True)
    (half_removed / MANIFEST_FILE).write_text("{}")

    write_into(directory_of(tmp_path))

    assert half_removed.name not in names_in(tmp_path)


def names_in(directory: Path) -> list[str]:
    return sorted(entry.name for entry in directory.iterdir())


def set_mtime_of_tree(directory: Path, seconds: int) -> None:
    """Set the modification time of a directory and of everything under it."""
    for path in [directory, *directory.rglob("*")]:
        os.utime(path, (seconds, seconds))
