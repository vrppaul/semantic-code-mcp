"""Where a project's index lives, and what every store does there alike.

A project's index directory holds an index for each identity — the chunker's version — and
layout — the store that keeps it and the version of that store's schema — each in a
directory of its own named after them, with its write lock beside it. So a session of
another version or store, or a server upgraded or rolled back, uses an index of its own and
never builds this one again, nor meets another store's files; switching back finds the old
one still there. The vectors of each model live inside the index, kept apart by a digest of
what made them. A store is built on its index's directory and owns only what lies inside it.

An index nobody used for UNUSED_INDEX_RETENTION is removed, and so is what 0.4 kept
directly in the project's directory. A write renames a new manifest into the index's
directory and a read touches a stamp there, at most once per USE_MARK_INTERVAL, so an
index's age is the newest change to its directory or to what lies directly inside it.
Each store looks for the project's unused indexes as often as it marks its own. It takes
an index's write lock without waiting, so it never removes one a writer holds, looks at
its age again under the lock, and renames the directory out of every store's reach
before deleting it: a writer that comes later writes a new index, which a run fills.
The lock lies beside the directory, not in it, so the rename works where an open file
pins its directory (Windows); lock files are never deleted, since a writer may be
waiting on one.

The project's index directory holds a `.gitignore` of `*`, so a project-local index
never shows in the project's own repository, nor what 0.4 left there.
"""

import contextlib
import hashlib
import math
import os
import shutil
import tempfile
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import ClassVar, override

import structlog
from filelock import FileLock
from pydantic import BaseModel, ConfigDict, ValidationError

from semantic_code_mcp.domain.interfaces import IndexStore
from semantic_code_mcp.domain.models import IndexIdentity, IndexManifest, StoreKind

log = structlog.get_logger()

# An index's directory, under the project's, named by its layout and its identity's digest;
# its write lock is a file beside it, of the same name with WRITE_LOCK_SUFFIX
INDEX_DIRECTORY_FORMAT = "schema-{schema_version}-{identity_digest}-{store_kind}"
INDEX_DIRECTORY_PATTERN = "schema-*"
# How many hex digits of a digest name an identity: an index's, or a model's in its index
IDENTITY_DIGEST_LENGTH = 16
WRITE_LOCK_SUFFIX = ".lock"
# An index nobody read or wrote for this long is removed: no session used its model or
# version for a week, and a rebuild costs minutes where keeping it costs the disk it fills
UNUSED_INDEX_RETENTION = timedelta(days=7)
# How often a store marks its index as used and looks for the project's unused indexes:
# rarely enough to cost nothing, far more often than an index takes to count as unused
USE_MARK_INTERVAL = timedelta(hours=1)
USE_STAMP_FILE = "last_used"
# Taken without waiting: an index whose writer holds its lock is in use
NO_WAIT_SECONDS = 0
# What 0.4 kept directly in a project's index directory: its chunks table and its record
# of mtimes, removed together
LEGACY_ENTRIES = ("chunks.lance", "file_mtimes.json")
# An index is renamed to this prefix before it is deleted, out of reach of every store
REMOVING_PREFIX = ".removing-"
MANIFEST_FILE = "manifest.json"
GITIGNORE_FILE = ".gitignore"
IGNORE_EVERYTHING = "*\n"
# Written beside its target, then renamed over it, so a crash never leaves half a file
TEMPORARY_SUFFIX = ".tmp"


class IndexLayout(BaseModel):
    """How a store lays an index out: the store's kind, and the version of its schema."""

    model_config = ConfigDict(frozen=True)

    store_kind: StoreKind
    # Bumped by any change to the store's tables or to what one is made from
    schema_version: int


class LeftIndex(BaseModel):
    """An index of the project this store does not use: the paths it is made of, and the
    lock its writers hold, which 0.4 had none of."""

    model_config = ConfigDict(frozen=True)

    parts: list[Path]
    write_lock: Path | None


class IndexDirectory:
    """The directory of a project's index of one identity in one layout: its write lock, its
    manifest, the stamp of its use, and the sweep of the project's indexes nobody uses.

    Nothing touches the disk until it is first used, and only a writer makes the directory.
    """

    def __init__(
        self,
        project_directory: Path,
        identity: IndexIdentity,
        layout: IndexLayout,
        *,
        lock_timeout_seconds: float,
    ) -> None:
        self._project_directory = project_directory
        self._identity = identity
        self._layout = layout
        self._path = project_directory / index_directory_name(identity, layout)
        self._lock_timeout_seconds = lock_timeout_seconds

    @property
    def path(self) -> Path:
        """Where the index lives."""
        return self._path

    @property
    def lock_timeout_seconds(self) -> float:
        """How long a write waits while another holds the index."""
        return self._lock_timeout_seconds

    @property
    def write_lock(self) -> Path:
        """The file a writer of this index locks, beside its directory."""
        return write_lock_of(self._path)

    def exists(self) -> bool:
        """Whether a write has made the directory."""
        return self._path.is_dir()

    @contextmanager
    def writing(self) -> Generator[None]:
        """Hold the index for one write: marked as used, the project's directory made for the
        write lock that lies in it, the lock taken within the timeout the directory was given,
        so two server processes on one index never write side by side, and the index's own
        directory made under it, as a removal renames an index only under that lock. A write
        that ends without an error records what built the index in its manifest, under the
        lock still."""
        self.note_use()
        self._project_directory.mkdir(parents=True, exist_ok=True)
        with FileLock(self.write_lock, timeout=self._lock_timeout_seconds):
            self._create()
            yield
            self._write_manifest()

    def _create(self) -> None:
        """Make the directory and the project's `.gitignore`, and mark the index as used, as
        a first write would leave the project's unused indexes for its first read otherwise."""
        self._path.mkdir(parents=True, exist_ok=True)
        ignore_file = self._project_directory / GITIGNORE_FILE
        if not ignore_file.exists():
            ignore_file.write_text(IGNORE_EVERYTHING)
        self.note_use()

    def manifest(self) -> IndexManifest | None:
        """What built the index; None before its first write, or where the manifest cannot
        be read, which the next write writes again."""
        path = self._path / MANIFEST_FILE
        if not path.is_file():
            return None
        try:
            return IndexManifest.model_validate_json(path.read_bytes())
        except ValidationError as error:
            log.warning("manifest_unreadable", directory=str(self._path), error=str(error))
            return None

    def _write_manifest(self) -> None:
        """Record the index's identity and layout as written now: the manifest whole, renamed
        over the old one."""
        manifest = IndexManifest(
            identity=self._identity,
            store_kind=self._layout.store_kind,
            schema_version=self._layout.schema_version,
            indexed_at=datetime.now(UTC),
        )
        write_atomically(self._path / MANIFEST_FILE, manifest.model_dump_json())

    def note_use(self) -> None:
        """Mark this index as used, so another store leaves it, and remove the project's
        indexes nobody used for UNUSED_INDEX_RETENTION, once its stamp is USE_MARK_INTERVAL
        old. The stamp's own mtime keeps that time, on the clock the remover reads, so a
        session whose machine slept for a week marks its index at its first read; and every
        process of one index shares the one stamp. An index not made yet has nothing to
        mark. Neither can fail a read: what cannot be done now is left for the next time."""
        stamp = self._path / USE_STAMP_FILE
        if not self._path.is_dir():
            return
        if seconds_since_change(stamp) < USE_MARK_INTERVAL.total_seconds():
            return
        try:
            stamp.touch()
        except OSError as error:
            log.debug("index_use_not_marked", directory=str(self._path), error=str(error))
        remove_unused_indexes(self._project_directory, self._path)


class DirectoryStore(IndexStore):
    """A store built on its index's directory, in the layout its class names, so whoever
    picks a store builds it alike without importing what keeps the index."""

    LAYOUT: ClassVar[IndexLayout]

    def __init__(self, directory: IndexDirectory) -> None:
        self._directory = directory

    @override
    def manifest(self) -> IndexManifest | None:
        return self._directory.manifest()


def index_directory_name(identity: IndexIdentity, layout: IndexLayout) -> str:
    """The name of the directory an index of this identity and layout lives in."""
    return INDEX_DIRECTORY_FORMAT.format(
        schema_version=layout.schema_version,
        identity_digest=identity_digest(identity),
        store_kind=layout.store_kind,
    )


def identity_digest(identity: BaseModel) -> str:
    """A short digest of what an identity holds, which names it on disk and in a table: the
    same for equal identities, another for any field changed."""
    digest = hashlib.sha256(identity.model_dump_json().encode()).hexdigest()
    return digest[:IDENTITY_DIGEST_LENGTH]


def write_lock_of(index_directory: Path) -> Path:
    return index_directory.with_name(index_directory.name + WRITE_LOCK_SUFFIX)


def remove_unused_indexes(project_directory: Path, own_directory: Path) -> None:
    """Remove the project's indexes nobody used for UNUSED_INDEX_RETENTION, and finish the
    removals that stopped after their rename. One that vanishes, is locked or cannot be
    removed meanwhile is left for the next time."""
    unused_before = time.time() - UNUSED_INDEX_RETENTION.total_seconds()
    for entry in project_directory.glob(f"{REMOVING_PREFIX}*"):
        try:
            delete_entry(entry)
        except OSError as error:
            log.debug("removal_not_finished", entry=str(entry), error=str(error))
    for left in left_indexes(project_directory, own_directory):
        parts = [str(part) for part in left.parts]
        try:
            if remove_if_unused(left, unused_before):
                log.info("unused_index_removed", parts=parts)
        except OSError as error:
            log.debug("unused_index_left", parts=parts, error=str(error))


def left_indexes(project_directory: Path, own_directory: Path) -> list[LeftIndex]:
    """The project's indexes other than this store's: one per identity directory, and the
    entries 0.4 kept in the project's directory, as one."""
    others = [
        LeftIndex(parts=[directory], write_lock=write_lock_of(directory))
        for directory in sorted(project_directory.glob(INDEX_DIRECTORY_PATTERN))
        if directory.is_dir() and directory != own_directory
    ]
    legacy = [
        project_directory / name for name in LEGACY_ENTRIES if (project_directory / name).exists()
    ]
    return [*others, LeftIndex(parts=legacy, write_lock=None)] if legacy else others


def remove_if_unused(left: LeftIndex, unused_before: float) -> bool:
    """Remove an index nobody used since `unused_before`, unless a writer holds its lock;
    whether it went. Its age is read again under the lock, which a writer that finished in
    between has moved on, and its parts are renamed out of reach before they are deleted."""
    if last_change(left) >= unused_before:
        return False
    lock = (
        contextlib.nullcontext()
        if left.write_lock is None
        else FileLock(left.write_lock, timeout=NO_WAIT_SECONDS)
    )
    with lock:
        if last_change(left) >= unused_before:
            return False
        moved = [part.rename(part.with_name(REMOVING_PREFIX + part.name)) for part in left.parts]
    for part in moved:
        delete_entry(part)
    return True


def last_change(left: LeftIndex) -> float:
    """When an index was last used, in seconds since the epoch: the latest change to any of
    its parts or to what lies directly inside one, where a write renames a new manifest
    and a read touches a stamp."""
    entries = [
        *left.parts,
        *(child for part in left.parts if part.is_dir() for child in part.iterdir()),
    ]
    return max(entry.lstat().st_mtime for entry in entries)


def seconds_since_change(path: Path) -> float:
    """How long ago a file last changed, by the wall clock; for ever if it cannot be read."""
    try:
        return time.time() - path.stat().st_mtime
    except OSError:
        return math.inf


def delete_entry(entry: Path) -> None:
    if entry.is_dir():
        shutil.rmtree(entry)
    else:
        entry.unlink(missing_ok=True)


def write_atomically(path: Path, text: str) -> None:
    """Write a file whole or not at all: beside it first, then renamed over it."""
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=TEMPORARY_SUFFIX)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as temporary_file:
            temporary_file.write(text)
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
