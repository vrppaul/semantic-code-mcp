"""LanceDB storage of one project's index: the chunks with their vectors, the record of
each file they came from, and the manifest of what built them.

A project's index directory holds an index for each identity — the model and its
revision, the chunker version — and layout, each in a directory of its own named after
them, with its write lock beside it. So a session of another model, revision or version,
or a server upgraded or rolled back, uses an index of its own and never builds this one
again; switching back finds the old one still there.

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

A write is one call, under the index's write lock, so two server processes on one index
never write side by side. Its chunks take the place of the chunks of the files it writes
and removes in one merge keyed by chunk id, so a reader finds a file's old chunks or its
new ones; then their records; the manifest last. The chunks table is made by
the first write that has vectors, as wide as they are; vectors of another width (a model
replaced under a name nothing pins) drop the records, then the table, and the index starts
again from the write, its other files unrecorded so the next run indexes them.

Reads take no lock: each opens its table again, at its latest version, so a long-lived
connection still sees what another process wrote, and a table not yet made reads as
empty. So old versions are pruned only once they are older than any read can last. The
upkeep that compacts the tables, folds new rows into the full-text index and prunes runs
when an index run that wrote asks for it: rows the full-text index has not folded in
score on another scale, so a run leaves none. Each upkeep adds the index a part that no
later one merges, so now and then it rebuilds the index whole.

The project's index directory holds a `.gitignore` of `*`, so a project-local index
never shows in the project's own repository, nor what 0.4 left there.
"""

import contextlib
import hashlib
import math
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import override

import lancedb
import numpy
import pyarrow
import structlog
from filelock import FileLock
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, ValidationError

from semantic_code_mcp.domain.globs import PATH_SEPARATOR
from semantic_code_mcp.domain.interfaces import IndexStore, Vector, VectorMatrix
from semantic_code_mcp.domain.models import (
    HIGHEST_SIMILARITY,
    LOWEST_SIMILARITY,
    Candidate,
    Chunk,
    ChunkKind,
    IndexCounts,
    IndexIdentity,
    IndexManifest,
    IndexUpdate,
    Language,
    SearchCandidates,
    SearchQuery,
    SearchScope,
    SourceFile,
)
from semantic_code_mcp.storage.globs import glob_pattern
from semantic_code_mcp.storage.search_text import search_text

log = structlog.get_logger()

# Bumped by any change to the tables' columns or to what one is made from, its full-text
# index among them (5: stop words kept); each version keeps an index of its own
SCHEMA_VERSION = 5

# An index's directory, under the project's, named by its layout and its identity's digest;
# its write lock is a file beside it, of the same name with WRITE_LOCK_SUFFIX
INDEX_DIRECTORY_FORMAT = "schema-{schema_version}-{identity_digest}"
INDEX_DIRECTORY_PATTERN = "schema-*"
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

CHUNKS_TABLE = "chunks"
FILES_TABLE = "files"
MANIFEST_FILE = "manifest.json"
GITIGNORE_FILE = ".gitignore"
IGNORE_EVERYTHING = "*\n"
# Written beside its target, then renamed over it, so a crash never leaves half a file
TEMPORARY_SUFFIX = ".tmp"

# Each upkeep folds new rows into the full-text index as a part of their own, which no later
# upkeep merges: a store rebuilds the index whole at its first upkeep and once in this many
# after (on 0.30.2 a rebuild took 63-75 ms on click, 1.2-2.1 s at 100k chunks)
UPKEEPS_PER_FULL_TEXT_REBUILD = 20
# Versions older than this are pruned by upkeep; a read finishes in milliseconds
OLD_VERSION_RETENTION = timedelta(seconds=10)

# Both tables name a file by its path from the project root, in POSIX form
PATH_COLUMN = "path"
CHUNK_ID_COLUMN = "chunk_id"
VECTOR_COLUMN = "vector"
LANGUAGE_COLUMN = "language"
KIND_COLUMN = "kind"
NAME_COLUMN = "name"
PARENT_COLUMN = "parent"
LINE_START_COLUMN = "line_start"
LINE_END_COLUMN = "line_end"
CONTENT_COLUMN = "content"
# What the full-text index holds of a chunk, which only it reads (`search_text.py`)
SEARCH_TEXT_COLUMN = "search_text"
MTIME_COLUMN = "mtime_ns"
SIZE_COLUMN = "size_bytes"
# The axis of a vector matrix along one vector: rows are chunks, columns components
VECTOR_LENGTH_AXIS = 1
# The length of a vector of zeros, which has no direction to compare
NO_LENGTH = 0.0
FULL_TEXT_QUERY_TYPE = "fts"
FULL_TEXT_INDEX_TYPE = "FTS"
# The full-text index keeps the words LanceDB drops by default (`a`, `the`, `is`, `not`):
# the questions the harness chooses by gained 2.1 in chunk reciprocal-rank sum, the
# held-out ones lost 0.5, for a full-text index 6-8% larger (ADR 006, stage 6)
REMOVE_STOP_WORDS = False
COSINE_METRIC = "cosine"
# What LanceDB adds to a hit, the distance of a vector hit and the score of a text hit:
# named among the columns a search reads, or each search warns that a later LanceDB will
# leave them out
DISTANCE_COLUMN = "_distance"
FULL_TEXT_SCORE_COLUMN = "_score"
# What a search finds when the index holds no chunks yet
NO_CANDIDATES = SearchCandidates(by_vector=[], by_keyword=[])
# How a single quote is written inside a quoted SQL string
SQL_QUOTE = "'"
SQL_ESCAPED_QUOTE = "''"
# How a filter's conditions are joined, one negated, and a column matched by a regex
SQL_AND = " AND "
SQL_NOT = "NOT "
REGEX_MATCH_FUNCTION = "regexp_like"
PREFIX_MATCH_FUNCTION = "starts_with"

FILES_SCHEMA = pyarrow.schema(
    [
        pyarrow.field(PATH_COLUMN, pyarrow.utf8()),
        pyarrow.field(MTIME_COLUMN, pyarrow.int64()),
        pyarrow.field(SIZE_COLUMN, pyarrow.int64()),
    ]
)


def chunks_schema(dimension: int) -> pyarrow.Schema:
    """The chunks table's columns, its vectors `dimension` wide."""
    return pyarrow.schema(
        [
            pyarrow.field(CHUNK_ID_COLUMN, pyarrow.utf8()),
            pyarrow.field(VECTOR_COLUMN, pyarrow.list_(pyarrow.float32(), dimension)),
            pyarrow.field(PATH_COLUMN, pyarrow.utf8()),
            pyarrow.field(LANGUAGE_COLUMN, pyarrow.utf8()),
            pyarrow.field(KIND_COLUMN, pyarrow.utf8()),
            pyarrow.field(NAME_COLUMN, pyarrow.utf8()),
            pyarrow.field(PARENT_COLUMN, pyarrow.utf8()),
            pyarrow.field(LINE_START_COLUMN, pyarrow.int32()),
            pyarrow.field(LINE_END_COLUMN, pyarrow.int32()),
            pyarrow.field(CONTENT_COLUMN, pyarrow.utf8()),
            pyarrow.field(SEARCH_TEXT_COLUMN, pyarrow.utf8()),
        ]
    )


class StoredChunk(BaseModel):
    """A row of the chunks table as a search returns it."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    chunk_id: str
    path: str
    language: Language
    kind: ChunkKind
    name: str
    parent: str | None
    line_start: int
    line_end: int
    content: str

    def candidate(self, similarity: float) -> Candidate:
        """This row as a chunk a search found, with its similarity to the query."""
        chunk = Chunk(
            chunk_id=self.chunk_id,
            path=Path(self.path),
            language=self.language,
            kind=self.kind,
            name=self.name,
            parent=self.parent,
            line_start=self.line_start,
            line_end=self.line_end,
            content=self.content,
        )
        return Candidate(chunk=chunk, similarity=similarity)


# What a search reads of each candidate: the columns a stored chunk is made of, which its
# fields name, and its vector; not its search text
CANDIDATE_COLUMNS = (*StoredChunk.model_fields, VECTOR_COLUMN)


class LeftIndex(BaseModel):
    """An index of the project this store does not use: the paths it is made of, and the
    lock its writers hold, which 0.4 had none of."""

    model_config = ConfigDict(frozen=True)

    parts: list[Path]
    write_lock: Path | None


class StoredFile(BaseModel):
    """A row of the files table."""

    model_config = ConfigDict(frozen=True)

    path: str
    mtime_ns: int
    size_bytes: int

    def source(self) -> SourceFile:
        """This row as the record of a file the index holds."""
        return SourceFile(path=Path(self.path), mtime_ns=self.mtime_ns, size_bytes=self.size_bytes)


class LanceDBIndexStore(IndexStore):
    """One project's index of one identity, in a LanceDB database directory of its own
    under the project's index directory.

    Nothing touches the disk until it is first used, and only a write makes the
    directory; the connection is opened once, by whichever worker thread needs it first.
    """

    def __init__(
        self, project_directory: Path, identity: IndexIdentity, *, lock_timeout_seconds: float
    ) -> None:
        self._project_directory = project_directory
        self._identity = identity
        self._directory = project_directory / index_directory_name(identity)
        self._lock_timeout_seconds = lock_timeout_seconds
        self._connecting = threading.Lock()
        self._connection: lancedb.DBConnection | None = None
        # How many upkeeps this store runs before it rebuilds the full-text index; none
        # before its first, as it does not know how many parts an earlier process left
        self._upkeeps_until_rebuild = 0

    @property
    def directory(self) -> Path:
        """Where this index lives."""
        return self._directory

    @property
    def write_lock(self) -> Path:
        """The file a writer of this index locks, beside its directory."""
        return write_lock_of(self._directory)

    @override
    def manifest(self) -> IndexManifest | None:
        path = self._directory / MANIFEST_FILE
        if not path.is_file():
            return None
        try:
            return IndexManifest.model_validate_json(path.read_bytes())
        except ValidationError as error:
            # Read as an index without a manifest, which its next write writes again
            log.warning("manifest_unreadable", directory=str(self._directory), error=str(error))
            return None

    @override
    def indexed_files(self) -> list[SourceFile]:
        table = self._open(FILES_TABLE)
        if table is None:
            return []
        return [StoredFile.model_validate(row).source() for row in table.to_arrow().to_pylist()]

    @override
    def counts(self) -> IndexCounts:
        return IndexCounts(files=self._count(FILES_TABLE), chunks=self._count(CHUNKS_TABLE))

    @override
    def chunked_paths(self) -> set[Path]:
        table = self._open(CHUNKS_TABLE)
        if table is None:
            return set()
        # One column of every row, no vectors read; each path made a Path once, not per row
        rows = table.search().select([PATH_COLUMN]).limit(None).to_arrow()
        paths = rows.column(PATH_COLUMN).unique()
        return {Path(path) for path in paths.to_pylist()}

    @override
    def stored_vectors(self, chunk_ids: list[str]) -> dict[str, Vector]:
        table = self._open(CHUNKS_TABLE)
        if table is None or not chunk_ids:
            return {}
        quoted = ", ".join(quote_sql(chunk_id) for chunk_id in chunk_ids)
        rows = (
            table.search()
            .where(f"{CHUNK_ID_COLUMN} IN ({quoted})")
            .select([CHUNK_ID_COLUMN, VECTOR_COLUMN])
            .limit(None)
            .to_arrow()
        )
        matrix = vector_matrix(rows)
        return dict(zip(rows.column(CHUNK_ID_COLUMN).to_pylist(), matrix, strict=True))

    @override
    def keep_up(self) -> None:
        with FileLock(self.write_lock, timeout=self._lock_timeout_seconds):
            for name in (CHUNKS_TABLE, FILES_TABLE):
                table = self._open(name)
                if table is not None:
                    table.optimize(cleanup_older_than=OLD_VERSION_RETENTION)
                    log.debug("table_kept_up", table=name)
            self._rebuild_full_text_index_when_due()

    @override
    def write(self, update: IndexUpdate, vectors: VectorMatrix) -> None:
        if len(update.chunks) != len(vectors):
            message = f"{len(update.chunks)} chunks but {len(vectors)} vectors"
            raise ValueError(message)
        self._note_use()
        self._project_directory.mkdir(parents=True, exist_ok=True)
        with FileLock(self.write_lock, timeout=self._lock_timeout_seconds):
            database = self._created_database()
            if update.chunks:
                self._fit_layout(database, vectors.shape[VECTOR_LENGTH_AXIS])
            self._write_chunks(database, update, vectors)
            self._write_records(database, update)
            manifest = IndexManifest(
                identity=self._identity,
                schema_version=SCHEMA_VERSION,
                indexed_at=datetime.now(UTC),
            )
            write_atomically(self._directory / MANIFEST_FILE, manifest.model_dump_json())
        log.debug(
            "index_written",
            files=len(update.files),
            chunks=len(update.chunks),
            removed=len(update.removed_paths),
        )

    @override
    def search_candidates(
        self, query_vector: Vector, query: SearchQuery, per_channel: int
    ) -> SearchCandidates:
        table = self._open(CHUNKS_TABLE)
        if table is None:
            return NO_CANDIDATES
        # An open table reads the version it was opened at, so both channels see one index
        scope = scope_filter(query.scope)
        # lancedb's stubs type search() as the base builder, which lacks distance_type()
        by_vector = table.search(query_vector).distance_type(COSINE_METRIC)  # ty: ignore[unresolved-attribute]
        if scope is not None:
            # Filtered before the nearest are taken, so a narrow scope still fills them
            by_vector = by_vector.where(scope, prefilter=True)
        nearest = by_vector.select([*CANDIDATE_COLUMNS, DISTANCE_COLUMN]).limit(per_channel)
        return SearchCandidates(
            by_vector=candidates_of(nearest.to_arrow(), query_vector),
            by_keyword=keyword_candidates(table, query.text, scope, query_vector, per_channel),
        )

    def _fit_layout(self, database: lancedb.DBConnection, width: int) -> None:
        """Start the index again when its chunks table does not hold chunks of this width,
        as a model replaced under a name nothing pins gives. The records go first, then the table:
        a write stopped between the two never leaves records of chunks that are gone."""
        stored_chunks = self._open(CHUNKS_TABLE)
        if stored_chunks is None or stored_chunks.schema == chunks_schema(width):
            return
        log.warning("index_layout_replaced", directory=str(self._directory), width=width)
        for name in (FILES_TABLE, CHUNKS_TABLE):
            if self._open(name) is not None:
                database.drop_table(name)

    def _write_chunks(
        self, database: lancedb.DBConnection, update: IndexUpdate, vectors: VectorMatrix
    ) -> None:
        """Put the chunks of the files written in place of what the index held of them and
        of the files removed, in one commit; a chunk whose id is stored already is updated
        where it stands. Makes the table, and its full-text index, if there is none."""
        table = self._open(CHUNKS_TABLE)
        if table is None:
            if not update.chunks:
                return
            table = database.create_table(CHUNKS_TABLE, data=chunks_table(update.chunks, vectors))
        else:
            replaced = [*(file.path for file in update.files), *update.removed_paths]
            if not replaced:
                return
            rows = (
                chunks_table(update.chunks, vectors)
                if update.chunks
                else table.schema.empty_table()
            )
            (
                table.merge_insert(CHUNK_ID_COLUMN)
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .when_not_matched_by_source_delete(path_filter(PATH_COLUMN, replaced))
                .execute(rows)
            )
        if not has_full_text_index(table):
            make_full_text_index(table)

    def _write_records(self, database: lancedb.DBConnection, update: IndexUpdate) -> None:
        """Record the files written and drop the records of the files removed, after their
        chunks: a write stopped between the two leaves the old records, which the next run
        reads as files changed or gone, and writes again."""
        table = self._open(FILES_TABLE)
        if table is None:
            if update.files:
                database.create_table(FILES_TABLE, data=files_table(update.files))
            return
        if update.removed_paths:
            table.delete(path_filter(PATH_COLUMN, update.removed_paths))
        if update.files:
            (
                table.merge_insert(PATH_COLUMN)
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(files_table(update.files))
            )

    def _rebuild_full_text_index_when_due(self) -> None:
        """Rebuild the full-text index whole, which merges the parts upkeeps added, at this
        store's first upkeep and once in UPKEEPS_PER_FULL_TEXT_REBUILD after it."""
        if self._upkeeps_until_rebuild > 0:
            self._upkeeps_until_rebuild -= 1
            return
        chunks = self._open(CHUNKS_TABLE)
        if chunks is None:
            return
        make_full_text_index(chunks)
        self._upkeeps_until_rebuild = UPKEEPS_PER_FULL_TEXT_REBUILD - 1
        log.debug("full_text_index_rebuilt")

    def _note_use(self) -> None:
        """Mark this index as used, so another store leaves it, and remove the project's
        indexes nobody used for UNUSED_INDEX_RETENTION, once its stamp is USE_MARK_INTERVAL
        old. The stamp's own mtime keeps that time, on the clock the remover reads, so a
        session whose machine slept for a week marks its index at its first read; and every
        process of one index shares the one stamp. An index not made yet has nothing to
        mark. Neither can fail a read: what cannot be done now is left for the next time."""
        stamp = self._directory / USE_STAMP_FILE
        if not self._directory.is_dir():
            return
        if seconds_since_change(stamp) < USE_MARK_INTERVAL.total_seconds():
            return
        try:
            stamp.touch()
        except OSError as error:
            log.debug("index_use_not_marked", directory=str(self._directory), error=str(error))
        remove_unused_indexes(self._project_directory, self._directory)

    def _count(self, name: str) -> int:
        table = self._open(name)
        return 0 if table is None else table.count_rows()

    def _open(self, name: str) -> lancedb.table.Table | None:
        """A table at its latest version; None when it is not made yet."""
        database = self._existing_database()
        if database is None or name not in database.list_tables().tables:
            return None
        return database.open_table(name)

    def _existing_database(self) -> lancedb.DBConnection | None:
        """The connection, for a reader; None while the index directory is not made yet."""
        self._note_use()
        if not self._directory.is_dir():
            return None
        return self._connect()

    def _created_database(self) -> lancedb.DBConnection:
        """The connection, for a writer, making the directory and the project's
        `.gitignore` first. Called under the write lock: a removal renames an index only
        under that lock, so it cannot take the directory between its making and the write."""
        self._directory.mkdir(parents=True, exist_ok=True)
        ignore_file = self._project_directory / GITIGNORE_FILE
        if not ignore_file.exists():
            ignore_file.write_text(IGNORE_EVERYTHING)
        return self._connect()

    def _connect(self) -> lancedb.DBConnection:
        """The connection, opened once, by whichever thread needs it first."""
        with self._connecting:
            if self._connection is None:
                self._connection = lancedb.connect(str(self._directory))
            return self._connection


def index_directory_name(identity: IndexIdentity) -> str:
    """The name of the directory an index of this identity and layout lives in."""
    digest = hashlib.sha256(identity.model_dump_json().encode()).hexdigest()
    return INDEX_DIRECTORY_FORMAT.format(
        schema_version=SCHEMA_VERSION, identity_digest=digest[:IDENTITY_DIGEST_LENGTH]
    )


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


def chunks_table(chunks: list[Chunk], vectors: VectorMatrix) -> pyarrow.Table:
    """Chunks as rows of the chunks schema, each with the vector of its row in the matrix.

    The matrix goes into Arrow as one flat buffer, without a Python float per component.
    """
    width = vectors.shape[VECTOR_LENGTH_AXIS]
    vector_column = pyarrow.FixedSizeListArray.from_arrays(pyarrow.array(vectors.ravel()), width)
    return pyarrow.table(
        {
            CHUNK_ID_COLUMN: [chunk.chunk_id for chunk in chunks],
            VECTOR_COLUMN: vector_column,
            PATH_COLUMN: [chunk.path.as_posix() for chunk in chunks],
            LANGUAGE_COLUMN: [chunk.language for chunk in chunks],
            KIND_COLUMN: [chunk.kind for chunk in chunks],
            NAME_COLUMN: [chunk.name for chunk in chunks],
            PARENT_COLUMN: [chunk.parent for chunk in chunks],
            LINE_START_COLUMN: [chunk.line_start for chunk in chunks],
            LINE_END_COLUMN: [chunk.line_end for chunk in chunks],
            CONTENT_COLUMN: [chunk.content for chunk in chunks],
            SEARCH_TEXT_COLUMN: [search_text(chunk) for chunk in chunks],
        },
        schema=chunks_schema(width),
    )


def files_table(files: list[SourceFile]) -> pyarrow.Table:
    """The records of indexed files as rows of the files schema."""
    return pyarrow.table(
        {
            PATH_COLUMN: [file.path.as_posix() for file in files],
            MTIME_COLUMN: [file.mtime_ns for file in files],
            SIZE_COLUMN: [file.size_bytes for file in files],
        },
        schema=FILES_SCHEMA,
    )


def make_full_text_index(table: lancedb.table.Table) -> None:
    """Index the chunks' search text for full-text search, whole, in place of any index the
    table has."""
    table.create_fts_index(SEARCH_TEXT_COLUMN, replace=True, remove_stop_words=REMOVE_STOP_WORDS)


def has_full_text_index(table: lancedb.table.Table) -> bool:
    return any(index.index_type == FULL_TEXT_INDEX_TYPE for index in table.list_indices())


def path_filter(column: str, paths: list[Path]) -> str:
    """A filter matching the rows whose column holds one of the paths."""
    quoted = ", ".join(quote_sql(path.as_posix()) for path in paths)
    return f"{column} IN ({quoted})"


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


def keyword_candidates(
    table: lancedb.table.Table,
    query_text: str,
    scope: str | None,
    query_vector: Vector,
    limit: int,
) -> list[Candidate]:
    """The chunks of the scope whose words best match the query's, best first; none while
    the table has no full-text index, as after a write stopped between the table and its
    index."""
    if not has_full_text_index(table):
        return []
    by_keyword = table.search(query_text, query_type=FULL_TEXT_QUERY_TYPE)
    if scope is not None:
        by_keyword = by_keyword.where(scope)
    rows = by_keyword.select([*CANDIDATE_COLUMNS, FULL_TEXT_SCORE_COLUMN]).limit(limit)
    return candidates_of(rows.to_arrow(), query_vector)


def scope_filter(scope: SearchScope) -> str | None:
    """The filter that keeps a search's channels to its scope; None for a scope that
    leaves nothing out."""
    conditions = []
    if scope.kinds:
        conditions.append(values_filter(KIND_COLUMN, scope.kinds))
    if scope.languages:
        conditions.append(values_filter(LANGUAGE_COLUMN, scope.languages))
    if scope.path_glob is not None:
        conditions.append(glob_filter(scope.path_glob, scope.directory))
    if scope.exclude_glob is not None:
        conditions.append(SQL_NOT + glob_filter(scope.exclude_glob, scope.directory))
    if scope.directory is not None:
        conditions.append(directory_filter(scope.directory))
    return SQL_AND.join(conditions) or None


def values_filter(column: str, values: Sequence[str]) -> str:
    """A filter matching the rows whose column holds one of the values."""
    quoted = ", ".join(quote_sql(value) for value in values)
    return f"{column} IN ({quoted})"


def glob_filter(glob: str, base: Path | None) -> str:
    """A filter matching the rows whose path the glob names, read from `base`, the
    directory a search was asked from, when there is one."""
    pattern = glob_pattern(glob, base)
    return f"{REGEX_MATCH_FUNCTION}({PATH_COLUMN}, {quote_sql(pattern)})"


def directory_filter(directory: Path) -> str:
    """A filter matching the rows whose path lies in the directory: a prefix compared as
    it is written, which costs less than a regex (29 against 40 ms at 100k rows, a
    review's probe)."""
    prefix = directory.as_posix() + PATH_SEPARATOR
    return f"{PREFIX_MATCH_FUNCTION}({PATH_COLUMN}, {quote_sql(prefix)})"


def candidates_of(rows: pyarrow.Table, query_vector: Vector) -> list[Candidate]:
    """Rows a channel found, in its order, each with its vector's similarity to the query."""
    similarities = cosine_similarities(vector_matrix(rows), query_vector)
    chunks = rows.drop_columns([VECTOR_COLUMN]).to_pylist()
    return [
        StoredChunk.model_validate(row).candidate(float(similarity))
        for row, similarity in zip(chunks, similarities, strict=True)
    ]


def vector_matrix(rows: pyarrow.Table) -> VectorMatrix:
    """The vectors of rows as a matrix, a row each, read as one flat buffer."""
    vectors = rows.column(VECTOR_COLUMN).combine_chunks()
    return vectors.flatten().to_numpy().reshape(rows.num_rows, vectors.type.list_size)


def cosine_similarities(vectors: VectorMatrix, query: Vector) -> NDArray[numpy.float32]:
    """The cosine similarity of each row's vector to the query's: 0 where either is all
    zeros, which points nowhere, and held to its range, which rounding oversteps by a hair."""
    norms = numpy.linalg.norm(vectors, axis=VECTOR_LENGTH_AXIS) * numpy.linalg.norm(query)
    products = vectors @ query
    similarities = numpy.divide(
        products, norms, out=numpy.zeros_like(products), where=norms > NO_LENGTH
    )
    return numpy.clip(similarities, LOWEST_SIMILARITY, HIGHEST_SIMILARITY)


def quote_sql(value: str) -> str:
    """A string as a quoted SQL literal."""
    return SQL_QUOTE + value.replace(SQL_QUOTE, SQL_ESCAPED_QUOTE) + SQL_QUOTE
