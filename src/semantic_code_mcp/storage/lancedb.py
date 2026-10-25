"""LanceDB storage of one project's index: the chunks with their vectors, the record of
each file they came from, and the manifest of what built them.

A project's index directory holds an index for each identity — the model, the chunker
version — and layout, each in a directory of its own named after them. So a session of
another model or version, or a server upgraded or rolled back, uses an index of its own
and never builds this one again; switching back finds the old one still there. An index
nobody wrote for UNUSED_INDEX_RETENTION is removed by the next write of another, and so
is what 0.4 kept directly in the project's directory.

A write is one call, under a file lock in the index's directory, so two server processes
on one index never write side by side. A write of changed files replaces those files'
chunks, then their records; a whole-index write removes every record, then every chunk,
then writes both again. The manifest comes last either way. The chunks table is made by
the first write that has vectors, as wide as they are; vectors of another width (a model
swapped under the same name) drop the records, then the table, and the index starts
again from the write, its other files unrecorded so the next run indexes them.

Reads take no lock: each opens its table again, at its latest version, so a long-lived
connection still sees what another process wrote, and a table not yet made reads as
empty. So old versions are pruned only once they are older than any
read can last; the upkeep that compacts the tables, folds new rows into the full-text
index and prunes runs after a write once a table has gathered enough versions, not on
every write, where it would cost more than the write itself; after a whole index is
written it runs at once, so its rows are not searched unindexed until then.

The index directory holds a `.gitignore` of `*`, so a project-local index never shows
in the project's own repository.
"""

import hashlib
import os
import shutil
import tempfile
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import override

import lancedb
import pyarrow
import structlog
from filelock import FileLock
from pydantic import BaseModel, ConfigDict, ValidationError

from semantic_code_mcp.domain.interfaces import IndexStore, Vector, VectorMatrix
from semantic_code_mcp.domain.models import (
    HIGHEST_SCORE,
    LOWEST_SCORE,
    Chunk,
    ChunkLocation,
    ChunkType,
    IndexCounts,
    IndexIdentity,
    IndexManifest,
    IndexUpdate,
    SearchResult,
    SourceFile,
    WriteScope,
)

log = structlog.get_logger()

# Bumped by any change to the tables' columns; each version keeps an index of its own
SCHEMA_VERSION = 1

# An index's directory, under the project's, named by its layout and its identity's digest
INDEX_DIRECTORY_FORMAT = "schema-{schema_version}-{identity_digest}"
INDEX_DIRECTORY_PATTERN = "schema-*"
IDENTITY_DIGEST_LENGTH = 16
# An index nobody wrote for this long is removed by the next write of another index of the
# project: no session indexed with its model or version for a week, and a rebuild costs
# minutes where keeping it costs the disk it fills
UNUSED_INDEX_RETENTION = timedelta(days=7)
# What 0.4 kept directly in a project's index directory: its chunks table and its record
# of mtimes, removed together
LEGACY_ENTRIES = ("chunks.lance", "file_mtimes.json")

CHUNKS_TABLE = "chunks"
FILES_TABLE = "files"
MANIFEST_FILE = "manifest.json"
WRITE_LOCK_FILE = "write.lock"
GITIGNORE_FILE = ".gitignore"
IGNORE_EVERYTHING = "*\n"
# Written beside its target, then renamed over it, so a crash never leaves half a file
TEMPORARY_SUFFIX = ".tmp"

# Upkeep runs once a table has more new versions than this: LanceDB's own rule of thumb
# is to optimize after about twenty modifications
VERSIONS_BEFORE_UPKEEP = 20
# A table not kept up since the process started counts its versions from the first
NEVER_KEPT_UP = 0
# Versions older than this are pruned by upkeep; a read finishes in milliseconds
OLD_VERSION_RETENTION = timedelta(seconds=10)

VECTOR_COLUMN = "vector"
FILE_PATH_COLUMN = "file_path"
LINE_START_COLUMN = "line_start"
LINE_END_COLUMN = "line_end"
CONTENT_COLUMN = "content"
CHUNK_TYPE_COLUMN = "chunk_type"
NAME_COLUMN = "name"
PATH_COLUMN = "path"
MTIME_COLUMN = "mtime_ns"
SIZE_COLUMN = "size_bytes"
# The axis of a vector matrix along one vector: rows are chunks, columns components
VECTOR_LENGTH_AXIS = 1
# A filter every row matches
MATCH_EVERY_ROW = "true"
FULL_TEXT_QUERY_TYPE = "fts"
FULL_TEXT_INDEX_TYPE = "FTS"
COSINE_METRIC = "cosine"
# Columns LanceDB adds to a hit: the cosine distance of a vector hit, the score of a text hit
DISTANCE_COLUMN = "_distance"
FULL_TEXT_SCORE_COLUMN = "_score"

# Score normalization constants
COSINE_DISTANCE_MAX = 2.0  # cosine distance range is [0, 2] by definition
FULL_TEXT_SCORE_DIVISOR = 10.0  # empirical: the full-text scores seen peak around 5-15
FULL_TEXT_DEFAULT_SCORE = 0.5  # fallback when a text hit carries no score
# How a single quote is written inside a quoted SQL string
SQL_QUOTE = "'"
SQL_ESCAPED_QUOTE = "''"

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
            pyarrow.field(VECTOR_COLUMN, pyarrow.list_(pyarrow.float32(), dimension)),
            pyarrow.field(FILE_PATH_COLUMN, pyarrow.utf8()),
            pyarrow.field(LINE_START_COLUMN, pyarrow.int32()),
            pyarrow.field(LINE_END_COLUMN, pyarrow.int32()),
            pyarrow.field(CONTENT_COLUMN, pyarrow.utf8()),
            pyarrow.field(CHUNK_TYPE_COLUMN, pyarrow.utf8()),
            pyarrow.field(NAME_COLUMN, pyarrow.utf8()),
        ]
    )


class StoredChunk(BaseModel):
    """A row of the chunks table as a search returns it."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    file_path: str
    line_start: int
    line_end: int
    content: str
    chunk_type: ChunkType
    name: str

    def scored(self, score: float) -> SearchResult:
        """This row as a search hit with a score."""
        return SearchResult(
            file_path=self.file_path,
            line_start=self.line_start,
            line_end=self.line_end,
            content=self.content,
            chunk_type=self.chunk_type,
            name=self.name,
            score=score,
        )


class LeftIndex(BaseModel):
    """An index of the project this store does not use, by the paths it is made of."""

    model_config = ConfigDict(frozen=True)

    parts: list[Path]


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
        # The version each table had after its last upkeep; only a write changes it
        self._kept_up_at: dict[str, int] = {}

    @property
    def directory(self) -> Path:
        """Where this index lives."""
        return self._directory

    @override
    def manifest(self) -> IndexManifest | None:
        path = self._directory / MANIFEST_FILE
        if not path.is_file():
            return None
        try:
            return IndexManifest.model_validate_json(path.read_bytes())
        except ValidationError as error:
            # Read as an index without a manifest, which the next run rebuilds
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
    def write(self, update: IndexUpdate, vectors: VectorMatrix) -> None:
        if len(update.chunks) != len(vectors):
            message = f"{len(update.chunks)} chunks but {len(vectors)} vectors"
            raise ValueError(message)
        database = self._created_database()
        with FileLock(self._directory / WRITE_LOCK_FILE, timeout=self._lock_timeout_seconds):
            if update.chunks:
                self._fit_layout(database, vectors.shape[VECTOR_LENGTH_AXIS])
            match update.scope:
                case WriteScope.everything:
                    self._empty()
                case WriteScope.changed_files:
                    self._remove_files(update)
            self._add_chunks(database, update.chunks, vectors)
            self._record_files(database, update.files)
            manifest = IndexManifest(
                identity=self._identity,
                schema_version=SCHEMA_VERSION,
                indexed_at=datetime.now(UTC),
            )
            write_atomically(self._directory / MANIFEST_FILE, manifest.model_dump_json())
            self._keep_up(every_table=update.scope == WriteScope.everything)
        self._remove_unused_indexes()
        log.debug(
            "index_written",
            scope=update.scope,
            files=len(update.files),
            chunks=len(update.chunks),
            removed=len(update.removed_paths),
        )

    def search_vector(self, query_vector: Vector, limit: int) -> list[SearchResult]:
        """The chunks nearest to a vector, scored by cosine similarity mapped to [0, 1]."""
        table = self._open(CHUNKS_TABLE)
        if table is None:
            return []
        query = table.search(query_vector)
        # lancedb's stubs type search() as the base builder, which lacks metric()
        rows = query.metric(COSINE_METRIC).limit(limit).to_list()  # ty: ignore[unresolved-attribute]
        return [
            StoredChunk.model_validate(row).scored(
                clamp_score(HIGHEST_SCORE - row[DISTANCE_COLUMN] / COSINE_DISTANCE_MAX)
            )
            for row in rows
        ]

    def search_full_text(self, query_text: str, limit: int) -> list[SearchResult]:
        """The chunks whose text best matches the words of a query."""
        table = self._open(CHUNKS_TABLE)
        if table is None or not has_full_text_index(table):
            return []
        rows = table.search(query_text, query_type=FULL_TEXT_QUERY_TYPE).limit(limit).to_list()
        return [
            StoredChunk.model_validate(row).scored(
                clamp_score(
                    row.get(FULL_TEXT_SCORE_COLUMN, FULL_TEXT_DEFAULT_SCORE)
                    / FULL_TEXT_SCORE_DIVISOR
                )
            )
            for row in rows
        ]

    @override
    def search_hybrid(
        self,
        query_vector: Vector,
        query_text: str,
        limit: int,
        vector_weight: float,
    ) -> list[SearchResult]:
        """Vector and full-text hits, weighted and summed per chunk, best first."""
        merged = merge_weighted(
            self.search_vector(query_vector, limit),
            vector_weight,
            self.search_full_text(query_text, limit),
            HIGHEST_SCORE - vector_weight,
        )
        return sorted(merged, key=lambda result: result.score, reverse=True)[:limit]

    def _fit_layout(self, database: lancedb.DBConnection, width: int) -> None:
        """Start the index again when its chunks table does not hold chunks of this width,
        as a model swapped under the same name gives. The records go first, then the table:
        a write stopped between the two never leaves records of chunks that are gone."""
        stored_chunks = self._open(CHUNKS_TABLE)
        if stored_chunks is None or stored_chunks.schema == chunks_schema(width):
            return
        log.warning("index_layout_replaced", directory=str(self._directory), width=width)
        for name in (FILES_TABLE, CHUNKS_TABLE):
            if self._open(name) is not None:
                database.drop_table(name)
                self._kept_up_at.pop(name, None)

    def _empty(self) -> None:
        """Remove every record and chunk before a whole index is written again.

        The records go first: a write that stops between the two leaves chunks without
        records, which the next run reads as incomplete and builds again, never records of
        chunks that are gone. Rows are deleted and the tables kept, so a reader finds them
        throughout.
        """
        for name in (FILES_TABLE, CHUNKS_TABLE):
            table = self._open(name)
            if table is not None:
                table.delete(MATCH_EVERY_ROW)

    def _remove_files(self, update: IndexUpdate) -> None:
        """Remove the chunks of every file written or removed, and the records of the removed."""
        written_or_removed = [*(file.path for file in update.files), *update.removed_paths]
        stored_chunks = self._open(CHUNKS_TABLE)
        if stored_chunks is not None and written_or_removed:
            stored_chunks.delete(path_filter(FILE_PATH_COLUMN, written_or_removed))
        stored_files = self._open(FILES_TABLE)
        if stored_files is not None and update.removed_paths:
            stored_files.delete(path_filter(PATH_COLUMN, update.removed_paths))

    def _add_chunks(
        self, database: lancedb.DBConnection, chunks: list[Chunk], vectors: VectorMatrix
    ) -> None:
        """Add chunks, making the table as wide as their vectors if there is none, and its
        full-text index if it has none; upkeep keeps that index current afterwards."""
        if not chunks:
            return
        rows = chunks_table(chunks, vectors)
        table = self._open(CHUNKS_TABLE)
        if table is None:
            table = database.create_table(CHUNKS_TABLE, data=rows)
        else:
            table.add(rows)
        if not has_full_text_index(table):
            table.create_fts_index(CONTENT_COLUMN)

    def _record_files(self, database: lancedb.DBConnection, files: list[SourceFile]) -> None:
        """Record the files written, after their chunks: a crash between the two leaves a
        file unrecorded, and so indexed again by the next run."""
        if not files:
            return
        rows = files_table(files)
        table = self._open(FILES_TABLE)
        if table is None:
            database.create_table(FILES_TABLE, data=rows)
            return
        table.merge_insert(
            PATH_COLUMN
        ).when_matched_update_all().when_not_matched_insert_all().execute(rows)

    def _keep_up(self, *, every_table: bool) -> None:
        """Compact a table, fold its new rows into its indexes and prune its old versions:
        every table after a whole index is written, else a table that has gathered more
        than VERSIONS_BEFORE_UPKEEP versions since its last upkeep in this process."""
        for name in (CHUNKS_TABLE, FILES_TABLE):
            table = self._open(name)
            if table is None:
                continue
            versions_since = table.version - self._kept_up_at.get(name, NEVER_KEPT_UP)
            if every_table or versions_since > VERSIONS_BEFORE_UPKEEP:
                table.optimize(cleanup_older_than=OLD_VERSION_RETENTION)
                self._kept_up_at[name] = table.version
                log.debug("table_kept_up", table=name)

    def _remove_unused_indexes(self) -> None:
        """Remove each other index of the project that nothing wrote for
        UNUSED_INDEX_RETENTION. One that vanishes or cannot be read meanwhile is left for
        the next write. A session that writes such an index at the very moment it goes
        fails that write once; its next run builds the index again."""
        unused_before = time.time() - UNUSED_INDEX_RETENTION.total_seconds()
        for left in left_indexes(self._project_directory, self._directory):
            try:
                if last_change(left) < unused_before:
                    remove_index(left)
                    log.info("unused_index_removed", parts=[str(part) for part in left.parts])
            except OSError as error:
                log.warning("unused_index_not_removed", error=str(error))

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
        if not self._directory.is_dir():
            return None
        return self._connect()

    def _created_database(self) -> lancedb.DBConnection:
        """The connection, for a writer, making the directory and its `.gitignore` first."""
        self._directory.mkdir(parents=True, exist_ok=True)
        ignore_file = self._directory / GITIGNORE_FILE
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


def left_indexes(project_directory: Path, own_directory: Path) -> list[LeftIndex]:
    """The project's indexes other than this store's: one per identity directory, and the
    entries 0.4 kept in the project's directory, as one."""
    others = [
        LeftIndex(parts=[directory])
        for directory in sorted(project_directory.glob(INDEX_DIRECTORY_PATTERN))
        if directory != own_directory
    ]
    legacy = [
        project_directory / name for name in LEGACY_ENTRIES if (project_directory / name).exists()
    ]
    return [*others, LeftIndex(parts=legacy)] if legacy else others


def last_change(left: LeftIndex) -> float:
    """When an index was last written, in seconds since the epoch: the latest change to any
    of its parts or to what lies directly inside one, where each write leaves a new file."""
    entries = [
        *left.parts,
        *(child for part in left.parts if part.is_dir() for child in part.iterdir()),
    ]
    return max(entry.lstat().st_mtime for entry in entries)


def remove_index(left: LeftIndex) -> None:
    for part in left.parts:
        if part.is_dir():
            shutil.rmtree(part)
        else:
            part.unlink(missing_ok=True)


def chunks_table(chunks: list[Chunk], vectors: VectorMatrix) -> pyarrow.Table:
    """Chunks as rows of the chunks schema, each with the vector of its row in the matrix.

    The matrix goes into Arrow as one flat buffer, without a Python float per component.
    """
    width = vectors.shape[VECTOR_LENGTH_AXIS]
    vector_column = pyarrow.FixedSizeListArray.from_arrays(pyarrow.array(vectors.ravel()), width)
    return pyarrow.table(
        {
            VECTOR_COLUMN: vector_column,
            FILE_PATH_COLUMN: [chunk.file_path for chunk in chunks],
            LINE_START_COLUMN: [chunk.line_start for chunk in chunks],
            LINE_END_COLUMN: [chunk.line_end for chunk in chunks],
            CONTENT_COLUMN: [chunk.content for chunk in chunks],
            CHUNK_TYPE_COLUMN: [chunk.chunk_type for chunk in chunks],
            NAME_COLUMN: [chunk.name for chunk in chunks],
        },
        schema=chunks_schema(width),
    )


def files_table(files: list[SourceFile]) -> pyarrow.Table:
    """The records of indexed files as rows of the files schema."""
    return pyarrow.table(
        {
            PATH_COLUMN: [str(file.path) for file in files],
            MTIME_COLUMN: [file.mtime_ns for file in files],
            SIZE_COLUMN: [file.size_bytes for file in files],
        },
        schema=FILES_SCHEMA,
    )


def has_full_text_index(table: lancedb.table.Table) -> bool:
    return any(index.index_type == FULL_TEXT_INDEX_TYPE for index in table.list_indices())


def path_filter(column: str, paths: list[Path]) -> str:
    """A filter matching the rows whose column holds one of the paths."""
    quoted = ", ".join(quote_sql(str(path)) for path in paths)
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


def merge_weighted(
    primary: list[SearchResult],
    primary_weight: float,
    secondary: list[SearchResult],
    secondary_weight: float,
) -> list[SearchResult]:
    """Merge two ranked lists: each hit's score times its list's weight, summed per chunk."""
    merged: dict[ChunkLocation, SearchResult] = {
        result.location: result.with_score(result.score * primary_weight) for result in primary
    }
    for result in secondary:
        weighted = result.score * secondary_weight
        earlier = merged.get(result.location)
        if earlier is None:
            merged[result.location] = result.with_score(weighted)
        else:
            merged[result.location] = earlier.with_score(clamp_score(earlier.score + weighted))
    return list(merged.values())


def clamp_score(score: float) -> float:
    """A score held to [0, 1]."""
    return max(LOWEST_SCORE, min(HIGHEST_SCORE, score))


def quote_sql(value: str) -> str:
    """A string as a quoted SQL literal."""
    return SQL_QUOTE + value.replace(SQL_QUOTE, SQL_ESCAPED_QUOTE) + SQL_QUOTE
