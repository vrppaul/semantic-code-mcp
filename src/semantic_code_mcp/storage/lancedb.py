"""LanceDB storage of one project's index: the chunks with their vectors, the record of
each file they came from, and the manifest of what built them.

A write is one call: the chunks of the changed and removed files are replaced, then
their file records, then the manifest, all under a file lock in the index directory,
so two server processes on one project never write side by side. The chunks table is
made by the first write that has vectors, as wide as they are, and made again when a
rebuild brings vectors of another width or the table predates this schema.

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

import os
import tempfile
import threading
from datetime import timedelta
from pathlib import Path
from typing import override

import lancedb
import pyarrow
import structlog
from filelock import FileLock
from pydantic import BaseModel, ConfigDict, ValidationError

from semantic_code_mcp.domain.errors import IndexWriteError
from semantic_code_mcp.domain.interfaces import IndexStore, Vector, VectorMatrix
from semantic_code_mcp.domain.models import (
    HIGHEST_SCORE,
    LOWEST_SCORE,
    Chunk,
    ChunkLocation,
    ChunkType,
    IndexCounts,
    IndexManifest,
    IndexUpdate,
    SearchResult,
    SourceFile,
    WriteScope,
)

log = structlog.get_logger()

# Bumped by any change to the tables' columns; an index of another version is rebuilt
SCHEMA_VERSION = 1

CHUNKS_TABLE = "chunks"
FILES_TABLE = "files"
MANIFEST_FILE = "manifest.json"
WRITE_LOCK_FILE = "write.lock"
GITIGNORE_FILE = ".gitignore"
IGNORE_EVERYTHING = "*\n"
# What 0.4 kept beside its table, removed when its index is rebuilt
LEGACY_FILES = ("file_mtimes.json",)
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
    """One project's index in a LanceDB database directory.

    Nothing touches the disk until it is first used, and only a write makes the
    directory; the connection is opened once, by whichever worker thread needs it first.
    """

    def __init__(self, directory: Path, *, lock_timeout_seconds: float) -> None:
        self._directory = directory
        self._lock_timeout_seconds = lock_timeout_seconds
        self._connecting = threading.Lock()
        self._connection: lancedb.DBConnection | None = None
        # The version each table had after its last upkeep; only a write changes it
        self._kept_up_at: dict[str, int] = {}

    @property
    @override
    def schema_version(self) -> int:
        return SCHEMA_VERSION

    @override
    def manifest(self) -> IndexManifest | None:
        try:
            text = (self._directory / MANIFEST_FILE).read_text()
        except FileNotFoundError:
            return None
        try:
            return IndexManifest.model_validate_json(text)
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
            match update.scope:
                case WriteScope.everything:
                    self._empty(database, update, vectors)
                case WriteScope.changed_files:
                    self._remove_files(update)
            self._add_chunks(database, update.chunks, vectors)
            self._record_files(database, update.files)
            write_atomically(self._directory / MANIFEST_FILE, update.manifest.model_dump_json())
            self._keep_up(every_table=update.scope == WriteScope.everything)
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

    def _empty(
        self, database: lancedb.DBConnection, update: IndexUpdate, vectors: VectorMatrix
    ) -> None:
        """Remove every chunk and record before a whole index is written again.

        Rows are deleted and the tables kept, so a reader finds them throughout. A chunks
        table is dropped instead, to be made again, when it is not of the width and layout
        of the chunks written, or when no chunk is written to tell what those would be.
        """
        chunks = self._open(CHUNKS_TABLE)
        if chunks is not None:
            width = vectors.shape[VECTOR_LENGTH_AXIS]
            if update.chunks and chunks.schema == chunks_schema(width):
                chunks.delete(MATCH_EVERY_ROW)
            else:
                database.drop_table(CHUNKS_TABLE)
        files = self._open(FILES_TABLE)
        if files is not None:
            files.delete(MATCH_EVERY_ROW)
        for name in LEGACY_FILES:
            (self._directory / name).unlink(missing_ok=True)

    def _remove_files(self, update: IndexUpdate) -> None:
        """Remove the chunks of every file written or removed, and the records of the removed."""
        written_or_removed = [*(file.path for file in update.files), *update.removed_paths]
        chunks = self._open(CHUNKS_TABLE)
        if chunks is not None and written_or_removed:
            chunks.delete(path_filter(FILE_PATH_COLUMN, written_or_removed))
        files = self._open(FILES_TABLE)
        if files is not None and update.removed_paths:
            files.delete(path_filter(PATH_COLUMN, update.removed_paths))

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
        elif table.schema != rows.schema:
            # A model swapped under the same name: the manifest cannot tell, the table can
            message = (
                "The index holds chunks of another layout or vector width than the "
                f"{vectors.shape[VECTOR_LENGTH_AXIS]}-wide vectors the model now gives; "
                "re-index with force=true to build it again"
            )
            raise IndexWriteError(message)
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
