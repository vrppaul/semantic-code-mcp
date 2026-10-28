"""LanceDB storage of one project's index: the chunks with their vectors, the record of
each file they came from, and the manifest of what built them, in a directory of its own
(`index_directories.py` holds where it lies, its lock, its manifest and the sweep of the
project's unused indexes).

A write is one call, under the index's write lock, so two server processes on one index
never write side by side. Its chunks take the place of the chunks of the files it writes
and removes in one merge keyed by chunk id, so a reader finds a file's old chunks or its
new ones; then their records; the manifest last. The chunks table is made by
the first write that has vectors, as wide as they are; vectors of another width (a model
replaced under a name nothing pins) drop the records, then the table, and the index starts
again from the write, its other files unrecorded so the next run indexes them.

Reads take no lock: each opens its table again, at its latest version, so a long-lived
connection still sees what another process wrote, and a table not yet made reads as
empty. So old versions are pruned only once they are older than any read can last.

A run's last write keeps the index up. After its chunks and before their records it folds
the new rows into the full-text index, as rows it has not folded in score on another scale:
a run cancelled as that write lands still leaves them folded, and a write stopped before the
fold leaves its files stale for the next run to write and fold again. After the records it
prunes old versions, and now and then first compacts the tables and rebuilds the full-text
index whole, which takes seconds at 100k chunks, so a reader meanwhile finds the run's files
recorded: each fold adds the index a part no later one merges, and counts the rows an edit
replaced in the index's word statistics until then. LanceDB's own optimize compacts at
every call, merging every fragment under a million rows, so at 100k chunks each edit
rewrote the whole table.
"""

import threading
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import override

import lance
import lancedb
import numpy
import pyarrow
import structlog
from lancedb.query import FullTextOperator, MatchQuery
from pydantic import BaseModel, ConfigDict

from semantic_code_mcp.domain.globs import PATH_SEPARATOR
from semantic_code_mcp.domain.interfaces import IndexStore, Vector, VectorMatrix
from semantic_code_mcp.domain.models import (
    Candidate,
    Chunk,
    ChunkKind,
    IndexCounts,
    IndexedFiles,
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
from semantic_code_mcp.storage.index_directories import IndexDirectory, index_directory_name
from semantic_code_mcp.storage.search_text import search_text
from semantic_code_mcp.storage.similarity import VECTOR_LENGTH_AXIS, cosine_similarities

log = structlog.get_logger()

# Bumped by any change to the tables' columns or to what one is made from, its full-text
# index among them (5: stop words kept); each version keeps an index of its own
SCHEMA_VERSION = 5

CHUNKS_TABLE = "chunks"
FILES_TABLE = "files"

# A store compacts the tables and rebuilds the full-text index at its first upkeep, as it
# cannot know what an earlier process left, and once in this many after: between, parts of
# the index and fragments pile up, and replaced rows still count in the word statistics
# (100k chunks: an upkeep 69-77 ms, one that compacts 2.3-2.4 s)
UPKEEPS_PER_COMPACTION = 20
# ... and at once when the full-text index counts more rows than the table holds by this
# share of them: the rows writes replaced weigh in its word statistics until a rebuild, and
# a file edited again and again piles them up (19 appends to click's core.py moved 4 of 32
# answers one place, a review's probe)
REPLACED_ROWS_SHARE_BEFORE_COMPACTION = 0.05
# Compaction merges fragments up to this many rows, so a large table keeps fragments a flat
# vector search reads side by side (100k chunks: a search 40 ms, 54 with one fragment)
ROWS_PER_FRAGMENT = 4096
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
FULL_TEXT_QUERY_TYPE = "fts"
FULL_TEXT_INDEX_TYPE = "FTS"
# The full-text index's name, the one LanceDB gives it by default
FULL_TEXT_INDEX_NAME = f"{SEARCH_TEXT_COLUMN}_idx"
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
# What parts the values of an SQL list, `IN ('a', 'b')`
SQL_LIST_SEPARATOR = ", "
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
        self._identity = identity
        self._directory = IndexDirectory(
            project_directory,
            index_directory_name(identity, SCHEMA_VERSION),
            lock_timeout_seconds=lock_timeout_seconds,
        )
        self._connecting = threading.Lock()
        self._connection: lancedb.DBConnection | None = None
        # How many upkeeps this store runs before it compacts; none before its first
        self._upkeeps_until_compaction = 0

    @property
    def directory(self) -> Path:
        """Where this index lives."""
        return self._directory.path

    @property
    def write_lock(self) -> Path:
        """The file a writer of this index locks, beside its directory."""
        return self._directory.write_lock

    @override
    def manifest(self) -> IndexManifest | None:
        return self._directory.manifest()

    @override
    def indexed_files(self) -> IndexedFiles:
        table = self._open(FILES_TABLE)
        rows = [] if table is None else table.to_arrow().to_pylist()
        stored = [StoredFile.model_validate(row) for row in rows]
        recorded_paths = {file.path for file in stored}
        return IndexedFiles(
            records=[file.source() for file in stored],
            unrecorded_paths=[
                Path(path) for path in self._chunked_paths() if path not in recorded_paths
            ],
        )

    @override
    def counts(self) -> IndexCounts:
        return IndexCounts(files=self._count(FILES_TABLE), chunks=self._count(CHUNKS_TABLE))

    @override
    def stored_vectors(self, chunk_ids: list[str]) -> dict[str, Vector]:
        table = self._open(CHUNKS_TABLE)
        if table is None or not chunk_ids:
            return {}
        quoted = SQL_LIST_SEPARATOR.join(quote_sql(chunk_id) for chunk_id in chunk_ids)
        rows = (
            table.search()
            .where(f"{CHUNK_ID_COLUMN} IN ({quoted})")
            .select([CHUNK_ID_COLUMN, VECTOR_COLUMN])
            .limit(None)
            .to_arrow()
        )
        matrix = vector_matrix(rows)
        found_ids = rows.column(CHUNK_ID_COLUMN).to_pylist()
        return dict(zip(found_ids, numpy.unstack(matrix), strict=True))

    @override
    def write(self, update: IndexUpdate, vectors: VectorMatrix) -> None:
        if len(update.chunks) != len(vectors):
            message = f"{len(update.chunks)} chunks but {len(vectors)} vectors"
            raise ValueError(message)
        with self._directory.writing():
            database = self._connect()
            if update.chunks:
                self._fit_layout(database, vectors.shape[VECTOR_LENGTH_AXIS])
            self._write_chunks(database, update, vectors)
            if update.last_of_run:
                # Before the records: a write stopped before the fold leaves its files stale
                self._fold(CHUNKS_TABLE)
            self._write_records(database, update)
            if update.last_of_run:
                # After them: a reader meanwhile finds the files the run wrote recorded
                self._keep_up()
            manifest = IndexManifest(
                identity=self._identity,
                schema_version=SCHEMA_VERSION,
                indexed_at=datetime.now(UTC),
            )
            self._directory.write_manifest(manifest)
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
        log.warning("index_layout_replaced", directory=str(self.directory), width=width)
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

    def _chunked_paths(self) -> list[str]:
        """The paths, as stored, of the files the chunks come from: one column of every row,
        no vectors read."""
        table = self._open(CHUNKS_TABLE)
        if table is None:
            return []
        rows = table.search().select([PATH_COLUMN]).limit(None).to_arrow()
        return rows.column(PATH_COLUMN).unique().to_pylist()

    def _fold(self, name: str) -> None:
        """Fold a table's new rows into its indexes, if it is made."""
        table = self._open(name)
        if table is not None:
            table.to_lance().optimize.optimize_indices()

    def _keep_up(self) -> None:
        """Compact the tables when due, then fold and prune each: what a run's end does once
        its records are written."""
        self._compact_when_due()
        for name in (CHUNKS_TABLE, FILES_TABLE):
            table = self._open(name)
            if table is not None:
                fold_and_prune(table.to_lance())
                log.debug("table_kept_up", table=name)

    def _compact_when_due(self) -> None:
        """Compact both tables, then rebuild the full-text index whole over the rows left, at
        this store's first upkeep, once in UPKEEPS_PER_COMPACTION after it, and whenever the
        rows writes replaced weigh too much in the full-text index."""
        chunks = self._open(CHUNKS_TABLE)
        replaced_rows_weigh = chunks is not None and replaced_rows_weigh_too_much(chunks)
        if self._upkeeps_until_compaction > 0 and not replaced_rows_weigh:
            self._upkeeps_until_compaction -= 1
            return
        for name in (CHUNKS_TABLE, FILES_TABLE):
            table = self._open(name)
            if table is not None:
                table.to_lance().optimize.compact_files(target_rows_per_fragment=ROWS_PER_FRAGMENT)
        chunks = self._open(CHUNKS_TABLE)
        if chunks is not None:
            make_full_text_index(chunks)
        self._upkeeps_until_compaction = UPKEEPS_PER_COMPACTION - 1
        log.debug("index_compacted")

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
        self._directory.note_use()
        if not self._directory.exists():
            return None
        return self._connect()

    def _connect(self) -> lancedb.DBConnection:
        """The connection, opened once, by whichever thread needs it first."""
        with self._connecting:
            if self._connection is None:
                self._connection = lancedb.connect(str(self._directory.path))
            return self._connection


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


def replaced_rows_weigh_too_much(chunks: lancedb.table.Table) -> bool:
    """Whether the full-text index counts rows the table no longer holds, the rows writes
    replaced, past REPLACED_ROWS_SHARE_BEFORE_COMPACTION of those it does. The index's own
    statistics count them; the table's fragments do not, as Lance drops a fragment whose
    rows were all replaced."""
    statistics = chunks.to_lance().stats.index_stats(FULL_TEXT_INDEX_NAME)
    counted_rows = sum(part["num_docs"] for part in statistics["indices"])
    held_rows = statistics["num_indexed_rows"]
    return counted_rows - held_rows > REPLACED_ROWS_SHARE_BEFORE_COMPACTION * held_rows


def fold_and_prune(dataset: lance.LanceDataset) -> None:
    """Fold a table's new rows into its indexes, then prune its versions older than
    OLD_VERSION_RETENTION, among them what a compaction or rebuild before it replaced."""
    dataset.optimize.optimize_indices()
    dataset.cleanup_old_versions(older_than=OLD_VERSION_RETENTION)


def make_full_text_index(table: lancedb.table.Table) -> None:
    """Index the chunks' search text for full-text search, whole, in place of any index the
    table has."""
    table.create_fts_index(
        SEARCH_TEXT_COLUMN,
        replace=True,
        remove_stop_words=REMOVE_STOP_WORDS,
        name=FULL_TEXT_INDEX_NAME,
    )


def has_full_text_index(table: lancedb.table.Table) -> bool:
    return any(index.index_type == FULL_TEXT_INDEX_TYPE for index in table.list_indices())


def path_filter(column: str, paths: list[Path]) -> str:
    """A filter matching the rows whose column holds one of the paths."""
    quoted = SQL_LIST_SEPARATOR.join(quote_sql(path.as_posix()) for path in paths)
    return f"{column} IN ({quoted})"


def keyword_candidates(
    table: lancedb.table.Table,
    query_text: str,
    scope: str | None,
    query_vector: Vector,
    limit: int,
) -> list[Candidate]:
    """The chunks of the scope whose words best match the query's, best first; none while
    the table has no full-text index, as after a write stopped between the table and its
    index. The text is matched as words, any of which may match, never read as LanceDB's
    query syntax: an agent's question wholly in quotes would be a phrase, which an index
    without word positions cannot answer."""
    if not has_full_text_index(table):
        return []
    words = MatchQuery(query_text, SEARCH_TEXT_COLUMN, operator=FullTextOperator.OR)
    by_keyword = table.search(words, query_type=FULL_TEXT_QUERY_TYPE)
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
    quoted = SQL_LIST_SEPARATOR.join(quote_sql(value) for value in values)
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


def quote_sql(value: str) -> str:
    """A string as a quoted SQL literal."""
    return SQL_QUOTE + value.replace(SQL_QUOTE, SQL_ESCAPED_QUOTE) + SQL_QUOTE
