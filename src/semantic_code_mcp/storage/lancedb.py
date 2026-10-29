"""LanceDB storage of one project's index: the chunks, the record of each file they came
from, the vectors of each model that embedded them, and the manifest of what built them, in
the index's directory, which the store is built on (`index_directories.py` holds where it
lies, its lock, its manifest and the sweep of the project's unused indexes).

A write is one call, under the index's write lock, so two server processes on one index
never write side by side. Its chunks take the place of the chunks of the files it writes
and removes in one merge keyed by chunk id, so a reader finds a file's old chunks or its
new ones; then their records; the manifest last. A model's vectors live in a table of their
own, named by the digest of the model's identity, beside each vector its chunk's id and the
columns a scope filters by, as a vector search filters before it cuts and cannot join. A
chunk's id digests its path, occurrence and text, so a write keeps the vectors of the ids it
keeps and drops, before its chunks, those of the ids it drops: a write stopped between leaves
chunks to embed again, never a vector of a chunk that is gone. Vectors of another width than
a model's table holds (a model replaced under a name nothing pins) take the table's place.

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
rewrote the whole table. A model's table, which each addition gives a fragment, is pruned
after every addition and compacted once its fragments pile up.
"""

import threading
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import ClassVar, override

import lance
import lancedb
import numpy
import pyarrow
import structlog
from lancedb.query import FullTextOperator, MatchQuery
from pydantic import BaseModel, ConfigDict

from semantic_code_mcp.domain.globs import PATH_SEPARATOR
from semantic_code_mcp.domain.interfaces import Vector, VectorMatrix
from semantic_code_mcp.domain.models import (
    Candidate,
    Chunk,
    ChunkKind,
    FileStamp,
    IndexCounts,
    IndexedFiles,
    IndexUpdate,
    Language,
    SearchCandidates,
    SearchQuery,
    SearchScope,
    SourceFile,
    StoreKind,
    VectorIdentity,
)
from semantic_code_mcp.storage.globs import glob_pattern
from semantic_code_mcp.storage.index_directories import (
    DirectoryStore,
    IndexDirectory,
    IndexLayout,
    identity_digest,
)
from semantic_code_mcp.storage.search_text import search_text
from semantic_code_mcp.storage.similarity import VECTOR_LENGTH_AXIS, cosine_similarities

log = structlog.get_logger()

CHUNKS_TABLE = "chunks"
FILES_TABLE = "files"
# A model's vectors are kept in the table of this prefix and its identity's digest
VECTORS_TABLE_PREFIX = "vectors_"

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
# A model's table is compacted once an addition leaves it more fragments than this: each
# addition of a slice of vectors gives one, which a vector search reads side by side
VECTOR_FRAGMENTS_BEFORE_COMPACTION = 16
# Versions older than this are pruned by upkeep; a read finishes in milliseconds
OLD_VERSION_RETENTION = timedelta(seconds=10)

# Both tables name a file by its path from the project root, in POSIX form
PATH_COLUMN = "path"
CHUNK_ID_COLUMN = "chunk_id"
VECTOR_COLUMN = "vector"
LANGUAGE_COLUMN = "language"
KIND_COLUMN = "kind"
NAME_COLUMN = "name"
# A chunk's name as Python's casefold() reads it, which a search for a name matches
FOLDED_NAME_COLUMN = "folded_name"
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
CHUNKS_SCHEMA = pyarrow.schema(
    [
        pyarrow.field(CHUNK_ID_COLUMN, pyarrow.utf8()),
        pyarrow.field(PATH_COLUMN, pyarrow.utf8()),
        pyarrow.field(LANGUAGE_COLUMN, pyarrow.utf8()),
        pyarrow.field(KIND_COLUMN, pyarrow.utf8()),
        pyarrow.field(NAME_COLUMN, pyarrow.utf8()),
        pyarrow.field(FOLDED_NAME_COLUMN, pyarrow.utf8()),
        pyarrow.field(PARENT_COLUMN, pyarrow.utf8()),
        pyarrow.field(LINE_START_COLUMN, pyarrow.int32()),
        pyarrow.field(LINE_END_COLUMN, pyarrow.int32()),
        pyarrow.field(CONTENT_COLUMN, pyarrow.utf8()),
        pyarrow.field(SEARCH_TEXT_COLUMN, pyarrow.utf8()),
    ]
)
# What a model's table holds of each chunk beside its vector: its id, and the columns a
# search's scope filters by
VECTOR_SCOPE_COLUMNS = (CHUNK_ID_COLUMN, PATH_COLUMN, KIND_COLUMN, LANGUAGE_COLUMN)


def vectors_schema(dimension: int) -> pyarrow.Schema:
    """A model's table's columns, its vectors `dimension` wide."""
    return pyarrow.schema(
        [
            pyarrow.field(CHUNK_ID_COLUMN, pyarrow.utf8()),
            pyarrow.field(PATH_COLUMN, pyarrow.utf8()),
            pyarrow.field(KIND_COLUMN, pyarrow.utf8()),
            pyarrow.field(LANGUAGE_COLUMN, pyarrow.utf8()),
            pyarrow.field(VECTOR_COLUMN, pyarrow.list_(pyarrow.float32(), dimension)),
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

    def chunk(self) -> Chunk:
        """This row as the chunk it holds."""
        return Chunk(
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


# What a search reads of each chunk: the columns a stored chunk is made of, which its fields
# name; not its search text
CHUNK_COLUMNS = tuple(StoredChunk.model_fields)


class StoredFile(BaseModel):
    """A row of the files table."""

    model_config = ConfigDict(frozen=True)

    path: str
    mtime_ns: int
    size_bytes: int

    def stamp(self) -> FileStamp:
        """This row's file as the index recorded it."""
        return FileStamp(mtime_ns=self.mtime_ns, size_bytes=self.size_bytes)


class LanceDBIndexStore(DirectoryStore):
    """One project's index of one identity, in a LanceDB database in the index's directory.

    Nothing touches the disk until it is first used, and only a write makes the
    directory; the connection is opened once, by whichever worker thread needs it first.
    """

    # The schema's version is bumped by any change to the tables' columns or to what one is
    # made from, its full-text index among them (5: stop words kept; 6: vectors in a table
    # of their own, by model, and the chunks' names case-folded)
    LAYOUT: ClassVar[IndexLayout] = IndexLayout(store_kind=StoreKind.lancedb, schema_version=6)

    def __init__(self, directory: IndexDirectory) -> None:
        super().__init__(directory)
        self._connecting = threading.Lock()
        self._connection: lancedb.DBConnection | None = None
        # How many upkeeps this store runs before it compacts; none before its first
        self._upkeeps_until_compaction = 0

    @override
    def indexed_files(self) -> IndexedFiles:
        table = self._open(FILES_TABLE)
        rows = [] if table is None else table.to_arrow().to_pylist()
        stored = [StoredFile.model_validate(row) for row in rows]
        records = {file.path: file.stamp() for file in stored}
        return IndexedFiles(
            records=records,
            unrecorded_paths=[path for path in self._chunked_paths() if path not in records],
        )

    @override
    def counts(self, model: VectorIdentity) -> IndexCounts:
        return IndexCounts(
            files=self._count(FILES_TABLE),
            chunks=self._count(CHUNKS_TABLE),
            chunks_with_vectors=self._count(vectors_table_name(model)),
        )

    @override
    def write(self, update: IndexUpdate) -> None:
        with self._directory.writing(records_manifest=True):
            database = self._connect()
            self._write_chunks(database, update)
            if update.last_of_run:
                # Before the records: a write stopped before the fold leaves its files stale
                self._fold(CHUNKS_TABLE)
            self._write_records(database, update)
            if update.last_of_run:
                # After them: a reader meanwhile finds the files the run wrote recorded
                self._keep_up()
        log.debug(
            "index_written",
            files=len(update.files),
            chunks=len(update.chunks),
            removed=len(update.removed_paths),
        )

    @override
    def chunks_without_vectors(self, model: VectorIdentity) -> list[Chunk]:
        chunks = self._open(CHUNKS_TABLE)
        if chunks is None:
            return []
        embedded = set(self._vector_chunk_ids(model))
        rows = chunks.search().select(list(CHUNK_COLUMNS)).limit(None).to_arrow().to_pylist()
        return [
            StoredChunk.model_validate(row).chunk()
            for row in rows
            if row[CHUNK_ID_COLUMN] not in embedded
        ]

    @override
    def add_vectors(
        self, model: VectorIdentity, chunk_ids: list[str], vectors: VectorMatrix
    ) -> None:
        if len(chunk_ids) != len(vectors):
            message = f"{len(chunk_ids)} chunk ids but {len(vectors)} vectors"
            raise ValueError(message)
        if not chunk_ids:
            return
        with self._directory.writing(records_manifest=True):
            chunks = self._open(CHUNKS_TABLE)
            if chunks is None:
                # No chunks table holds no chunk to add a vector to
                return
            held = rows_with_ids(chunks, chunk_ids, VECTOR_SCOPE_COLUMNS)
            rows = vectors_rows(held, chunk_ids, vectors)
            self._add_to_model_table(self._connect(), vectors_table_name(model), rows)
        log.debug("vectors_added", model=model.model_name, chunks=rows.num_rows)

    @override
    def remove_vectors(self, model: VectorIdentity) -> None:
        name = vectors_table_name(model)
        with self._directory.writing(records_manifest=True):
            if self._open(name) is not None:
                self._connect().drop_table(name)

    @override
    def search_words(self, query: SearchQuery, limit: int) -> list[Chunk]:
        chunks = self._open(CHUNKS_TABLE)
        if chunks is None:
            return []
        return matching_chunks(chunks, query.text, scope_filter(query.scope), limit)

    @override
    def chunks_named(self, folded_name: str, scope: SearchScope) -> list[Chunk]:
        chunks = self._open(CHUNKS_TABLE)
        if chunks is None:
            return []
        named = values_filter(FOLDED_NAME_COLUMN, [folded_name])
        scope_condition = scope_filter(scope)
        condition = named if scope_condition is None else SQL_AND.join([named, scope_condition])
        rows = chunks.search().where(condition).select(list(CHUNK_COLUMNS)).limit(None).to_arrow()
        found = [StoredChunk.model_validate(row).chunk() for row in rows.to_pylist()]
        return sorted(found, key=lambda chunk: (chunk.path, chunk.line_start))

    @override
    def search(
        self, model: VectorIdentity, query_vector: Vector, query: SearchQuery, per_channel: int
    ) -> SearchCandidates:
        chunks = self._open(CHUNKS_TABLE)
        if chunks is None:
            return NO_CANDIDATES
        # Each table is opened once, at its latest version, so the two channels read one
        # version of each; a vector whose chunk the chunks' version no longer holds is left out
        vectors = self._open(vectors_table_name(model))
        scope = scope_filter(query.scope)
        return SearchCandidates(
            by_vector=nearest_candidates(chunks, vectors, query_vector, scope, per_channel),
            by_keyword=with_similarities(
                matching_chunks(chunks, query.text, scope, per_channel), vectors, query_vector
            ),
        )

    def _write_chunks(self, database: lancedb.DBConnection, update: IndexUpdate) -> None:
        """Put the chunks of the files written in place of what the index held of them and
        of the files removed, in one commit; a chunk whose id is stored already is updated
        where it stands, and keeps its vectors. The vectors of the chunks dropped go first.
        Makes the table, and its full-text index, if there is none."""
        table = self._open(CHUNKS_TABLE)
        if table is None:
            if not update.chunks:
                return
            table = database.create_table(CHUNKS_TABLE, data=chunks_table(update.chunks))
        else:
            replaced = [*(file.path for file in update.files), *update.removed_paths]
            if not replaced:
                return
            self._drop_vectors_of_dropped_chunks(table, replaced, update.chunks)
            rows = chunks_table(update.chunks) if update.chunks else CHUNKS_SCHEMA.empty_table()
            (
                table.merge_insert(CHUNK_ID_COLUMN)
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .when_not_matched_by_source_delete(path_filter(PATH_COLUMN, replaced))
                .execute(rows)
            )
        if not has_full_text_index(table):
            make_full_text_index(table)

    def _drop_vectors_of_dropped_chunks(
        self, chunks: lancedb.table.Table, replaced: list[Path], written: list[Chunk]
    ) -> None:
        """Drop every model's vectors of the chunks the files replaced held that the write
        does not keep by id; nothing to read where no model has a table."""
        tables = self._vector_tables()
        if not tables:
            return
        kept = {chunk.chunk_id for chunk in written}
        rows = (
            chunks.search()
            .where(path_filter(PATH_COLUMN, replaced))
            .select([CHUNK_ID_COLUMN])
            .limit(None)
            .to_arrow()
        )
        stored_ids = rows.column(CHUNK_ID_COLUMN).to_pylist()
        dropped = [chunk_id for chunk_id in stored_ids if chunk_id not in kept]
        if not dropped:
            return
        for table in tables:
            table.delete(values_filter(CHUNK_ID_COLUMN, dropped))

    def _add_to_model_table(
        self, database: lancedb.DBConnection, name: str, rows: pyarrow.Table
    ) -> None:
        """Add vectors to a model's table, made by its first addition as wide as they are and
        made again when they are of another width; a chunk's vector stored already, which
        another session may have added, stays. Pruned after, and compacted when its fragments
        pile up."""
        table = self._open(name)
        if table is not None and table.schema != rows.schema:
            log.warning("model_vectors_replaced", directory=str(self._directory.path), table=name)
            database.drop_table(name)
            table = None
        if table is None:
            table = database.create_table(name, data=rows)
        else:
            table.merge_insert(CHUNK_ID_COLUMN).when_not_matched_insert_all().execute(rows)
        dataset = table.to_lance()
        if len(dataset.get_fragments()) > VECTOR_FRAGMENTS_BEFORE_COMPACTION:
            dataset.optimize.compact_files(target_rows_per_fragment=ROWS_PER_FRAGMENT)
        prune_old_versions(table.to_lance())

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
        """The paths, as stored, of the files the chunks come from: one column of every row."""
        table = self._open(CHUNKS_TABLE)
        if table is None:
            return []
        rows = table.search().select([PATH_COLUMN]).limit(None).to_arrow()
        return rows.column(PATH_COLUMN).unique().to_pylist()

    def _vector_chunk_ids(self, model: VectorIdentity) -> list[str]:
        """The ids of the chunks the model's table holds a vector of."""
        table = self._open(vectors_table_name(model))
        if table is None:
            return []
        rows = table.search().select([CHUNK_ID_COLUMN]).limit(None).to_arrow()
        return rows.column(CHUNK_ID_COLUMN).to_pylist()

    def _vector_tables(self) -> list[lancedb.table.Table]:
        """The tables of every model's vectors, each at its latest version."""
        database = self._existing_database()
        if database is None:
            return []
        return [
            database.open_table(name)
            for name in database.list_tables().tables
            if name.startswith(VECTORS_TABLE_PREFIX)
        ]

    def _fold(self, name: str) -> None:
        """Fold a table's new rows into its indexes, if it is made."""
        table = self._open(name)
        if table is not None:
            table.to_lance().optimize.optimize_indices()

    def _keep_up(self) -> None:
        """Compact the tables when due, then prune each, the models' tables among them, which
        a write's drops leave versions of: what a run's end does once its records are written.
        Nothing is left to fold: the run's chunks were folded before the records, and a
        compaction rebuilds the full-text index whole; a fold of nothing would only leave
        another version."""
        self._compact_when_due()
        tables = [self._open(CHUNKS_TABLE), self._open(FILES_TABLE), *self._vector_tables()]
        for table in tables:
            if table is not None:
                prune_old_versions(table.to_lance())
                log.debug("table_kept_up", table=table.name)

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


def vectors_table_name(model: VectorIdentity) -> str:
    """The table a model's vectors are kept in."""
    return VECTORS_TABLE_PREFIX + identity_digest(model)


def chunks_table(chunks: list[Chunk]) -> pyarrow.Table:
    """Chunks as rows of the chunks schema."""
    return pyarrow.table(
        {
            CHUNK_ID_COLUMN: [chunk.chunk_id for chunk in chunks],
            PATH_COLUMN: [chunk.path.as_posix() for chunk in chunks],
            LANGUAGE_COLUMN: [chunk.language for chunk in chunks],
            KIND_COLUMN: [chunk.kind for chunk in chunks],
            NAME_COLUMN: [chunk.name for chunk in chunks],
            FOLDED_NAME_COLUMN: [chunk.name.casefold() for chunk in chunks],
            PARENT_COLUMN: [chunk.parent for chunk in chunks],
            LINE_START_COLUMN: [chunk.line_start for chunk in chunks],
            LINE_END_COLUMN: [chunk.line_end for chunk in chunks],
            CONTENT_COLUMN: [chunk.content for chunk in chunks],
            SEARCH_TEXT_COLUMN: [search_text(chunk) for chunk in chunks],
        },
        schema=CHUNKS_SCHEMA,
    )


def vectors_rows(held: pyarrow.Table, chunk_ids: list[str], vectors: VectorMatrix) -> pyarrow.Table:
    """The vectors of the chunks the chunks table holds, as rows of a model's table: each
    with its chunk's scope columns, those of ids it no longer holds left out. The matrix goes
    into Arrow as one flat buffer, without a Python float per component."""
    position_of = {chunk_id: position for position, chunk_id in enumerate(chunk_ids)}
    positions = [position_of[chunk_id] for chunk_id in held.column(CHUNK_ID_COLUMN).to_pylist()]
    kept = vectors[positions].astype(numpy.float32)
    width = vectors.shape[VECTOR_LENGTH_AXIS]
    vector_column = pyarrow.FixedSizeListArray.from_arrays(pyarrow.array(kept.ravel()), width)
    return pyarrow.table(
        {
            **{column: held.column(column) for column in VECTOR_SCOPE_COLUMNS},
            VECTOR_COLUMN: vector_column,
        },
        schema=vectors_schema(width),
    )


def rows_with_ids(
    table: lancedb.table.Table, chunk_ids: Sequence[str], columns: Sequence[str]
) -> pyarrow.Table:
    """The columns of the rows of the chunks of these ids a table holds, in its order."""
    return (
        table.search()
        .where(values_filter(CHUNK_ID_COLUMN, chunk_ids))
        .select(list(columns))
        .limit(None)
        .to_arrow()
    )


def files_table(files: list[SourceFile]) -> pyarrow.Table:
    """The records of indexed files as rows of the files schema."""
    return pyarrow.table(
        {
            PATH_COLUMN: [file.path.as_posix() for file in files],
            MTIME_COLUMN: [file.stamp.mtime_ns for file in files],
            SIZE_COLUMN: [file.stamp.size_bytes for file in files],
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


def prune_old_versions(dataset: lance.LanceDataset) -> None:
    """Prune a table's versions older than OLD_VERSION_RETENTION, among them what a fold,
    a compaction or a rebuild before it replaced."""
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
    return values_filter(column, [path.as_posix() for path in paths])


def nearest_candidates(
    chunks: lancedb.table.Table,
    vectors: lancedb.table.Table | None,
    query_vector: Vector,
    scope: str | None,
    limit: int,
) -> list[Candidate]:
    """The chunks of the scope whose vectors of the model lie nearest the query's, nearest
    first; none where the model has no vectors."""
    if vectors is None:
        return []
    # lancedb's stubs type search() as the base builder, which lacks distance_type()
    by_vector = vectors.search(query_vector).distance_type(COSINE_METRIC)  # ty: ignore[unresolved-attribute]
    if scope is not None:
        # Filtered before the nearest are taken, so a narrow scope still fills them
        by_vector = by_vector.where(scope, prefilter=True)
    rows = (
        by_vector.select([CHUNK_ID_COLUMN, VECTOR_COLUMN, DISTANCE_COLUMN]).limit(limit).to_arrow()
    )
    found_ids = rows.column(CHUNK_ID_COLUMN).to_pylist()
    similarities = cosine_similarities(vector_matrix(rows), query_vector)
    chunks_by_id = stored_chunks_by_id(chunks, found_ids)
    return [
        Candidate(chunk=chunks_by_id[chunk_id], similarity=float(similarity))
        for chunk_id, similarity in zip(found_ids, similarities, strict=True)
        if chunk_id in chunks_by_id
    ]


def with_similarities(
    found: list[Chunk], vectors: lancedb.table.Table | None, query_vector: Vector
) -> list[Candidate]:
    """Chunks the keyword channel found, in their order, each with its vector's similarity
    where the model's table holds one."""
    similarities = similarities_by_id(vectors, [chunk.chunk_id for chunk in found], query_vector)
    return [Candidate(chunk=chunk, similarity=similarities.get(chunk.chunk_id)) for chunk in found]


def matching_chunks(
    chunks: lancedb.table.Table, query_text: str, scope: str | None, limit: int
) -> list[Chunk]:
    """The chunks of the scope whose words best match the query's, best first; none while
    the table has no full-text index, as after a write stopped between the table and its
    index. The text is matched as words, any of which may match, never read as LanceDB's
    query syntax: an agent's question wholly in quotes would be a phrase, which an index
    without word positions cannot answer."""
    if not has_full_text_index(chunks):
        return []
    words = MatchQuery(query_text, SEARCH_TEXT_COLUMN, operator=FullTextOperator.OR)
    by_keyword = chunks.search(words, query_type=FULL_TEXT_QUERY_TYPE)
    if scope is not None:
        by_keyword = by_keyword.where(scope)
    rows = by_keyword.select([*CHUNK_COLUMNS, FULL_TEXT_SCORE_COLUMN]).limit(limit).to_arrow()
    return [StoredChunk.model_validate(row).chunk() for row in rows.to_pylist()]


def stored_chunks_by_id(chunks: lancedb.table.Table, chunk_ids: list[str]) -> dict[str, Chunk]:
    """The chunks of these ids the chunks table holds, by id."""
    if not chunk_ids:
        return {}
    rows = rows_with_ids(chunks, chunk_ids, CHUNK_COLUMNS).to_pylist()
    return {row[CHUNK_ID_COLUMN]: StoredChunk.model_validate(row).chunk() for row in rows}


def similarities_by_id(
    vectors: lancedb.table.Table | None, chunk_ids: list[str], query_vector: Vector
) -> dict[str, float]:
    """The similarity to the query of the vector of each of these chunks the model's table
    holds, by chunk id."""
    if vectors is None or not chunk_ids:
        return {}
    rows = rows_with_ids(vectors, chunk_ids, [CHUNK_ID_COLUMN, VECTOR_COLUMN])
    similarities = cosine_similarities(vector_matrix(rows), query_vector)
    return dict(zip(rows.column(CHUNK_ID_COLUMN).to_pylist(), similarities.tolist(), strict=True))


def scope_filter(scope: SearchScope) -> str | None:
    """The filter that keeps a search's channels to its scope; None for a scope that
    leaves nothing out. It reads only the columns both the chunks table and a model's table
    hold."""
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


def vector_matrix(rows: pyarrow.Table) -> VectorMatrix:
    """The vectors of rows as a matrix, a row each, read as one flat buffer."""
    vectors = rows.column(VECTOR_COLUMN).combine_chunks()
    return vectors.flatten().to_numpy().reshape(rows.num_rows, vectors.type.list_size)


def quote_sql(value: str) -> str:
    """A string as a quoted SQL literal."""
    return SQL_QUOTE + value.replace(SQL_QUOTE, SQL_ESCAPED_QUOTE) + SQL_QUOTE
