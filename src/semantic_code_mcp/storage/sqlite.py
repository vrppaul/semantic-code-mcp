"""SQLite storage of one project's index: the chunks, their words in a full-text index, the
record of each file they came from, the vectors of each model that embedded them, and the
manifest of what built them, in one database file in the index's directory, which the store
is built on (`index_directories.py` holds where it lies, its lock, its manifest and the sweep
of the project's unused indexes).

A write of words is one transaction, under the index's write lock beside SQLite's own: the
chunks of the files written take the place of those files' and the removed files' chunks,
their words with them, then the files' records; the manifest follows the commit. So a reader
finds a file's old chunks or its new ones with their records, a write stopped midway leaves
the index as it was, and no chunk is ever left without its record. The full-text index is
FTS5's, kept exact row by row, so nothing is folded at a run's end; it holds no copy of the
text (`content=''`), only what finds it.

A run that finds no database builds it whole (ADR 009, D12): its writes go to a database of
their own beside the index's, without a journal or a wait for the disk, and its last write
switches it to WAL and renames it into place, so no reader sees it before it is whole, and a
build stopped midway leaves only a file the next build deletes. Its writes put chunks in a
database that holds none of their files' yet, so they delete nothing first.

Vectors live in a table of their own, keyed by the digest of the model's identity and the
chunk's id, which digests the chunk's path, occurrence and text: a vector stored under the
two is that text's vector of that model for as long as both live. So a write of words keeps
the vectors of the chunk ids it keeps and drops those of the ids it drops, and a model's
vectors are added, in a transaction each, to chunks already stored.

Each call opens a connection of its own, as the store's calls come from worker threads, and a
search reads both its channels inside one read transaction: one state of the index, while
another process writes beside it (WAL). A question is matched as words, any of which may
match: each token as FTS5's tokenizer splits text, quoted, the tokens joined by OR, so an
agent's question is never read as FTS5's own syntax (ADR 007, D7).

The vector channel searches a matrix of a model's vectors the store holds between searches
(ADR 009, D8), which the search's read transaction checks is of the state it reads: a
vector's row id is never given twice in one database and a stored vector never changes, so
the database's own id, made with it, the last row id the vectors table gave and the model's
count of vectors tell whether any write, of this process or another, changed them since, or
another database took the place of the one the matrix was read from. A later state of the
same database refreshes the matrix by its rows, the deleted dropped and the new read, and
both channels take their similarities from it.
"""

import re
import sqlite3
import threading
import uuid
from collections.abc import Generator, Sequence
from contextlib import closing, contextmanager
from pathlib import Path
from typing import ClassVar, override

import numpy
import structlog
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from semantic_code_mcp.domain.globs import PATH_SEPARATOR
from semantic_code_mcp.domain.interfaces import Vector, VectorMatrix
from semantic_code_mcp.domain.models import (
    Candidate,
    Chunk,
    ChunkKind,
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
from semantic_code_mcp.storage.similarity import (
    VECTOR_LENGTH_AXIS,
    unit_rows,
    unit_similarities,
    unit_vector,
)

log = structlog.get_logger()

DATABASE_FILE = "index.sqlite"
# A new database is made under this name beside its own, then renamed into place whole
UNFINISHED_SUFFIX = ".unfinished"
# FTS5's tokenizer: Porter's stemmer over unicode61's words, which split at what is no letter
# or digit, as `search_text` expects of identifiers' parts; stop words are kept
FULL_TEXT_TOKENIZER = "porter unicode61"
# A token of a question as unicode61 reads one: letters and digits
QUERY_TOKEN = re.compile(r"[^\W_]+")
QUOTE = '"'
EITHER_WORD = " OR "
VECTOR_TYPE = numpy.float32
VECTOR_COMPONENT_BYTES = numpy.dtype(VECTOR_TYPE).itemsize
ROW_ID_TYPE = numpy.int64
# What the vectors are read after for a matrix read whole: before every row id SQLite gives
BEFORE_EVERY_ROW = 0
# The count of a model's vectors where it has none, whose search finds by words alone
NO_VECTORS = 0
# The function SQLite's REGEXP operator calls, with its pattern and the text, in that order
REGEXP_FUNCTION = "regexp"
REGEXP_ARGUMENTS = 2
# Statements are built of this module's constants and placeholders alone; every value a
# search or a write brings is bound, never written into them
# What a scope that leaves nothing out filters by
NO_CONDITION = "1"
CONDITION_SEPARATOR = " AND "
VALUE_SEPARATOR = ", "
VALUE_PLACEHOLDER = "?"
# How far a new chunk's row id lies past the last one the table gave
NEXT_ROW_ID_STEP = 1

CREATE_CHUNKS = (
    "CREATE TABLE chunks ("
    "row_id INTEGER PRIMARY KEY AUTOINCREMENT, chunk_id TEXT NOT NULL UNIQUE, "
    "path TEXT NOT NULL, "
    "language TEXT NOT NULL, kind TEXT NOT NULL, name TEXT NOT NULL, folded_name TEXT NOT NULL, "
    "parent TEXT, line_start INTEGER NOT NULL, line_end INTEGER NOT NULL, content TEXT NOT NULL)"
)
# Finds a file's chunks for a write, and gives a search's scope from its narrow entries
# rather than from the chunks' rows, which their text makes wide
CREATE_CHUNKS_BY_PATH = "CREATE INDEX chunks_by_path ON chunks (path, kind, language)"
# Finds the chunks of a name, its case aside, as Python's casefold() reads it, not SQLite's
# NOCASE, which folds ASCII alone
CREATE_CHUNKS_BY_NAME = "CREATE INDEX chunks_by_name ON chunks (folded_name)"
CREATE_FILES = (
    "CREATE TABLE files ("
    "path TEXT PRIMARY KEY, mtime_ns INTEGER NOT NULL, size_bytes INTEGER NOT NULL)"
)
CREATE_WORDS = (
    "CREATE VIRTUAL TABLE words USING fts5("
    f"search_text, tokenize='{FULL_TEXT_TOKENIZER}', content='', contentless_delete=1)"
)
# A vector of a chunk by a model, the model named by its identity's digest; keyed by the
# chunk's id first, so a write of words drops a dropped chunk's vectors of every model by it
CREATE_VECTORS = (
    "CREATE TABLE vectors ("
    "row_id INTEGER PRIMARY KEY AUTOINCREMENT, model TEXT NOT NULL, chunk_id TEXT NOT NULL, "
    "vector BLOB NOT NULL, UNIQUE (chunk_id, model))"
)
# A random id made with the database, one row: a database made again in its place starts its
# row ids again, so a matrix held of the old one is told apart by it
CREATE_DATABASE_ID = "CREATE TABLE database (id TEXT NOT NULL)"
CREATE_TABLES = (
    CREATE_CHUNKS,
    CREATE_CHUNKS_BY_PATH,
    CREATE_CHUNKS_BY_NAME,
    CREATE_FILES,
    CREATE_WORDS,
    CREATE_VECTORS,
    CREATE_DATABASE_ID,
)
INSERT_DATABASE_ID = "INSERT INTO database (id) VALUES (?)"
SELECT_DATABASE_ID = "SELECT id FROM database"
# Opens a database that exists, never makes one: a read racing the index's removal fails
# rather than leave an empty file a write would take for a made database
EXISTING_DATABASE = "?mode=rw"
WRITE_AHEAD_LOG = "PRAGMA journal_mode=WAL"
# A whole build's database keeps no journal and does not wait for the disk: nobody reads it
# before its last write, and a build stopped midway is thrown away
NO_JOURNAL = "PRAGMA journal_mode=OFF"
NO_WAIT_FOR_THE_DISK = "PRAGMA synchronous=OFF"
FULL_TEXT_COMPILED_IN = "SELECT sqlite_compileoption_used('ENABLE_FTS5')"
BEGIN_READ = "BEGIN"
BEGIN_WRITE = "BEGIN IMMEDIATE"
# The columns a chunk is read from, which `chunk_of` reads by name, named by their table as
# a search joins the chunks with their vectors
CHUNK_COLUMNS = (
    "chunks.chunk_id, chunks.path, chunks.language, chunks.kind, chunks.name, chunks.parent, "
    "chunks.line_start, chunks.line_end, chunks.content"
)
INSERT_CHUNK = (
    "INSERT INTO chunks (row_id, chunk_id, path, language, kind, name, folded_name, parent, "
    "line_start, line_end, content) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)
INSERT_WORDS = "INSERT INTO words (rowid, search_text) VALUES (?, ?)"
SELECT_CHUNK_IDS_OF_PATH = "SELECT chunk_id FROM chunks WHERE path = ?"
DELETE_WORDS_OF_PATH = "DELETE FROM words WHERE rowid IN (SELECT row_id FROM chunks WHERE path = ?)"
DELETE_CHUNKS_OF_PATH = "DELETE FROM chunks WHERE path = ?"
DELETE_VECTORS_OF_CHUNK = "DELETE FROM vectors WHERE chunk_id = ?"
DELETE_RECORD = "DELETE FROM files WHERE path = ?"
UPSERT_RECORD = "INSERT OR REPLACE INTO files (path, mtime_ns, size_bytes) VALUES (?, ?, ?)"
# A model's vector of a chunk the index holds; none for a chunk it no longer holds, nor in
# place of one stored already, which another session may have added: of the same text, it
# is the same vector
INSERT_VECTOR = (
    "INSERT INTO vectors (model, chunk_id, vector) "
    "SELECT ?, chunk_id, ? FROM chunks WHERE chunk_id = ? ON CONFLICT DO NOTHING"
)
STORED_WIDTH = "SELECT length(vector) FROM vectors WHERE model = ? LIMIT 1"
DELETE_VECTORS_OF_MODEL = "DELETE FROM vectors WHERE model = ?"
CHUNKS_WITHOUT_VECTORS = (
    f"SELECT {CHUNK_COLUMNS} FROM chunks WHERE NOT EXISTS ("  # nosec B608: constants
    "SELECT 1 FROM vectors WHERE vectors.chunk_id = chunks.chunk_id AND vectors.model = ?) "
    "ORDER BY chunks.row_id"
)
# The last row id the chunks table ever gave, which AUTOINCREMENT never gives again
LAST_ROW_ID = "SELECT coalesce(max(seq), 0) FROM sqlite_sequence WHERE name = 'chunks'"
LAST_VECTOR_ROW_ID = "SELECT coalesce(max(seq), 0) FROM sqlite_sequence WHERE name = 'vectors'"
COUNT_VECTORS = "SELECT count(*) FROM vectors WHERE model = ?"
VECTORS_STATE = f"SELECT ({SELECT_DATABASE_ID}), ({LAST_VECTOR_ROW_ID}), ({COUNT_VECTORS})"  # nosec B608
SELECT_VECTOR_ROW_IDS = "SELECT row_id FROM vectors WHERE model = ?"
VECTORS_AFTER = "SELECT row_id, vector FROM vectors WHERE model = ? AND row_id > ? ORDER BY row_id"
# The chunks whose words best match a question, best first, when the scope keeps every chunk
BEST_MATCHES = "SELECT rowid FROM words WHERE words MATCH ? ORDER BY bm25(words) LIMIT ?"
SELECT_RECORDS = "SELECT path, mtime_ns, size_bytes FROM files"
COUNT_RECORDS = "SELECT count(*) FROM files"
COUNT_CHUNKS = "SELECT count(*) FROM chunks"
# Statements a call completes with its list of placeholders or its scope's condition, each
# named in braces; the values they take are bound
SELECT_CHUNKS = f"SELECT row_id, {CHUNK_COLUMNS} FROM chunks WHERE row_id IN ({{placeholders}})"  # nosec B608
SELECT_NAMED = (
    f"SELECT {CHUNK_COLUMNS} FROM chunks WHERE folded_name = ? AND {{condition}} "  # nosec B608
    "ORDER BY path, line_start"
)
# The chunks of a model's vectors, by the vectors' row ids
SELECT_NEAREST = (
    f"SELECT vectors.row_id AS vector_row_id, {CHUNK_COLUMNS} FROM vectors "  # nosec B608
    "JOIN chunks ON chunks.chunk_id = vectors.chunk_id WHERE vectors.row_id IN ({placeholders})"
)
# Chunks by their row ids, with the row id of their vector of a model, None where they have none
SELECT_MATCHED = (
    f"SELECT chunks.row_id AS row_id, vectors.row_id AS vector_row_id, {CHUNK_COLUMNS} "  # nosec B608
    "FROM chunks LEFT JOIN vectors ON vectors.chunk_id = chunks.chunk_id AND vectors.model = ? "
    "WHERE chunks.row_id IN ({placeholders})"
)
# The row ids of a model's vectors whose chunks a scope keeps
SELECT_VECTOR_ROW_IDS_IN_SCOPE = (
    "SELECT vectors.row_id FROM vectors JOIN chunks ON chunks.chunk_id = vectors.chunk_id "
    "WHERE vectors.model = ? AND {condition}"
)
# The chunks of a scope whose words best match a question: the scope's chunks listed once, from
# the index of their narrow columns; the unary plus keeps SQLite from handing the list to FTS5
# as one lookup a row id, a full-text query each (21 s against 24 ms at 88k chunks)
SCOPED_BEST_MATCHES = (
    "SELECT rowid FROM words WHERE words MATCH ? "
    "AND +rowid IN (SELECT row_id FROM chunks WHERE {condition}) "
    "ORDER BY bm25(words) LIMIT ?"
)
KINDS_CONDITION = "kind IN ({placeholders})"
LANGUAGES_CONDITION = "language IN ({placeholders})"
# A glob is matched once a file, against the files' records, not once a chunk
GLOB_CONDITION = "path IN (SELECT path FROM files WHERE path REGEXP ?)"
EXCLUDED_GLOB_CONDITION = "path NOT IN (SELECT path FROM files WHERE path REGEXP ?)"
DIRECTORY_CONDITION = "substr(path, 1, ?) = ?"
# What a search finds when the index holds no chunks yet
NO_CANDIDATES = SearchCandidates(by_vector=[], by_keyword=[])
# A sort that keeps equals in the order they came, so the earlier row of two as near leads
STABLE_SORT = "stable"


class SqliteStoreError(RuntimeError):
    """SQLite could not read or write the index: its file locked past the timeout, the disk
    full, the database damaged, or a Python whose SQLite lacks FTS5."""


class ScopeCondition(BaseModel):
    """A search's scope as a condition on the chunks table, and the values it binds."""

    model_config = ConfigDict(frozen=True)

    sql: str
    values: list[str | int]

    @property
    def keeps_every_chunk(self) -> bool:
        return self.sql == NO_CONDITION


class VectorsState(BaseModel):
    """Where a model's vectors stand: the id of the database they are in, the digest of the
    model, the last row id the vectors table ever gave, and how many the model has. A write
    that changes them gives new row ids or deletes rows alone, so a later state of one
    database has a later last row id, or the same one and fewer vectors."""

    model_config = ConfigDict(frozen=True)

    database_id: str
    model: str
    last_row_id: int
    vectors: int

    def precedes(self, later: VectorsState) -> bool:
        """Whether this state came before the later one of the same model in the same
        database, whose vectors are then this one's less those deleted since, and those past
        this one's last row id."""
        return (
            self.database_id == later.database_id
            and self.model == later.model
            and (
                self.last_row_id < later.last_row_id
                or (self.last_row_id == later.last_row_id and self.vectors > later.vectors)
            )
        )


class RowSimilarities(BaseModel):
    """How close each vector of a model is to a query's, by the vectors' row ids in their
    ascending order."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    row_ids: NDArray[numpy.int64]
    similarities: NDArray[numpy.float32]

    def of(self, row_id: int | None) -> float | None:
        """The similarity of the vector of this row id; None for a chunk without one."""
        if row_id is None:
            return None
        return float(self.similarities[numpy.searchsorted(self.row_ids, row_id)])

    def among(self, row_ids: NDArray[numpy.int64]) -> RowSimilarities:
        """Those of the vectors of these row ids alone."""
        kept = numpy.isin(self.row_ids, row_ids)
        return RowSimilarities(row_ids=self.row_ids[kept], similarities=self.similarities[kept])

    def nearest(self, limit: int) -> list[int]:
        """The row ids of the `limit` vectors nearest the query, nearest first; of two as
        near, the earlier row first. Only the vectors at least as near as the `limit`-th are
        sorted, which a partition finds without sorting them all."""
        if limit < len(self.similarities):
            cut = len(self.similarities) - limit
            least_kept = numpy.partition(self.similarities, cut)[cut]
            positions = numpy.flatnonzero(self.similarities >= least_kept)
        else:
            positions = numpy.arange(len(self.similarities))
        order = numpy.argsort(-self.similarities[positions], kind=STABLE_SORT)[:limit]
        return self.row_ids[positions[order]].tolist()


# How close the vectors of a model that has none are to a query
NO_SIMILARITIES = RowSimilarities(
    row_ids=numpy.empty(0, dtype=ROW_ID_TYPE), similarities=numpy.empty(0, dtype=VECTOR_TYPE)
)


class HeldVectors(BaseModel):
    """A model's vectors in one state, held by the store between searches: each scaled to
    length one, a row each, in the order of their row ids."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    state: VectorsState
    row_ids: NDArray[numpy.int64]
    unit_vectors: VectorMatrix

    def similarities_to(self, query_vector: Vector) -> RowSimilarities:
        """How close each vector is to the query's."""
        return RowSimilarities(
            row_ids=self.row_ids,
            similarities=unit_similarities(self.unit_vectors, unit_vector(query_vector)),
        )


class SqliteIndexStore(DirectoryStore):
    """One project's index of one identity, in a SQLite database in the index's directory.

    Nothing touches the disk until it is first used, and only a write makes the database.
    """

    # The schema's version is bumped by any change to the tables or to what one is made from,
    # its full-text index among them (2: a row id never given twice, which the held vectors
    # rest on; 3: the database's id; 4: vectors in a table of their own, by model, and the
    # chunks' names case-folded)
    LAYOUT: ClassVar[IndexLayout] = IndexLayout(store_kind=StoreKind.sqlite, schema_version=4)

    def __init__(self, directory: IndexDirectory) -> None:
        super().__init__(directory)
        # The vectors of the latest state a search read, kept for the next; searches run in
        # worker threads, so they take and replace them under a lock
        self._holding = threading.Lock()
        self._held: HeldVectors | None = None
        # Whether the run this store holds builds the database whole, and whether that build
        # has made its database yet
        self._building_whole = False
        self._whole_build_made = False

    @override
    def try_hold_for_run(self) -> bool:
        held = super().try_hold_for_run()
        # A run that finds no database builds it whole, out of every reader's sight
        self._building_whole = held and not self._database().is_file()
        self._whole_build_made = False
        return held

    @override
    def release_run(self) -> None:
        # A build stopped before its last write leaves its file for the next build to delete
        self._building_whole = False
        super().release_run()

    @override
    def indexed_files(self) -> IndexedFiles:
        with self._snapshot() as connection:
            rows = [] if connection is None else connection.execute(SELECT_RECORDS).fetchall()
        records = [
            SourceFile(path=Path(path), mtime_ns=mtime_ns, size_bytes=size_bytes)
            for path, mtime_ns, size_bytes in rows
        ]
        # A write is one transaction: no chunk is ever left without its record
        return IndexedFiles(records=records, unrecorded_paths=[])

    @override
    def counts(self, model: VectorIdentity) -> IndexCounts:
        with self._snapshot() as connection:
            if connection is None:
                return IndexCounts(files=0, chunks=0, chunks_with_vectors=0)
            files = connection.execute(COUNT_RECORDS).fetchone()[0]
            chunks = connection.execute(COUNT_CHUNKS).fetchone()[0]
            vectors = connection.execute(COUNT_VECTORS, [identity_digest(model)]).fetchone()[0]
        return IndexCounts(files=files, chunks=chunks, chunks_with_vectors=vectors)

    @override
    def write(self, update: IndexUpdate) -> None:
        # A whole build's database is in place, for a manifest to name, after its last write
        records_manifest = not self._building_whole or update.last_of_run
        with self._directory.writing(records_manifest=records_manifest):
            try:
                if self._building_whole:
                    self._write_whole_build(update)
                    return
                self._make_database()
                with closing(self._connect()) as connection, connection:
                    # The connection commits the transaction, or rolls it back on any error
                    connection.execute(BEGIN_WRITE)
                    replace_chunks(connection, update)
                    replace_records(connection, update)
            except sqlite3.Error as error:
                message = f"SQLite could not write the index in {self._directory.path}: {error}"
                raise SqliteStoreError(message) from error
        log.debug(
            "index_written",
            files=len(update.files),
            chunks=len(update.chunks),
            removed=len(update.removed_paths),
        )

    @override
    def chunks_without_vectors(self, model: VectorIdentity) -> list[Chunk]:
        with self._snapshot() as connection:
            rows = (
                []
                if connection is None
                else connection.execute(CHUNKS_WITHOUT_VECTORS, [identity_digest(model)])
            )
            return [chunk_of(row) for row in rows]

    @override
    def add_vectors(
        self, model: VectorIdentity, chunk_ids: list[str], vectors: VectorMatrix
    ) -> None:
        if len(chunk_ids) != len(vectors):
            message = f"{len(chunk_ids)} chunk ids but {len(vectors)} vectors"
            raise ValueError(message)
        if not chunk_ids:
            return
        model_key = identity_digest(model)
        with self._written_transaction() as connection:
            if connection is None:
                # No database holds no chunk to add a vector to
                return
            self._fit_width(connection, model_key, vectors.shape[VECTOR_LENGTH_AXIS])
            connection.executemany(
                INSERT_VECTOR,
                [
                    (model_key, vector.astype(VECTOR_TYPE).tobytes(), chunk_id)
                    for chunk_id, vector in zip(chunk_ids, vectors, strict=True)
                ],
            )
        log.debug("vectors_added", model=model.model_name, chunks=len(chunk_ids))

    @override
    def remove_vectors(self, model: VectorIdentity) -> None:
        with self._written_transaction() as connection:
            if connection is not None:
                connection.execute(DELETE_VECTORS_OF_MODEL, [identity_digest(model)])

    @override
    def search_words(self, query: SearchQuery, limit: int) -> list[Chunk]:
        condition = scope_condition(query.scope)
        with self._snapshot() as connection:
            if connection is None:
                return []
            row_ids = best_matches(connection, query.text, condition, limit)
            rows_by_id = {
                row["row_id"]: row
                for row in connection.execute(
                    SELECT_CHUNKS.format(placeholders=placeholders(row_ids)), row_ids
                )
            }
        return [chunk_of(rows_by_id[row_id]) for row_id in row_ids]

    @override
    def chunks_named(self, folded_name: str, scope: SearchScope) -> list[Chunk]:
        condition = scope_condition(scope)
        with self._snapshot() as connection:
            if connection is None:
                return []
            rows = connection.execute(
                SELECT_NAMED.format(condition=condition.sql), [folded_name, *condition.values]
            )
            return [chunk_of(row) for row in rows]

    @override
    def search(
        self, model: VectorIdentity, query_vector: Vector, query: SearchQuery, per_channel: int
    ) -> SearchCandidates:
        condition = scope_condition(query.scope)
        model_key = identity_digest(model)
        with self._snapshot() as connection:
            if connection is None:
                return NO_CANDIDATES
            state = vectors_state(connection, model_key)
            similarities = (
                NO_SIMILARITIES
                if state.vectors == NO_VECTORS
                else self._held_vectors(connection, state).similarities_to(query_vector)
            )
            return SearchCandidates(
                by_vector=nearest_candidates(
                    connection, model_key, similarities, condition, per_channel
                ),
                by_keyword=matched_candidates(
                    connection,
                    model_key,
                    best_matches(connection, query.text, condition, per_channel),
                    similarities,
                ),
            )

    def _held_vectors(self, connection: sqlite3.Connection, state: VectorsState) -> HeldVectors:
        """A model's vectors in the state the connection reads, which holds some: those the
        store holds, refreshed by rows when they are of an earlier state, read whole when it
        holds none, or another database's or model's, or a later state's, as another
        thread's search may have left. Kept for the next search unless the store holds a
        later state."""
        with self._holding:
            held = self._held
            if held is not None and held.state == state:
                return held
            if held is not None and held.state.precedes(state):
                current = refreshed_vectors(connection, held, state)
            else:
                current = vectors_read_whole(connection, state)
            if held is None or not state.precedes(held.state):
                self._held = current
            return current

    def _fit_width(self, connection: sqlite3.Connection, model_key: str, width: int) -> None:
        """Drop the model's vectors when they are of another width than this write's, as a
        model replaced under a name nothing pins gives: in the write's transaction, so its
        chunks are embedded again by the next search by meaning."""
        stored = connection.execute(STORED_WIDTH, [model_key]).fetchone()
        if stored is None or stored[0] == width * VECTOR_COMPONENT_BYTES:
            return
        log.warning("model_vectors_replaced", directory=str(self._directory.path), width=width)
        connection.execute(DELETE_VECTORS_OF_MODEL, [model_key])

    @contextmanager
    def _written_transaction(self) -> Generator[sqlite3.Connection | None]:
        """A write transaction on the database under the index's write lock, committed when
        it ends without an error; None where no database is made, which only a write of
        words makes."""
        if not self._database().is_file():
            yield None
            return
        with self._directory.writing(records_manifest=True):
            try:
                with closing(self._connect()) as connection, connection:
                    connection.execute(BEGIN_WRITE)
                    yield connection
            except sqlite3.Error as error:
                message = f"SQLite could not write the index in {self._directory.path}: {error}"
                raise SqliteStoreError(message) from error

    def _write_whole_build(self, update: IndexUpdate) -> None:
        """One write of a whole build, into its own database, made by its first write; the
        last switches it to WAL and renames it into place. Called under the write lock."""
        unfinished = self._unfinished_database()
        if not self._whole_build_made:
            unfinished.unlink(missing_ok=True)
            with closing(sqlite3.connect(unfinished, isolation_level=None)) as connection:
                make_tables(connection)
            self._whole_build_made = True
        with closing(sqlite3.connect(unfinished, isolation_level=None)) as connection:
            connection.execute(NO_JOURNAL)
            connection.execute(NO_WAIT_FOR_THE_DISK)
            with connection:
                connection.execute(BEGIN_WRITE)
                insert_chunks(connection, update)
                replace_records(connection, update)
            if update.last_of_run:
                connection.execute(WRITE_AHEAD_LOG)
        if update.last_of_run:
            unfinished.replace(self._database())
            self._building_whole = False

    def _make_database(self) -> None:
        """Make the database, its tables and its full-text index, if there is none: under a
        temporary name, renamed into place whole, so a reader finds no database or one it
        can read. Called under the write lock."""
        database = self._database()
        if database.is_file():
            return
        unfinished = self._unfinished_database()
        unfinished.unlink(missing_ok=True)
        with closing(sqlite3.connect(unfinished, isolation_level=None)) as connection:
            connection.execute(WRITE_AHEAD_LOG)
            make_tables(connection)
        unfinished.replace(database)

    def _unfinished_database(self) -> Path:
        """Where a database is made before it is renamed into place."""
        return self._database().with_name(DATABASE_FILE + UNFINISHED_SUFFIX)

    @contextmanager
    def _snapshot(self) -> Generator[sqlite3.Connection | None]:
        """A connection reading one state of the index inside a transaction of its own; None
        while the database is not made yet. The index is marked as used."""
        self._directory.note_use()
        if not self._database().is_file():
            yield None
            return
        try:
            with closing(self._connect()) as connection:
                connection.execute(BEGIN_READ)
                try:
                    yield connection
                finally:
                    connection.rollback()
        except sqlite3.Error as error:
            message = f"SQLite could not read the index in {self._directory.path}: {error}"
            raise SqliteStoreError(message) from error

    def _database(self) -> Path:
        return self._directory.path / DATABASE_FILE

    def _connect(self) -> sqlite3.Connection:
        """A connection of this call's own, waiting for SQLite's lock as long as for the
        index's, its transactions begun and ended by the store."""
        connection = sqlite3.connect(
            self._database().absolute().as_uri() + EXISTING_DATABASE,
            timeout=self._directory.lock_timeout_seconds,
            isolation_level=None,
            uri=True,
        )
        connection.create_function(
            REGEXP_FUNCTION, REGEXP_ARGUMENTS, matches_pattern, deterministic=True
        )
        # Rows read by their columns' names
        connection.row_factory = sqlite3.Row
        return connection


def make_tables(connection: sqlite3.Connection) -> None:
    """Make a new database's tables, its full-text index and its id; refused where this
    Python's SQLite lacks FTS5."""
    if not connection.execute(FULL_TEXT_COMPILED_IN).fetchone()[0]:
        message = "this Python's SQLite lacks FTS5, which the index's words need"
        raise SqliteStoreError(message)
    for statement in CREATE_TABLES:
        connection.execute(statement)
    connection.execute(INSERT_DATABASE_ID, [uuid.uuid4().hex])


def replace_chunks(connection: sqlite3.Connection, update: IndexUpdate) -> None:
    """Put the write's chunks, with their words, in place of what the index held of the files
    written and removed: the vectors of the chunks it keeps by id stay, those of the chunks it
    drops go. A chunk's id digests its path, so no other file holds one of the write's ids."""
    drop_replaced(connection, update)
    insert_chunks(connection, update)


def drop_replaced(connection: sqlite3.Connection, update: IndexUpdate) -> None:
    """Drop what the index holds of the files a write replaces and removes, and the vectors of
    the chunks it does not keep by id."""
    replaced_paths = [
        *(file.path.as_posix() for file in update.files),
        *(path.as_posix() for path in update.removed_paths),
    ]
    kept_ids = {chunk.chunk_id for chunk in update.chunks}
    dropped_ids = [
        chunk_id
        for path in replaced_paths
        for (chunk_id,) in connection.execute(SELECT_CHUNK_IDS_OF_PATH, [path])
        if chunk_id not in kept_ids
    ]
    connection.executemany(DELETE_VECTORS_OF_CHUNK, [(chunk_id,) for chunk_id in dropped_ids])
    connection.executemany(DELETE_WORDS_OF_PATH, [(path,) for path in replaced_paths])
    connection.executemany(DELETE_CHUNKS_OF_PATH, [(path,) for path in replaced_paths])


def insert_chunks(connection: sqlite3.Connection, update: IndexUpdate) -> None:
    """Store a write's chunks with their words, under row ids past every one the table gave."""
    first_row_id = connection.execute(LAST_ROW_ID).fetchone()[0] + NEXT_ROW_ID_STEP
    row_ids = range(first_row_id, first_row_id + len(update.chunks))
    connection.executemany(
        INSERT_CHUNK,
        [
            (
                row_id,
                chunk.chunk_id,
                chunk.path.as_posix(),
                chunk.language,
                chunk.kind,
                chunk.name,
                chunk.name.casefold(),
                chunk.parent,
                chunk.line_start,
                chunk.line_end,
                chunk.content,
            )
            for row_id, chunk in zip(row_ids, update.chunks, strict=True)
        ],
    )
    connection.executemany(
        INSERT_WORDS,
        [
            (row_id, search_text(chunk))
            for row_id, chunk in zip(row_ids, update.chunks, strict=True)
        ],
    )


def replace_records(connection: sqlite3.Connection, update: IndexUpdate) -> None:
    """Drop the records of the files removed, and record the files written as the scan found
    them."""
    connection.executemany(DELETE_RECORD, [(path.as_posix(),) for path in update.removed_paths])
    connection.executemany(
        UPSERT_RECORD,
        [(file.path.as_posix(), file.mtime_ns, file.size_bytes) for file in update.files],
    )


def vectors_state(connection: sqlite3.Connection, model_key: str) -> VectorsState:
    """Where a model's vectors stand in the state the connection reads."""
    database_id, last_row_id, vectors = connection.execute(VECTORS_STATE, [model_key]).fetchone()
    return VectorsState(
        database_id=database_id, model=model_key, last_row_id=last_row_id, vectors=vectors
    )


def vectors_read_whole(connection: sqlite3.Connection, state: VectorsState) -> HeldVectors:
    """Every vector of a model, as of the state the connection reads, which holds some."""
    rows = connection.execute(VECTORS_AFTER, [state.model, BEFORE_EVERY_ROW]).fetchall()
    width = len(rows[0]["vector"]) // VECTOR_COMPONENT_BYTES
    return HeldVectors(
        state=state,
        row_ids=numpy.array([row["row_id"] for row in rows], dtype=ROW_ID_TYPE),
        unit_vectors=unit_rows(matrix_of([row["vector"] for row in rows], width)),
    )


def refreshed_vectors(
    connection: sqlite3.Connection, held: HeldVectors, state: VectorsState
) -> HeldVectors:
    """A model's vectors held of an earlier state brought to the one the connection reads:
    those deleted since dropped, those added since read. Read whole when none of the held is
    left, as when vectors of another width replaced the model's."""
    present = numpy.fromiter(
        (row_id for (row_id,) in connection.execute(SELECT_VECTOR_ROW_IDS, [state.model])),
        dtype=ROW_ID_TYPE,
    )
    kept = numpy.isin(held.row_ids, present)
    if not kept.any():
        return vectors_read_whole(connection, state)
    rows = connection.execute(VECTORS_AFTER, [state.model, held.state.last_row_id]).fetchall()
    width = held.unit_vectors.shape[VECTOR_LENGTH_AXIS]
    added = unit_rows(matrix_of([row["vector"] for row in rows], width))
    return HeldVectors(
        state=state,
        row_ids=numpy.concatenate(
            [held.row_ids[kept], numpy.array([row["row_id"] for row in rows], dtype=ROW_ID_TYPE)]
        ),
        unit_vectors=numpy.concatenate([held.unit_vectors[kept], added]),
    )


def matrix_of(blobs: list[bytes], width: int) -> VectorMatrix:
    """Stored vectors of one width as a matrix, a row each, read as one buffer."""
    return numpy.frombuffer(b"".join(blobs), dtype=VECTOR_TYPE).reshape(len(blobs), width)


def nearest_candidates(
    connection: sqlite3.Connection,
    model_key: str,
    similarities: RowSimilarities,
    condition: ScopeCondition,
    limit: int,
) -> list[Candidate]:
    """The chunks of the scope whose vectors of the model are nearest the query's, nearest
    first."""
    in_scope = (
        similarities
        if condition.keeps_every_chunk
        else similarities.among(vector_row_ids_in(connection, model_key, condition))
    )
    vector_row_ids = in_scope.nearest(limit)
    rows_by_vector = {
        row["vector_row_id"]: row
        for row in connection.execute(
            SELECT_NEAREST.format(placeholders=placeholders(vector_row_ids)), vector_row_ids
        )
    }
    return [
        Candidate(
            chunk=chunk_of(rows_by_vector[vector_row_id]),
            similarity=similarities.of(vector_row_id),
        )
        for vector_row_id in vector_row_ids
    ]


def vector_row_ids_in(
    connection: sqlite3.Connection, model_key: str, condition: ScopeCondition
) -> NDArray[numpy.int64]:
    """The row ids of a model's vectors whose chunks a scope keeps."""
    rows = connection.execute(
        SELECT_VECTOR_ROW_IDS_IN_SCOPE.format(condition=condition.sql),
        [model_key, *condition.values],
    )
    return numpy.fromiter((row_id for (row_id,) in rows), dtype=ROW_ID_TYPE)


def matched_candidates(
    connection: sqlite3.Connection,
    model_key: str,
    row_ids: list[int],
    similarities: RowSimilarities,
) -> list[Candidate]:
    """The chunks of these row ids, which the keyword channel ranked, in their order, each
    with its vector's similarity where it has one. The channel ranks row ids alone and reads
    the columns of those it keeps, as reading them for every chunk a question's words match
    cost a search at 88k chunks 140 ms."""
    rows_by_id = {
        row["row_id"]: row
        for row in connection.execute(
            SELECT_MATCHED.format(placeholders=placeholders(row_ids)), [model_key, *row_ids]
        )
    }
    return [
        Candidate(
            chunk=chunk_of(rows_by_id[row_id]),
            similarity=similarities.of(rows_by_id[row_id]["vector_row_id"]),
        )
        for row_id in row_ids
    ]


def best_matches(
    connection: sqlite3.Connection, question: str, condition: ScopeCondition, limit: int
) -> list[int]:
    """The row ids of the scope's chunks whose words best match the question's, best first;
    none for a question of no word."""
    tokens = QUERY_TOKEN.findall(question)
    if not tokens:
        return []
    expression = EITHER_WORD.join(QUOTE + token + QUOTE for token in tokens)
    ranked = (
        connection.execute(BEST_MATCHES, [expression, limit])
        if condition.keeps_every_chunk
        else connection.execute(
            SCOPED_BEST_MATCHES.format(condition=condition.sql),
            [expression, *condition.values, limit],
        )
    )
    return [row_id for (row_id,) in ranked]


def chunk_of(row: sqlite3.Row) -> Chunk:
    """A row of CHUNK_COLUMNS as the chunk it holds."""
    return Chunk(
        chunk_id=row["chunk_id"],
        path=Path(row["path"]),
        language=Language(row["language"]),
        kind=ChunkKind(row["kind"]),
        name=row["name"],
        parent=row["parent"],
        line_start=row["line_start"],
        line_end=row["line_end"],
        content=row["content"],
    )


def scope_condition(scope: SearchScope) -> ScopeCondition:
    """The condition that keeps a search's channels to its scope, applied before either cuts."""
    clauses: list[str] = []
    values: list[str | int] = []
    if scope.kinds:
        clauses.append(KINDS_CONDITION.format(placeholders=placeholders(scope.kinds)))
        values.extend(scope.kinds)
    if scope.languages:
        clauses.append(LANGUAGES_CONDITION.format(placeholders=placeholders(scope.languages)))
        values.extend(scope.languages)
    if scope.path_glob is not None:
        clauses.append(GLOB_CONDITION)
        values.append(glob_pattern(scope.path_glob, scope.directory))
    if scope.exclude_glob is not None:
        clauses.append(EXCLUDED_GLOB_CONDITION)
        values.append(glob_pattern(scope.exclude_glob, scope.directory))
    if scope.directory is not None:
        prefix = scope.directory.as_posix() + PATH_SEPARATOR
        clauses.append(DIRECTORY_CONDITION)
        values.extend([len(prefix), prefix])
    return ScopeCondition(sql=CONDITION_SEPARATOR.join(clauses) or NO_CONDITION, values=values)


def matches_pattern(pattern: str, text: str) -> bool:
    """SQLite's REGEXP: whether a regular expression matches the text, Python's reading of
    it, which `glob_pattern` writes for."""
    return re.search(pattern, text) is not None


def placeholders(values: Sequence[object]) -> str:
    """As many of SQL's placeholders as there are values, for a statement's list."""
    return VALUE_SEPARATOR.join(VALUE_PLACEHOLDER for _ in values)
