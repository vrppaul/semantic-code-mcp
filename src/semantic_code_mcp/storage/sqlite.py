"""SQLite storage of one project's index: the chunks with their vectors, their words in a
full-text index, the record of each file they came from, and the manifest of what built them,
in one database file in a directory of its own (`index_directories.py` holds where it lies,
its lock, its manifest and the sweep of the project's unused indexes).

A write is one transaction, under the index's write lock beside SQLite's own: the chunks of
the files written take the place of those files' and the removed files' chunks, their words
with them, then the files' records; the manifest follows the commit. So a reader finds a
file's old chunks or its new ones with their records, a write stopped midway leaves the index
as it was, and no chunk is ever left without its record. The full-text index is FTS5's, kept
exact row by row, so nothing is folded at a run's end; it holds no copy of the text
(`content=''`), only what finds it.

Each call opens a connection of its own, as the store's calls come from worker threads, and a
search reads both its channels inside one read transaction: one state of the index, while
another process writes beside it (WAL). A question is matched as words, any of which may
match: each token as FTS5's tokenizer splits text, quoted, the tokens joined by OR, so an
agent's question is never read as FTS5's own syntax (ADR 007, D7).

The vector channel reads the vectors of the search's scope at each search: no matrix is held
between calls yet (ADR 009, D8).
"""

import re
import sqlite3
from collections.abc import Generator, Sequence
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import override

import numpy
import structlog
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

# Bumped by any change to the tables or to what one is made from, its full-text index among
# them; each version keeps an index of its own
SCHEMA_VERSION = 1
# What follows the name of this store's index directories, so they never meet another
# store's: LanceDB's keep the names they had before there were two
DIRECTORY_SUFFIX = "-sqlite"
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
# The function SQLite's REGEXP operator calls, with its pattern and the text, in that order
REGEXP_FUNCTION = "regexp"
REGEXP_ARGUMENTS = 2
# How many values one statement's list takes, far below SQLite's limit of variables
VALUES_PER_STATEMENT = 500
# Statements are built of this module's constants and placeholders alone; every value a
# search or a write brings is bound, never written into them
# What a scope that leaves nothing out filters by
NO_CONDITION = "1"
CONDITION_SEPARATOR = " AND "
VALUE_SEPARATOR = ", "
VALUE_PLACEHOLDER = "?"
# How far a new chunk's row id lies past the last one the table holds
NEXT_ROW_ID_STEP = 1

CREATE_CHUNKS = (
    "CREATE TABLE chunks ("
    "row_id INTEGER PRIMARY KEY, chunk_id TEXT NOT NULL UNIQUE, path TEXT NOT NULL, "
    "language TEXT NOT NULL, kind TEXT NOT NULL, name TEXT NOT NULL, parent TEXT, "
    "line_start INTEGER NOT NULL, line_end INTEGER NOT NULL, content TEXT NOT NULL, "
    "vector BLOB NOT NULL)"
)
CREATE_CHUNKS_BY_PATH = "CREATE INDEX chunks_by_path ON chunks (path)"
CREATE_FILES = (
    "CREATE TABLE files ("
    "path TEXT PRIMARY KEY, mtime_ns INTEGER NOT NULL, size_bytes INTEGER NOT NULL)"
)
CREATE_WORDS = (
    "CREATE VIRTUAL TABLE words USING fts5("
    f"search_text, tokenize='{FULL_TEXT_TOKENIZER}', content='', contentless_delete=1)"
)
CREATE_TABLES = (CREATE_CHUNKS, CREATE_CHUNKS_BY_PATH, CREATE_FILES, CREATE_WORDS)
WRITE_AHEAD_LOG = "PRAGMA journal_mode=WAL"
FULL_TEXT_COMPILED_IN = "SELECT sqlite_compileoption_used('ENABLE_FTS5')"
BEGIN_READ = "BEGIN"
BEGIN_WRITE = "BEGIN IMMEDIATE"
# The columns of a chunk a search returns, which `candidate_of` reads by name, and its vector
CANDIDATE_COLUMNS = (
    "chunk_id, path, language, kind, name, parent, line_start, line_end, content, vector"
)
INSERT_CHUNK = (
    "INSERT INTO chunks (row_id, chunk_id, path, language, kind, name, parent, line_start, "
    "line_end, content, vector) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)
INSERT_WORDS = "INSERT INTO words (rowid, search_text) VALUES (?, ?)"
DELETE_WORDS_OF_PATH = "DELETE FROM words WHERE rowid IN (SELECT row_id FROM chunks WHERE path = ?)"
DELETE_CHUNKS_OF_PATH = "DELETE FROM chunks WHERE path = ?"
DELETE_WORDS_OF_CHUNK = (
    "DELETE FROM words WHERE rowid IN (SELECT row_id FROM chunks WHERE chunk_id = ?)"
)
DELETE_CHUNK = "DELETE FROM chunks WHERE chunk_id = ?"
DELETE_RECORD = "DELETE FROM files WHERE path = ?"
UPSERT_RECORD = "INSERT OR REPLACE INTO files (path, mtime_ns, size_bytes) VALUES (?, ?, ?)"
# FTS5's command that empties a full-text index holding no copy of its text
DELETE_EVERY_WORD = "INSERT INTO words (words) VALUES ('delete-all')"
DELETE_EVERY_CHUNK = "DELETE FROM chunks"
DELETE_EVERY_RECORD = "DELETE FROM files"
STORED_WIDTH = "SELECT length(vector) FROM chunks LIMIT 1"
LAST_ROW_ID = "SELECT coalesce(max(row_id), 0) FROM chunks"
SELECT_RECORDS = "SELECT path, mtime_ns, size_bytes FROM files"
COUNT_RECORDS = "SELECT count(*) FROM files"
COUNT_CHUNKS = "SELECT count(*) FROM chunks"
# What a search finds when the index holds no chunks yet
NO_CANDIDATES = SearchCandidates(by_vector=[], by_keyword=[])


class SqliteStoreError(RuntimeError):
    """SQLite could not read or write the index: its file locked past the timeout, the disk
    full, the database damaged, or a Python whose SQLite lacks FTS5."""


class ScopeCondition(BaseModel):
    """A search's scope as a condition on the chunks table, and the values it binds."""

    model_config = ConfigDict(frozen=True)

    sql: str
    values: list[str | int]


class SqliteIndexStore(IndexStore):
    """One project's index of one identity, in a SQLite database of its own under the
    project's index directory.

    Nothing touches the disk until it is first used, and only a write makes the database.
    """

    def __init__(
        self, project_directory: Path, identity: IndexIdentity, *, lock_timeout_seconds: float
    ) -> None:
        self._identity = identity
        self._directory = IndexDirectory(
            project_directory,
            index_directory_name(identity, SCHEMA_VERSION) + DIRECTORY_SUFFIX,
            lock_timeout_seconds=lock_timeout_seconds,
        )
        self._lock_timeout_seconds = lock_timeout_seconds

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
        with self._snapshot() as connection:
            rows = [] if connection is None else connection.execute(SELECT_RECORDS).fetchall()
        records = [
            SourceFile(path=Path(path), mtime_ns=mtime_ns, size_bytes=size_bytes)
            for path, mtime_ns, size_bytes in rows
        ]
        # A write is one transaction: no chunk is ever left without its record
        return IndexedFiles(records=records, unrecorded_paths=[])

    @override
    def counts(self) -> IndexCounts:
        with self._snapshot() as connection:
            if connection is None:
                return IndexCounts(files=0, chunks=0)
            files = connection.execute(COUNT_RECORDS).fetchone()[0]
            chunks = connection.execute(COUNT_CHUNKS).fetchone()[0]
        return IndexCounts(files=files, chunks=chunks)

    @override
    def stored_vectors(self, chunk_ids: list[str]) -> dict[str, Vector]:
        with self._snapshot() as connection:
            if connection is None:
                return {}
            rows = [
                row
                for batch in batches(chunk_ids)
                for row in connection.execute(
                    f"SELECT chunk_id, vector FROM chunks "  # nosec B608: placeholders alone
                    f"WHERE chunk_id IN ({placeholders(batch)})",
                    batch,
                )
            ]
        return {chunk_id: vector_of(blob) for chunk_id, blob in rows}

    @override
    def write(self, update: IndexUpdate, vectors: VectorMatrix) -> None:
        if len(update.chunks) != len(vectors):
            message = f"{len(update.chunks)} chunks but {len(vectors)} vectors"
            raise ValueError(message)
        self._directory.prepare_for_writing()
        with self._directory.locked():
            self._directory.create()
            try:
                self._make_database()
                with closing(self._connect()) as connection, connection:
                    # The connection commits the transaction, or rolls it back on any error
                    connection.execute(BEGIN_WRITE)
                    if update.chunks:
                        self._fit_width(connection, vectors.shape[VECTOR_LENGTH_AXIS])
                    replace_chunks(connection, update, vectors)
                    replace_records(connection, update)
            except sqlite3.Error as error:
                message = f"SQLite could not write the index in {self.directory}: {error}"
                raise SqliteStoreError(message) from error
            self._directory.write_manifest(
                IndexManifest(
                    identity=self._identity,
                    schema_version=SCHEMA_VERSION,
                    indexed_at=datetime.now(UTC),
                )
            )
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
        condition = scope_condition(query.scope)
        with self._snapshot() as connection:
            if connection is None:
                return NO_CANDIDATES
            return SearchCandidates(
                by_vector=nearest_candidates(connection, query_vector, condition, per_channel),
                by_keyword=keyword_candidates(
                    connection, query.text, query_vector, condition, per_channel
                ),
            )

    def _fit_width(self, connection: sqlite3.Connection, width: int) -> None:
        """Start the index again when it holds vectors of another width than this write's,
        as a model replaced under a name nothing pins gives: in the write's transaction, so
        the files it does not hold are left unrecorded for the next run to index."""
        stored = connection.execute(STORED_WIDTH).fetchone()
        if stored is None or stored[0] == width * VECTOR_COMPONENT_BYTES:
            return
        log.warning("index_layout_replaced", directory=str(self.directory), width=width)
        for statement in (DELETE_EVERY_WORD, DELETE_EVERY_CHUNK, DELETE_EVERY_RECORD):
            connection.execute(statement)

    def _make_database(self) -> None:
        """Make the database, its tables and its full-text index, if there is none: under a
        temporary name, renamed into place whole, so a reader finds no database or one it
        can read. Called under the write lock."""
        database = self._directory.path / DATABASE_FILE
        if database.is_file():
            return
        unfinished = database.with_name(DATABASE_FILE + UNFINISHED_SUFFIX)
        unfinished.unlink(missing_ok=True)
        with closing(sqlite3.connect(unfinished, isolation_level=None)) as connection:
            if not connection.execute(FULL_TEXT_COMPILED_IN).fetchone()[0]:
                message = "this Python's SQLite lacks FTS5, which the index's words need"
                raise SqliteStoreError(message)
            connection.execute(WRITE_AHEAD_LOG)
            for statement in CREATE_TABLES:
                connection.execute(statement)
        unfinished.replace(database)

    @contextmanager
    def _snapshot(self) -> Generator[sqlite3.Connection | None]:
        """A connection reading one state of the index inside a transaction of its own; None
        while the database is not made yet. The index is marked as used."""
        self._directory.note_use()
        if not (self._directory.path / DATABASE_FILE).is_file():
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
            message = f"SQLite could not read the index in {self.directory}: {error}"
            raise SqliteStoreError(message) from error

    def _connect(self) -> sqlite3.Connection:
        """A connection of this call's own, waiting for SQLite's lock as long as for the
        index's, its transactions begun and ended by the store."""
        connection = sqlite3.connect(
            self._directory.path / DATABASE_FILE,
            timeout=self._lock_timeout_seconds,
            isolation_level=None,
        )
        connection.create_function(
            REGEXP_FUNCTION, REGEXP_ARGUMENTS, matches_pattern, deterministic=True
        )
        # Rows read by their columns' names
        connection.row_factory = sqlite3.Row
        return connection


def replace_chunks(
    connection: sqlite3.Connection, update: IndexUpdate, vectors: VectorMatrix
) -> None:
    """Put the write's chunks, with their words and vectors, in place of what the index held
    of the files written and removed, and of any chunk stored under one of their ids."""
    replaced_paths = [
        *(file.path.as_posix() for file in update.files),
        *(path.as_posix() for path in update.removed_paths),
    ]
    chunk_ids = [chunk.chunk_id for chunk in update.chunks]
    connection.executemany(DELETE_WORDS_OF_PATH, [(path,) for path in replaced_paths])
    connection.executemany(DELETE_CHUNKS_OF_PATH, [(path,) for path in replaced_paths])
    connection.executemany(DELETE_WORDS_OF_CHUNK, [(chunk_id,) for chunk_id in chunk_ids])
    connection.executemany(DELETE_CHUNK, [(chunk_id,) for chunk_id in chunk_ids])
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
                chunk.parent,
                chunk.line_start,
                chunk.line_end,
                chunk.content,
                vector.astype(VECTOR_TYPE).tobytes(),
            )
            for row_id, chunk, vector in zip(row_ids, update.chunks, vectors, strict=True)
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


def nearest_candidates(
    connection: sqlite3.Connection, query_vector: Vector, condition: ScopeCondition, limit: int
) -> list[Candidate]:
    """The chunks of the scope nearest the query's vector, nearest first."""
    rows = connection.execute(
        f"SELECT row_id, vector FROM chunks WHERE {condition.sql}",  # nosec B608: bound values
        condition.values,
    ).fetchall()
    if not rows:
        return []
    matrix = numpy.stack([vector_of(blob) for _, blob in rows])
    similarities = cosine_similarities(matrix, query_vector)
    nearest = numpy.argsort(-similarities, kind="stable")[:limit]
    row_ids = [rows[position][0] for position in nearest]
    chunks_by_row_id = {
        row["row_id"]: row
        for row in connection.execute(
            f"SELECT row_id, {CANDIDATE_COLUMNS} FROM chunks "  # nosec B608: placeholders alone
            f"WHERE row_id IN ({placeholders(row_ids)})",
            row_ids,
        )
    }
    return [candidate_of(chunks_by_row_id[row_id], query_vector) for row_id in row_ids]


def keyword_candidates(
    connection: sqlite3.Connection,
    question: str,
    query_vector: Vector,
    condition: ScopeCondition,
    limit: int,
) -> list[Candidate]:
    """The chunks of the scope whose words best match the question's, best first; none for a
    question of no word."""
    tokens = QUERY_TOKEN.findall(question)
    if not tokens:
        return []
    expression = EITHER_WORD.join(QUOTE + token + QUOTE for token in tokens)
    rows = connection.execute(
        f"SELECT {CANDIDATE_COLUMNS} FROM words JOIN chunks ON chunks.row_id = words.rowid "
        f"WHERE words MATCH ? AND {condition.sql} ORDER BY bm25(words) LIMIT ?",  # nosec B608: bound values
        [expression, *condition.values, limit],
    ).fetchall()
    return [candidate_of(row, query_vector) for row in rows]


def candidate_of(row: sqlite3.Row, query_vector: Vector) -> Candidate:
    """A row of CANDIDATE_COLUMNS as a chunk a search found, with its vector's similarity to
    the query's."""
    vector = vector_of(row["vector"])
    similarity = cosine_similarities(vector[numpy.newaxis, :], query_vector)[0]
    chunk = Chunk(
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
    return Candidate(chunk=chunk, similarity=float(similarity))


def scope_condition(scope: SearchScope) -> ScopeCondition:
    """The condition that keeps a search's channels to its scope, applied before either cuts."""
    clauses: list[str] = []
    values: list[str | int] = []
    if scope.kinds:
        clauses.append(f"kind IN ({placeholders(scope.kinds)})")
        values.extend(scope.kinds)
    if scope.languages:
        clauses.append(f"language IN ({placeholders(scope.languages)})")
        values.extend(scope.languages)
    if scope.path_glob is not None:
        clauses.append("path REGEXP ?")
        values.append(glob_pattern(scope.path_glob, scope.directory))
    if scope.exclude_glob is not None:
        clauses.append("NOT (path REGEXP ?)")
        values.append(glob_pattern(scope.exclude_glob, scope.directory))
    if scope.directory is not None:
        prefix = scope.directory.as_posix() + PATH_SEPARATOR
        clauses.append("substr(path, 1, ?) = ?")
        values.extend([len(prefix), prefix])
    return ScopeCondition(sql=CONDITION_SEPARATOR.join(clauses) or NO_CONDITION, values=values)


def matches_pattern(pattern: str, text: str) -> bool:
    """SQLite's REGEXP: whether a regular expression matches the text, Python's reading of
    it, which `glob_pattern` writes for."""
    return re.search(pattern, text) is not None


def vector_of(blob: bytes) -> Vector:
    """A stored vector, read from its bytes."""
    return numpy.frombuffer(blob, dtype=VECTOR_TYPE)


def placeholders(values: Sequence[object]) -> str:
    """As many of SQL's placeholders as there are values, for a statement's list."""
    return VALUE_SEPARATOR.join(VALUE_PLACEHOLDER for _ in values)


def batches(values: list[str]) -> list[list[str]]:
    """The values in lists of VALUES_PER_STATEMENT at most, one statement's each."""
    return [
        values[start : start + VALUES_PER_STATEMENT]
        for start in range(0, len(values), VALUES_PER_STATEMENT)
    ]
