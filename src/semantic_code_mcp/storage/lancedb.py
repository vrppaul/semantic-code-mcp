"""LanceDB vector storage: one table of chunks with their vectors and a full-text index."""

from pathlib import Path
from typing import override

import lancedb
import pyarrow
import structlog
from pydantic import BaseModel, ConfigDict

from semantic_code_mcp.domain.interfaces import Vector, VectorMatrix, VectorStore
from semantic_code_mcp.domain.models import (
    HIGHEST_SCORE,
    LOWEST_SCORE,
    Chunk,
    ChunkLocation,
    ChunkType,
    SearchResult,
    StoreStatistics,
)

log = structlog.get_logger()

EMBEDDING_DIMENSION = 384  # all-MiniLM-L6-v2
# The axis of a vector matrix along one vector: rows are chunks, columns components
VECTOR_LENGTH_AXIS = 1
TABLE_NAME = "chunks"
VECTOR_COLUMN = "vector"
FILE_PATH_COLUMN = "file_path"
LINE_START_COLUMN = "line_start"
LINE_END_COLUMN = "line_end"
CONTENT_COLUMN = "content"
CHUNK_TYPE_COLUMN = "chunk_type"
NAME_COLUMN = "name"
# A filter every row matches
MATCH_EVERY_ROW = "true"
FULL_TEXT_QUERY_TYPE = "fts"
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

CHUNKS_SCHEMA = pyarrow.schema(
    [
        pyarrow.field(VECTOR_COLUMN, pyarrow.list_(pyarrow.float32(), EMBEDDING_DIMENSION)),
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


class LanceDBVectorStore(VectorStore):
    """The chunks of one project in a LanceDB database directory."""

    def __init__(self, database_path: Path) -> None:
        self._database = lancedb.connect(str(database_path))
        self._database.create_table(TABLE_NAME, schema=CHUNKS_SCHEMA, exist_ok=True)

    @property
    def _table(self) -> lancedb.table.Table:
        return self._database.open_table(TABLE_NAME)

    @override
    def add_chunks(self, chunks: list[Chunk], vectors: VectorMatrix) -> None:
        if not chunks:
            return
        self._table.add(chunks_table(chunks, vectors))
        log.debug("added_chunks", count=len(chunks))
        self._rebuild_full_text_index()

    def search_vector(self, query_vector: Vector, limit: int) -> list[SearchResult]:
        """The chunks nearest to a vector, scored by cosine similarity mapped to [0, 1]."""
        table = self._table
        if table.count_rows() == 0:
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
        table = self._table
        if table.count_rows() == 0:
            return []
        try:
            rows = table.search(query_text, query_type=FULL_TEXT_QUERY_TYPE).limit(limit).to_list()
        except (OSError, ValueError, RuntimeError) as error:
            # No full-text index yet, or a query the index cannot parse
            log.warning("full_text_search_failed", error=str(error))
            return []
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

    @override
    def delete_by_files(self, file_paths: list[str]) -> None:
        if not file_paths:
            return
        quoted = ", ".join(quote_sql(file_path) for file_path in file_paths)
        self._table.delete(f"{FILE_PATH_COLUMN} IN ({quoted})")
        log.debug("deleted_chunks_for_files", count=len(file_paths))

    @override
    def statistics(self) -> StoreStatistics:
        table = self._table
        file_paths = table.to_arrow().column(FILE_PATH_COLUMN)
        return StoreStatistics(
            indexed_files=file_paths.unique().to_pylist(),
            chunks_count=table.count_rows(),
        )

    @override
    def clear(self) -> None:
        """Delete every row and keep the table, so a reader never finds it missing."""
        self._table.delete(MATCH_EVERY_ROW)

    def _rebuild_full_text_index(self) -> None:
        """Rebuild the full-text index over the chunk contents after a write."""
        try:
            self._table.create_fts_index(CONTENT_COLUMN, replace=True)
            log.debug("rebuilt_full_text_index", column=CONTENT_COLUMN)
        except (OSError, ValueError, RuntimeError) as error:
            log.warning("full_text_index_not_built", error=str(error))


def chunks_table(chunks: list[Chunk], vectors: VectorMatrix) -> pyarrow.Table:
    """Chunks as rows of the chunks schema, each with the vector of its row in the matrix.

    The matrix goes into Arrow as one flat buffer, without a Python float per component.
    """
    if len(chunks) != len(vectors):
        message = f"{len(chunks)} chunks but {len(vectors)} vectors"
        raise ValueError(message)
    vector_column = pyarrow.FixedSizeListArray.from_arrays(
        pyarrow.array(vectors.ravel()), vectors.shape[VECTOR_LENGTH_AXIS]
    )
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
        schema=CHUNKS_SCHEMA,
    )


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
