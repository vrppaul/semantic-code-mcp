"""LanceDB vector storage: one table of chunks with their vectors and a full-text index."""

from pathlib import Path
from typing import override

import lancedb
import pyarrow
import structlog
from pydantic import BaseModel, ConfigDict

from semantic_code_mcp.domain.interfaces import VectorStore
from semantic_code_mcp.domain.models import (
    ChunkLocation,
    ChunkType,
    ChunkWithEmbedding,
    SearchResult,
    StoreStatistics,
)

log = structlog.get_logger()

EMBEDDING_DIMENSION = 384  # all-MiniLM-L6-v2
TABLE_NAME = "chunks"
CONTENT_COLUMN = "content"
FILE_PATH_COLUMN = "file_path"
FULL_TEXT_QUERY_TYPE = "fts"
COSINE_METRIC = "cosine"
# Columns LanceDB adds to a hit: the cosine distance of a vector hit, the score of a text hit
DISTANCE_COLUMN = "_distance"
FULL_TEXT_SCORE_COLUMN = "_score"

# Score normalization constants
COSINE_DISTANCE_MAX = 2.0  # cosine distance range is [0, 2] by definition
FULL_TEXT_SCORE_DIVISOR = 10.0  # empirical: the full-text scores seen peak around 5-15
FULL_TEXT_DEFAULT_SCORE = 0.5  # fallback when a text hit carries no score
LOWEST_SCORE = 0.0
HIGHEST_SCORE = 1.0
# How a single quote is written inside a quoted SQL string
SQL_QUOTE = "'"
SQL_ESCAPED_QUOTE = "''"

CHUNKS_SCHEMA = pyarrow.schema(
    [
        pyarrow.field("vector", pyarrow.list_(pyarrow.float32(), EMBEDDING_DIMENSION)),
        pyarrow.field(FILE_PATH_COLUMN, pyarrow.utf8()),
        pyarrow.field("line_start", pyarrow.int32()),
        pyarrow.field("line_end", pyarrow.int32()),
        pyarrow.field(CONTENT_COLUMN, pyarrow.utf8()),
        pyarrow.field("chunk_type", pyarrow.utf8()),
        pyarrow.field("name", pyarrow.utf8()),
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
    def add_chunks(self, items: list[ChunkWithEmbedding]) -> None:
        if not items:
            return
        rows = [
            {
                "vector": item.embedding,
                FILE_PATH_COLUMN: item.chunk.file_path,
                "line_start": item.chunk.line_start,
                "line_end": item.chunk.line_end,
                CONTENT_COLUMN: item.chunk.content,
                "chunk_type": item.chunk.chunk_type,
                "name": item.chunk.name,
            }
            for item in items
        ]
        self._table.add(rows)
        log.debug("added_chunks", count=len(rows))
        self._rebuild_full_text_index()

    def search_vector(self, query_embedding: list[float], limit: int) -> list[SearchResult]:
        """The chunks nearest to a vector, scored by cosine similarity mapped to [0, 1]."""
        table = self._table
        if table.count_rows() == 0:
            return []
        query = table.search(query_embedding)
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
        query_embedding: list[float],
        query_text: str,
        limit: int,
        vector_weight: float,
    ) -> list[SearchResult]:
        """Vector and full-text hits, weighted and summed per chunk, best first."""
        merged = merge_weighted(
            self.search_vector(query_embedding, limit),
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
        self._database.drop_table(TABLE_NAME)
        self._database.create_table(TABLE_NAME, schema=CHUNKS_SCHEMA)

    def _rebuild_full_text_index(self) -> None:
        """Rebuild the full-text index over the chunk contents after a write."""
        try:
            self._table.create_fts_index(CONTENT_COLUMN, replace=True)
            log.debug("rebuilt_full_text_index", column=CONTENT_COLUMN)
        except (OSError, ValueError, RuntimeError) as error:
            log.warning("full_text_index_not_built", error=str(error))


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
