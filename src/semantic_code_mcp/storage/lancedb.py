"""LanceDB vector storage operations."""

from pathlib import Path

import lancedb
import pyarrow as pa
import structlog

from semantic_code_mcp.models import ChunkType, ChunkWithEmbedding, SearchResult

log = structlog.get_logger()

EMBEDDING_DIMENSION = 384  # all-MiniLM-L6-v2
TABLE_NAME = "chunks"

# Score normalization constants
COSINE_DISTANCE_MAX = 2.0  # cosine distance range is [0, 2] by definition
FTS_SCORE_DIVISOR = 10.0  # empirical: LanceDB tantivy FTS scores typically peak ~5-15
FTS_DEFAULT_SCORE = 0.5  # fallback when FTS returns no _score field

# Schema for the chunks table
CHUNKS_SCHEMA = pa.schema(
    [
        pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIMENSION)),
        pa.field("file_path", pa.utf8()),
        pa.field("line_start", pa.int32()),
        pa.field("line_end", pa.int32()),
        pa.field("content", pa.utf8()),
        pa.field("chunk_type", pa.utf8()),
        pa.field("name", pa.utf8()),
    ]
)


class LanceDBConnection:
    """Manages LanceDB connection (expensive, created once)."""

    def __init__(self, db_path: Path) -> None:
        """Initialize the LanceDB connection.

        Args:
            db_path: Path to the LanceDB database directory.
        """
        self.db_path = db_path
        self.db = lancedb.connect(str(db_path))
        self.ensure_table()

    def ensure_table(self) -> None:
        """Ensure the chunks table exists with FTS index."""
        try:
            self.db.create_table(TABLE_NAME, schema=CHUNKS_SCHEMA, exist_ok=True)
            log.debug("ensured_table", table=TABLE_NAME)
        except ValueError:
            # Table already exists (race condition or stale cache)
            pass

        # Ensure FTS index exists on content column
        self._ensure_fts_index()

    def _ensure_fts_index(self, *, force: bool = False) -> None:
        """Ensure full-text search index exists on content column.

        Args:
            force: If True, always rebuild the index. If False, skip if index exists.
        """
        try:
            table = self.table
            if table.count_rows() == 0:
                return  # Can't create index on empty table

            # Check if FTS index already exists (skip expensive rebuild)
            if not force:
                existing_indices = table.list_indices()
                fts_exists = any(
                    idx.index_type == "FTS" and "content" in idx.columns for idx in existing_indices
                )
                if fts_exists:
                    log.debug("fts_index_exists", column="content")
                    return

            # Create/rebuild FTS index on content column
            table.create_fts_index("content", replace=True)
            log.debug("created_fts_index", column="content")
        except (OSError, ValueError, RuntimeError) as e:
            # Index creation may fail: IO errors, empty table, or LanceDB internal errors
            log.warning("fts_index_creation_skipped", error=str(e))

    @property
    def table(self) -> lancedb.table.Table:
        """Get the chunks table."""
        return self.db.open_table(TABLE_NAME)

    def rebuild_fts_index(self) -> None:
        """Force rebuild of FTS index. Called after adding chunks."""
        self._ensure_fts_index(force=True)

    def drop_table(self) -> None:
        """Drop the chunks table."""
        try:
            self.db.drop_table(TABLE_NAME)
            log.debug("dropped_table", table=TABLE_NAME)
        except ValueError:
            # Table doesn't exist
            pass


class LanceDBVectorStore:
    """Per-request session wrapping the connection.

    Lightweight wrapper around LanceDBConnection that provides
    the VectorStoreProtocol interface for indexing and search operations.
    """

    def __init__(self, connection: LanceDBConnection) -> None:
        """Initialize the vector store session.

        Args:
            connection: Shared LanceDB connection.
        """
        self.connection = connection

    def add_chunks(self, items: list[ChunkWithEmbedding]) -> None:
        """Add chunks with their embeddings to the store.

        Args:
            items: List of ChunkWithEmbedding objects.
        """
        if not items:
            return

        data = [
            {
                "vector": item.embedding,
                "file_path": item.chunk.file_path,
                "line_start": item.chunk.line_start,
                "line_end": item.chunk.line_end,
                "content": item.chunk.content,
                "chunk_type": item.chunk.chunk_type,
                "name": item.chunk.name,
            }
            for item in items
        ]

        table = self.connection.table
        table.add(data)
        log.debug("added_chunks", count=len(data))

        # Rebuild FTS index after adding data
        self.connection.rebuild_fts_index()

    def search(self, query_embedding: list[float], limit: int = 10) -> list[SearchResult]:
        """Search for similar chunks.

        Args:
            query_embedding: The query vector.
            limit: Maximum number of results.

        Returns:
            List of SearchResult objects sorted by similarity.
        """
        table = self.connection.table

        if table.count_rows() == 0:
            return []

        results = table.search(query_embedding).metric("cosine").limit(limit).to_pandas()  # type: ignore[possibly-missing-attribute]  # lancedb stubs incomplete

        search_results = []
        for _, row in results.iterrows():
            # LanceDB returns _distance (cosine distance in [0, 2], lower is better)
            # Convert to score where higher is better: score = 1 - (distance / 2)
            distance = row["_distance"]
            score = max(0.0, min(1.0, 1.0 - distance / COSINE_DISTANCE_MAX))

            search_results.append(
                SearchResult(
                    file_path=row["file_path"],
                    line_start=int(row["line_start"]),
                    line_end=int(row["line_end"]),
                    content=row["content"],
                    chunk_type=ChunkType(row["chunk_type"]),
                    name=row["name"],
                    score=score,
                )
            )

        return search_results

    def search_fts(self, query_text: str, limit: int = 10) -> list[SearchResult]:
        """Search using full-text search only.

        Args:
            query_text: The text query to search for.
            limit: Maximum number of results.

        Returns:
            List of SearchResult objects.
        """
        table = self.connection.table

        if table.count_rows() == 0:
            return []

        try:
            results = table.search(query_text, query_type="fts").limit(limit).to_pandas()
        except (OSError, ValueError, RuntimeError) as e:
            # FTS may fail if index doesn't exist or query is invalid
            log.warning("fts_search_failed", error=str(e))
            return []

        search_results = []
        for _, row in results.iterrows():
            # FTS returns _score (higher is better, but scale varies)
            # Normalize to 0-1 range (approximate)
            score = min(1.0, row.get("_score", FTS_DEFAULT_SCORE) / FTS_SCORE_DIVISOR)

            search_results.append(
                SearchResult(
                    file_path=row["file_path"],
                    line_start=int(row["line_start"]),
                    line_end=int(row["line_end"]),
                    content=row["content"],
                    chunk_type=ChunkType(row["chunk_type"]),
                    name=row["name"],
                    score=score,
                )
            )

        return search_results

    def search_hybrid(
        self,
        query_embedding: list[float],
        query_text: str,
        limit: int = 10,
        vector_weight: float = 0.5,
    ) -> list[SearchResult]:
        """Search using both vector similarity and full-text search.

        Runs both searches in parallel and merges results, ensuring both
        semantically similar AND keyword-matching results are included.

        Args:
            query_embedding: The query vector.
            query_text: The text query for FTS.
            limit: Maximum number of results.
            vector_weight: Weight for vector search (0.0 to 1.0).
                          FTS weight is (1 - vector_weight).

        Returns:
            List of SearchResult objects combining both search methods.
        """
        vector_results = self.search(query_embedding, limit)
        fts_results = self.search_fts(query_text, limit)

        fts_weight = 1.0 - vector_weight
        merged = self._merge_results(vector_results, vector_weight, fts_results, fts_weight)

        merged.sort(key=lambda r: r.score, reverse=True)
        return merged[:limit]

    @staticmethod
    def _merge_results(
        primary: list[SearchResult],
        primary_weight: float,
        secondary: list[SearchResult],
        secondary_weight: float,
    ) -> list[SearchResult]:
        """Merge two ranked result lists with weighted scores."""
        seen: dict[tuple[str, int], SearchResult] = {}

        for r in primary:
            key = (r.file_path, r.line_start)
            seen[key] = r.model_copy(update={"score": r.score * primary_weight})

        for r in secondary:
            key = (r.file_path, r.line_start)
            weighted = r.score * secondary_weight
            if key in seen:
                combined = min(1.0, seen[key].score + weighted)
                seen[key] = seen[key].model_copy(update={"score": combined})
            else:
                seen[key] = r.model_copy(update={"score": weighted})

        return list(seen.values())

    def delete_by_file(self, file_path: str) -> None:
        """Delete all chunks for a specific file.

        Args:
            file_path: The file path to delete chunks for.
        """
        table = self.connection.table
        escaped = file_path.replace("'", "''")
        table.delete(f"file_path = '{escaped}'")
        log.debug("deleted_chunks_for_file", file_path=file_path)

    def get_indexed_files(self) -> list[str]:
        """Get list of all indexed file paths.

        Returns:
            List of unique file paths in the store.
        """
        table = self.connection.table

        if table.count_rows() == 0:
            return []

        df = table.to_pandas()
        return df["file_path"].unique().tolist()

    def count(self) -> int:
        """Count total chunks in the store.

        Returns:
            Number of chunks.
        """
        table = self.connection.table
        return table.count_rows()

    def clear(self) -> None:
        """Delete all chunks from the store."""
        self.connection.drop_table()
        # Recreate empty table
        self.connection.ensure_table()
