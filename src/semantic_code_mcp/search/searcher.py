"""Semantic search and ranking."""

from pathlib import Path

import structlog

from semantic_code_mcp.config import Settings, get_index_path
from semantic_code_mcp.indexer.embedder import Embedder
from semantic_code_mcp.indexer.indexer import Indexer
from semantic_code_mcp.models import SearchResult
from semantic_code_mcp.storage.lancedb import VectorStore

log = structlog.get_logger()


class Searcher:
    """Performs semantic search over indexed code.

    Automatically reindexes stale files before searching.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the searcher.

        Args:
            settings: Application settings.
        """
        self.settings = settings
        self.embedder = Embedder(settings)
        self.indexer = Indexer(settings)

    def search(
        self,
        project_path: Path,
        query: str,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Search for code semantically similar to the query.

        Automatically reindexes any stale files before searching.

        Args:
            project_path: Root directory of the project.
            query: Natural language search query.
            limit: Maximum number of results to return.

        Returns:
            List of SearchResult objects sorted by relevance.
        """
        project_path = project_path.resolve()
        index_path = get_index_path(self.settings, project_path)

        log.info("search_started", project=str(project_path), query=query)

        # Check if index exists
        if not index_path.exists():
            log.info("index_not_found", project=str(project_path))
            # Full index needed
            result = self.indexer.index(project_path)
            log.info(
                "auto_indexed",
                files=result.files_indexed,
                chunks=result.chunks_indexed,
            )

            # If still no index (empty project), return empty
            if not index_path.exists():
                return []

        # Check for stale files and reindex if needed
        status = self.indexer.get_status(project_path)
        if status.stale_files:
            log.info(
                "reindexing_stale_files",
                count=len(status.stale_files),
                files=status.stale_files[:5],  # Log first 5
            )
            result = self.indexer.index(project_path, force=False)
            log.info(
                "reindex_completed",
                files=result.files_indexed,
                chunks=result.chunks_indexed,
            )

        # Embed the query
        log.debug("embedding_query", query=query)
        query_embedding = self.embedder.embed_text(query)

        # Search the vector store
        store = VectorStore(index_path)
        results = store.search(query_embedding, limit=limit)

        log.info(
            "search_completed",
            results_count=len(results),
            top_score=results[0].score if results else None,
        )

        return results
