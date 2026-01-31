"""Tests for Indexer (embed + store only)."""

from pathlib import Path

import pytest

from semantic_code_mcp.chunkers.python import PythonChunker
from semantic_code_mcp.config import Settings, get_index_path
from semantic_code_mcp.embedder import Embedder
from semantic_code_mcp.indexer import Indexer
from semantic_code_mcp.models import Chunk, ChunkType, ScanPlan
from semantic_code_mcp.storage.lancedb import LanceDBConnection, LanceDBVectorStore


class TestIndexer:
    """Tests for Indexer embed+store with real dependencies."""

    @pytest.fixture
    def chunker(self) -> PythonChunker:
        return PythonChunker()

    @pytest.fixture
    def store(self, test_settings: Settings, sample_project: Path) -> LanceDBVectorStore:
        index_path = get_index_path(test_settings, sample_project)
        index_path.mkdir(parents=True, exist_ok=True)
        connection = LanceDBConnection(index_path)
        return LanceDBVectorStore(connection)

    @pytest.fixture
    def indexer(self, embedder: Embedder, store: LanceDBVectorStore) -> Indexer:
        return Indexer(embedder=embedder, store=store)

    def test_create_indexer(self, embedder: Embedder, store: LanceDBVectorStore):
        """Can create an indexer instance with injected dependencies."""
        indexer = Indexer(embedder=embedder, store=store)
        assert indexer is not None
        assert indexer.embedder is embedder
        assert indexer.store is store

    @pytest.mark.asyncio
    async def test_embed_and_store(
        self, indexer: Indexer, sample_project: Path, chunker: PythonChunker
    ):
        """embed_and_store embeds chunks and stores them."""
        files = [str(sample_project / "main.py"), str(sample_project / "utils.py")]
        chunks = []
        for f in files:
            chunks.extend(chunker.chunk_file(f))

        plan = ScanPlan(
            files_to_index=files,
            files_to_delete=[],
            all_files=files,
        )
        await indexer.embed_and_store(plan, chunks)

        indexed_files, count = indexer.get_store_stats()
        assert len(indexed_files) == 2
        assert count == len(chunks)
        assert count > 0

    def test_clear_store(self, indexer: Indexer):
        """clear_store delegates to store.clear()."""
        indexer.clear_store()
        indexed_files, count = indexer.get_store_stats()
        assert indexed_files == []
        assert count == 0

    def test_get_store_stats(self, indexer: Indexer):
        """get_store_stats returns empty state for fresh store."""
        indexed_files, count = indexer.get_store_stats()
        assert indexed_files == []
        assert count == 0


class TestIndexerWithMocks:
    """Tests for Indexer using mock dependencies for isolation."""

    def test_indexer_accepts_injected_deps(self, mock_embedder, mock_store):
        """Indexer stores injected dependencies."""
        indexer = Indexer(embedder=mock_embedder, store=mock_store)
        assert indexer.embedder is mock_embedder
        assert indexer.store is mock_store

    @pytest.mark.asyncio
    async def test_embed_and_store_calls_embedder(self, mock_embedder, mock_store):
        """embed_and_store calls embedder.embed_batch and store.add_chunks."""
        indexer = Indexer(embedder=mock_embedder, store=mock_store)
        chunks = [
            Chunk(
                file_path="/test.py",
                line_start=1,
                line_end=5,
                content="def test(): pass",
                chunk_type=ChunkType.FUNCTION,
                name="test",
            )
        ]
        plan = ScanPlan(
            files_to_index=["/test.py"],
            files_to_delete=[],
            all_files=["/test.py"],
        )
        await indexer.embed_and_store(plan, chunks)

        mock_embedder.embed_batch.assert_called_once()
        mock_store.add_chunks.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_and_store_empty_chunks(self, mock_embedder, mock_store):
        """embed_and_store with empty chunks skips embedding."""
        indexer = Indexer(embedder=mock_embedder, store=mock_store)
        plan = ScanPlan(
            files_to_index=["/test.py"],
            files_to_delete=[],
            all_files=["/test.py"],
        )
        await indexer.embed_and_store(plan, [])

        mock_embedder.embed_batch.assert_not_called()
        mock_store.add_chunks.assert_not_called()

    @pytest.mark.asyncio
    async def test_embed_and_store_deletes_stale_files(self, mock_embedder, mock_store):
        """embed_and_store deletes chunks for files_to_delete."""
        indexer = Indexer(embedder=mock_embedder, store=mock_store)
        plan = ScanPlan(
            files_to_index=[],
            files_to_delete=["/old.py", "/removed.py"],
            all_files=[],
        )
        await indexer.embed_and_store(plan, [])

        assert mock_store.delete_by_file.call_count == 2
        mock_store.delete_by_file.assert_any_call("/old.py")
        mock_store.delete_by_file.assert_any_call("/removed.py")
