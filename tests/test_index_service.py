"""Tests for IndexService."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from semantic_code_mcp.models import Chunk, ChunkType, IndexResult, ScanPlan
from semantic_code_mcp.services.index_service import IndexService


@pytest.fixture
def mock_indexer():
    indexer = MagicMock()
    indexer.scan_files.return_value = ["/a.py", "/b.py"]
    indexer.detect_changes.return_value = ScanPlan(
        files_to_index=["/a.py", "/b.py"],
        files_to_delete=[],
        all_files=["/a.py", "/b.py"],
    )

    async def fake_chunk_files(files):
        return [
            Chunk(
                file_path="/a.py",
                line_start=1,
                line_end=3,
                content="def foo(): pass",
                chunk_type=ChunkType.FUNCTION,
                name="foo",
            ),
        ]

    async def fake_embed_and_store(project_path, plan, chunks):
        return IndexResult(files_indexed=2, chunks_indexed=1, files_deleted=0)

    indexer.chunk_files = fake_chunk_files
    indexer.embed_and_store = fake_embed_and_store
    return indexer


@pytest.fixture
def index_service(mock_indexer) -> IndexService:
    return IndexService(mock_indexer)


class TestIndexService:
    """Tests for IndexService.index()."""

    @pytest.mark.asyncio
    async def test_index_returns_result_with_duration(self, index_service):
        """index() returns IndexResult with duration_seconds set."""
        result = await index_service.index(Path("/tmp/proj"), force=True)

        assert result.files_indexed == 2
        assert result.chunks_indexed == 1
        assert result.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_index_no_work_returns_zeros(self, index_service, mock_indexer):
        """index() returns zeros when no work is needed."""
        mock_indexer.detect_changes.return_value = ScanPlan(
            files_to_index=[],
            files_to_delete=[],
            all_files=["/a.py"],
        )

        result = await index_service.index(Path("/tmp/proj"))

        assert result.files_indexed == 0
        assert result.chunks_indexed == 0
        assert result.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_index_calls_progress(self, index_service):
        """index() calls progress callback."""
        progress_calls: list[tuple[float, float, str]] = []

        async def track_progress(progress: float, total: float, message: str) -> None:
            progress_calls.append((progress, total, message))

        await index_service.index(Path("/tmp/proj"), on_progress=track_progress)

        assert len(progress_calls) >= 3
        assert "Scanning" in progress_calls[0][2]

    @pytest.mark.asyncio
    async def test_index_force_passes_through(self, index_service, mock_indexer):
        """force=True is passed to detect_changes."""
        await index_service.index(Path("/tmp/proj"), force=True)

        mock_indexer.detect_changes.assert_called_once()
        _, kwargs = mock_indexer.detect_changes.call_args
        assert kwargs.get("force") is True
