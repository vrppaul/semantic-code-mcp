"""Tests for SearchService."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from semantic_code_mcp.models import ChunkType, IndexResult, IndexStatus, SearchResult
from semantic_code_mcp.services.index_service import IndexService
from semantic_code_mcp.services.search_service import SearchService


def _make_result(name: str, score: float, file_path: str = "/a.py") -> SearchResult:
    return SearchResult(
        file_path=file_path,
        line_start=1,
        line_end=5,
        content=f"def {name}(): pass",
        chunk_type=ChunkType.function,
        name=name,
        score=score,
    )


@pytest.fixture
def mock_index_service() -> IndexService:
    service = MagicMock(spec=IndexService)
    service.get_status.return_value = IndexStatus(
        is_indexed=True,
        last_updated=None,
        files_count=5,
        chunks_count=20,
        stale_files=[],
    )
    return service


@pytest.fixture
def search_service(mock_store, mock_embedder, mock_index_service) -> SearchService:
    return SearchService(store=mock_store, embedder=mock_embedder, index_service=mock_index_service)


class TestSearch:
    """Tests for search() with callback."""

    @pytest.mark.asyncio
    async def test_search_returns_results_from_store(self, search_service, mock_store):
        """Search returns results from the vector store."""
        mock_store.search_hybrid.return_value = [
            _make_result("foo", 0.9),
            _make_result("bar", 0.8),
        ]

        outcome = await search_service.search("find functions", Path("/tmp/proj"))

        assert len(outcome.results) == 2
        assert outcome.results[0].name == "foo"
        assert outcome.raw_count == 2

    @pytest.mark.asyncio
    async def test_search_filters_low_scores(self, search_service, mock_store):
        """Search filters results below min_score."""
        mock_store.search_hybrid.return_value = [
            _make_result("good", 0.9),
            _make_result("bad", 0.1),
        ]

        outcome = await search_service.search("find functions", Path("/tmp/proj"), min_score=0.3)

        assert len(outcome.results) == 1
        assert outcome.results[0].name == "good"
        assert outcome.filtered_count == 1

    @pytest.mark.asyncio
    async def test_search_captures_timing(self, search_service, mock_store):
        """Search outcome includes timing data."""
        mock_store.search_hybrid.return_value = [_make_result("foo", 0.9)]

        outcome = await search_service.search("test", Path("/tmp/proj"))

        assert outcome.embedding_ms >= 0
        assert outcome.search_ms >= 0
        assert outcome.total_ms >= 0

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, search_service, mock_store):
        """Search respects the limit parameter."""
        mock_store.search_hybrid.return_value = [
            _make_result(f"func{i}", 0.9 - i * 0.05) for i in range(10)
        ]

        outcome = await search_service.search("test", Path("/tmp/proj"), limit=3)

        assert len(outcome.results) <= 3

    @pytest.mark.asyncio
    async def test_search_calls_progress_callback(self, search_service, mock_store):
        """Search calls on_progress callback with expected values."""
        mock_store.search_hybrid.return_value = [_make_result("foo", 0.9)]

        progress_calls: list[tuple[float, float, str]] = []

        async def track_progress(progress: float, total: float, message: str) -> None:
            progress_calls.append((progress, total, message))

        await search_service.search("test", Path("/tmp/proj"), on_progress=track_progress)

        assert len(progress_calls) >= 2
        # First call should be checking index
        assert progress_calls[0][2] == "Checking index..."
        # Last call should report results
        assert "Found" in progress_calls[-1][2]


class TestGroupByFile:
    """Tests for _group_by_file ordering."""

    def test_groups_results_by_file(self, search_service):
        """Results are grouped by file path."""
        results = [
            _make_result("a1", 0.9, "/a.py"),
            _make_result("b1", 0.85, "/b.py"),
            _make_result("a2", 0.7, "/a.py"),
        ]

        grouped = search_service._group_by_file(results)

        # a.py has best score (0.9), so comes first
        assert grouped[0].file_path == "/a.py"
        assert grouped[1].file_path == "/a.py"
        assert grouped[2].file_path == "/b.py"

    def test_uses_max_score_for_file_ordering(self, search_service):
        """Files are ordered by their best chunk's score, not first chunk."""
        results = [
            _make_result("b_low", 0.5, "/b.py"),
            _make_result("a_low", 0.3, "/a.py"),
            _make_result("a_high", 0.95, "/a.py"),
        ]

        grouped = search_service._group_by_file(results)

        # a.py has max score 0.95, b.py has 0.5
        assert grouped[0].file_path == "/a.py"


class TestRecencyBoost:
    """Tests for _apply_recency_boost."""

    def test_boost_does_not_exceed_1(self, search_service):
        """Boosted score is capped at 1.0."""
        results = [_make_result("foo", 0.99)]

        boosted = search_service._apply_recency_boost(results)

        assert boosted[0][1] <= 1.0


class TestSearchAutoIndex:
    """Tests for search() auto-indexing behavior."""

    @pytest.mark.asyncio
    async def test_auto_indexes_when_not_indexed(
        self, search_service, mock_index_service, mock_store
    ):
        """Triggers indexing when project is not indexed."""
        mock_index_service.get_status.return_value = IndexStatus(
            is_indexed=False,
            last_updated=None,
            files_count=0,
            chunks_count=0,
            stale_files=[],
        )

        async def fake_index(project_path, force=False):
            return IndexResult(
                files_indexed=2, chunks_indexed=10, files_deleted=0, duration_seconds=0.5
            )

        mock_index_service.index = fake_index
        mock_store.search_hybrid.return_value = [_make_result("foo", 0.9)]

        outcome = await search_service.search("test", Path("/tmp/proj"), 10)

        assert outcome is not None
        assert outcome.index_result.files_indexed == 2

    @pytest.mark.asyncio
    async def test_reindexes_stale_files(self, search_service, mock_index_service, mock_store):
        """Re-indexes when stale files are detected."""
        mock_index_service.get_status.return_value = IndexStatus(
            is_indexed=True,
            last_updated=None,
            files_count=5,
            chunks_count=20,
            stale_files=["/a.py", "/b.py"],
        )

        async def fake_index(project_path, force=False):
            return IndexResult(
                files_indexed=2, chunks_indexed=5, files_deleted=0, duration_seconds=0.3
            )

        mock_index_service.index = fake_index
        mock_store.search_hybrid.return_value = []

        outcome = await search_service.search("test", Path("/tmp/proj"), 10)

        assert outcome is not None
        assert outcome.index_result.files_indexed == 2

    @pytest.mark.asyncio
    async def test_skips_indexing_when_up_to_date(
        self, search_service, mock_index_service, mock_store
    ):
        """Skips indexing when index is fresh."""
        mock_store.search_hybrid.return_value = [_make_result("foo", 0.9)]

        # mock_index_service already returns is_indexed=True, stale_files=[]
        outcome = await search_service.search("test", Path("/tmp/proj"), 10)

        assert outcome is not None
        # No indexing should have happened — index_result stays at default zeros
        assert outcome.index_result.files_indexed == 0
        mock_index_service.index.assert_not_called()
