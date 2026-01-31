"""Index service - orchestrates full indexing pipeline."""

import asyncio
import time
from pathlib import Path

from semantic_code_mcp.indexer.indexer import Indexer
from semantic_code_mcp.models import IndexResult
from semantic_code_mcp.protocols import ProgressCallback


class IndexService:
    """Orchestrates the full indexing pipeline with timing and progress."""

    def __init__(self, indexer: Indexer) -> None:
        self.indexer = indexer

    async def index(
        self,
        project_path: Path,
        force: bool = False,
        on_progress: ProgressCallback | None = None,
    ) -> IndexResult:
        """Full index: scan, detect changes, chunk, embed, with timing + progress.

        Args:
            project_path: Root directory of the project.
            force: If True, re-index all files regardless of changes.
            on_progress: Optional callback matching ctx.report_progress(progress, total, message).

        Returns:
            IndexResult with counts and total duration.
        """
        start = time.perf_counter()

        async def _progress(percent: float, message: str) -> None:
            if on_progress is not None:
                await on_progress(percent, 100, message)

        await _progress(5, "Scanning files...")
        files = await asyncio.to_thread(self.indexer.scan_files, project_path)

        await _progress(10, f"Found {len(files)} files, detecting changes...")
        plan = self.indexer.detect_changes(project_path, files, force=force)

        if not plan.has_work:
            return IndexResult(
                files_indexed=0,
                chunks_indexed=0,
                files_deleted=0,
                duration_seconds=round(time.perf_counter() - start, 3),
            )

        await _progress(20, f"Chunking {len(plan.files_to_index)} files...")
        chunks = await self.indexer.chunk_files(plan.files_to_index)

        await _progress(70, "Embedding and storing...")
        result = await self.indexer.embed_and_store(project_path, plan, chunks)

        result.duration_seconds = round(time.perf_counter() - start, 3)
        return result
