"""Index service — orchestrates full indexing pipeline."""

import asyncio
import time
from datetime import UTC, datetime
from pathlib import Path

import structlog

from semantic_code_mcp.chunkers.composite import CompositeChunker
from semantic_code_mcp.config import Settings, resolve_cache_dir
from semantic_code_mcp.discovery import SourceDiscovery
from semantic_code_mcp.indexer import Indexer
from semantic_code_mcp.models import (
    Chunk,
    IndexResult,
    IndexStatus,
    ScanPlan,
)
from semantic_code_mcp.protocols import ProgressCallback
from semantic_code_mcp.storage.cache import CACHE_FILENAME, FileChangeCache

log = structlog.get_logger()

CHUNK_BATCH_SIZE = 20


class IndexService:
    """Orchestrates the full indexing pipeline with timing and progress.

    Owns scanning, change detection, chunking, and status.
    Delegates embedding and storage to Indexer.
    """

    def __init__(
        self,
        settings: Settings,
        indexer: Indexer,
        chunker: CompositeChunker,
        cache_dir: Path | None = None,
    ) -> None:
        self.settings = settings
        self.indexer = indexer
        self.chunker = chunker
        self._cache_dir = cache_dir
        self.discovery = SourceDiscovery(
            extensions=chunker.supported_extensions,
            ignore_patterns=settings.ignore_patterns,
            use_gitignore=settings.use_gitignore,
        )

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
        files = await asyncio.to_thread(self.scan_files, project_path)

        await _progress(10, f"Found {len(files)} files, detecting changes...")
        plan = self.detect_changes(project_path, files, force=force)

        if not plan.has_work:
            return IndexResult(
                files_indexed=0,
                chunks_indexed=0,
                files_deleted=0,
                duration_seconds=round(time.perf_counter() - start, 3),
            )

        await _progress(20, f"Chunking {len(plan.files_to_index)} files...")
        chunks = await self.chunk_files(plan.files_to_index)

        await _progress(70, "Embedding and storing...")
        await self.indexer.embed_and_store(plan, chunks)

        # Update cache after successful embed+store
        cache_dir = resolve_cache_dir(self.settings, project_path, self._cache_dir)
        cache = FileChangeCache(cache_dir)
        if plan.files_to_delete:
            cache.remove_files(plan.files_to_delete)
        if plan.files_to_index:
            cache.update_files(plan.files_to_index)

        return IndexResult(
            files_indexed=len(plan.files_to_index),
            chunks_indexed=len(chunks),
            files_deleted=len(plan.files_to_delete),
            duration_seconds=round(time.perf_counter() - start, 3),
        )

    # --- Scanning ---

    def scan_files(self, project_path: Path) -> list[str]:
        """List the indexable source files of a project as absolute paths."""
        files = [str(path) for path in self.discovery.discover(project_path)]
        log.debug("scanned_files", project=str(project_path), count=len(files))
        return files

    # --- Change detection ---

    def detect_changes(
        self, project_path: Path, current_files: list[str], force: bool = False
    ) -> ScanPlan:
        """Detect which files need indexing/deletion.

        Args:
            project_path: Root directory of the project.
            current_files: List of absolute file paths from scan_files().
            force: If True, re-index all files.

        Returns:
            ScanPlan describing what work needs to be done.
        """
        cache_dir = resolve_cache_dir(self.settings, project_path, self._cache_dir)
        cache = FileChangeCache(cache_dir)

        if force:
            files_to_index = current_files
            files_to_delete: list[str] = []
            cache.clear()
            self.indexer.clear_store()
        else:
            changes = cache.get_changes(current_files)
            files_to_index = changes.stale_files
            files_to_delete = changes.deleted
            log.debug(
                "incremental_index",
                new=len(changes.new),
                modified=len(changes.modified),
                deleted=len(changes.deleted),
            )

        return ScanPlan(
            files_to_index=files_to_index,
            files_to_delete=files_to_delete,
            all_files=current_files,
        )

    # --- Chunking ---

    async def chunk_files(self, files: list[str]) -> list[Chunk]:
        """Chunk files in parallel batches."""
        all_chunks: list[Chunk] = []
        total_files = len(files)

        t0 = time.time()
        for batch_start in range(0, total_files, CHUNK_BATCH_SIZE):
            batch_end = min(batch_start + CHUNK_BATCH_SIZE, total_files)
            batch = files[batch_start:batch_end]

            chunk_tasks = [
                asyncio.to_thread(self.chunker.chunk_file, file_path) for file_path in batch
            ]
            batch_results = await asyncio.gather(*chunk_tasks)
            for chunks in batch_results:
                all_chunks.extend(chunks)

        log.debug(
            "chunking_completed",
            files=total_files,
            chunks=len(all_chunks),
            duration_ms=round((time.time() - t0) * 1000, 1),
        )
        return all_chunks

    # --- Status ---

    def get_status(self, project_path: Path) -> IndexStatus:
        """Get the index status for a project.

        Args:
            project_path: Root directory of the project.

        Returns:
            IndexStatus with current state information.
        """
        project_path = project_path.resolve()
        cache_dir = resolve_cache_dir(self.settings, project_path, self._cache_dir)

        if not cache_dir.exists():
            return IndexStatus(
                is_indexed=False,
                last_updated=None,
                files_count=0,
                chunks_count=0,
                stale_files=[],
            )

        cache = FileChangeCache(cache_dir)

        current_files = self.scan_files(project_path)
        stale_files = cache.get_stale_files(current_files)

        indexed_files, chunks_count = self.indexer.get_store_stats()

        cache_file = cache_dir / CACHE_FILENAME
        last_updated = None
        if cache_file.exists():
            last_updated = datetime.fromtimestamp(cache_file.stat().st_mtime, tz=UTC)

        return IndexStatus(
            is_indexed=len(indexed_files) > 0,
            last_updated=last_updated,
            files_count=len(indexed_files),
            chunks_count=chunks_count,
            stale_files=stale_files,
        )
