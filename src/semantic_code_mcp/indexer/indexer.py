"""Indexing orchestration: scanning, chunking, embedding, storing."""

import asyncio
import fnmatch
import subprocess  # nosec B404
import time
from datetime import datetime
from pathlib import Path

import structlog

from semantic_code_mcp.config import Settings
from semantic_code_mcp.models import (
    Chunk,
    ChunkWithEmbedding,
    IndexResult,
    IndexStatus,
    ScanPlan,
)
from semantic_code_mcp.protocols import ChunkerProtocol, EmbedderProtocol, VectorStoreProtocol
from semantic_code_mcp.storage.cache import CACHE_FILENAME, FileChangeCache

log = structlog.get_logger()

CHUNK_BATCH_SIZE = 20


class Indexer:
    """Orchestrates the indexing pipeline: scan, chunk, embed, store.

    All dependencies are injected via constructor.
    """

    def __init__(
        self,
        settings: Settings,
        embedder: EmbedderProtocol,
        store: VectorStoreProtocol,
        chunker: ChunkerProtocol,
        cache_dir: Path | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.chunker = chunker
        self._cache_dir = cache_dir

    def _resolve_cache_dir(self, project_path: Path) -> Path:
        """Get cache dir, falling back to settings-derived path."""
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            return self._cache_dir
        from semantic_code_mcp.config import get_index_path  # circular import guard

        path = get_index_path(self.settings, project_path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def scan_files(self, project_path: Path) -> list[str]:
        """Scan for Python files in the project.

        Uses git ls-files if available (fast, respects .gitignore).
        Falls back to os.walk with directory pruning.

        Args:
            project_path: Root directory to scan.

        Returns:
            List of absolute file paths.
        """
        project_path = project_path.resolve()

        # Try git ls-files first (fast, respects .gitignore)
        if self._is_git_repo(project_path):
            files = self._scan_with_git(project_path)
            if files is not None:
                log.debug("scanned_files_git", project=str(project_path), count=len(files))
                return files

        # Fall back to os.walk with pruning
        files = self._scan_with_walk(project_path)
        log.debug("scanned_files_walk", project=str(project_path), count=len(files))
        return files

    def _is_git_repo(self, project_path: Path) -> bool:
        """Check if project is a git repository."""
        return (project_path / ".git").is_dir()

    def _scan_with_git(self, project_path: Path) -> list[str] | None:
        """Scan using git ls-files. Returns None if not a git repo."""
        result = subprocess.run(  # nosec B603, B607
            ["git", "ls-files", "*.py"],
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None

        files = []
        for line in result.stdout.strip().split("\n"):
            if line:
                files.append(str(project_path / line))
        return files

    def _scan_with_walk(self, project_path: Path) -> list[str]:
        """Scan using os.walk with directory pruning."""
        # Directories to skip entirely
        skip_dirs = {".venv", ".git", "node_modules", "__pycache__", ".pytest_cache", "venv"}

        # Load additional patterns from gitignore
        gitignore_patterns: list[str] = []
        if self.settings.use_gitignore:
            gitignore_path = project_path / ".gitignore"
            if gitignore_path.exists():
                gitignore_patterns = self._parse_gitignore(gitignore_path)

        all_ignore_patterns = self.settings.ignore_patterns + gitignore_patterns

        files: list[str] = []
        for root, dirs, filenames in project_path.walk():
            # Prune directories in-place to avoid descending
            dirs[:] = [d for d in dirs if d not in skip_dirs]

            for filename in filenames:
                if not filename.endswith(".py"):
                    continue

                file_path = root / filename
                rel_path = file_path.relative_to(project_path)

                if self._should_ignore(str(rel_path), all_ignore_patterns):
                    continue

                files.append(str(file_path))

        return files

    def _parse_gitignore(self, gitignore_path: Path) -> list[str]:
        """Parse .gitignore file into patterns.

        Args:
            gitignore_path: Path to .gitignore file.

        Returns:
            List of ignore patterns.
        """
        patterns: list[str] = []
        try:
            content = gitignore_path.read_text()
            for line in content.splitlines():
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue
                # Convert directory patterns
                if line.endswith("/"):
                    patterns.append(line + "**")
                    patterns.append(line[:-1])  # Also match the directory itself
                else:
                    patterns.append(line)
                    patterns.append("**/" + line)  # Match in any subdirectory
        except OSError:
            pass
        return patterns

    def _should_ignore(self, rel_path: str, patterns: list[str]) -> bool:
        """Check if a file should be ignored.

        Args:
            rel_path: Relative path from project root.
            patterns: List of ignore patterns.

        Returns:
            True if the file should be ignored.
        """
        # Normalize path separators
        rel_path = rel_path.replace("\\", "/")

        for pattern in patterns:
            pattern = pattern.replace("\\", "/")
            if fnmatch.fnmatch(rel_path, pattern):
                return True
            # Check if any parent directory matches
            parts = rel_path.split("/")
            for i in range(len(parts)):
                partial = "/".join(parts[: i + 1])
                if fnmatch.fnmatch(partial, pattern):
                    return True
        return False

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
        cache_dir = self._resolve_cache_dir(project_path)
        cache = FileChangeCache(cache_dir)

        if force:
            files_to_index = current_files
            files_to_delete: list[str] = []
            cache.clear()
            self.store.clear()
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

    async def embed_and_store(
        self, project_path: Path, plan: ScanPlan, chunks: list[Chunk]
    ) -> IndexResult:
        """Delete stale data, embed chunks, store them, and update cache.

        Args:
            project_path: Root directory of the project (for cache resolution).
            plan: The ScanPlan from detect_changes().
            chunks: Chunks extracted by chunk_files().

        Returns:
            IndexResult with counts of work done.
        """
        cache_dir = self._resolve_cache_dir(project_path)
        cache = FileChangeCache(cache_dir)

        # Delete chunks for removed/modified files
        for file_path in plan.files_to_delete:
            self.store.delete_by_file(file_path)
            cache.remove_file(file_path)

        # For incremental: delete old chunks for files being re-indexed
        for file_path in plan.files_to_index:
            self.store.delete_by_file(file_path)

        if chunks:
            await self._embed_and_store(chunks)

        # Update cache with indexed files
        cache.update_files(plan.files_to_index)

        return IndexResult(
            files_indexed=len(plan.files_to_index),
            chunks_indexed=len(chunks),
            files_deleted=len(plan.files_to_delete),
        )

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

    async def _embed_and_store(self, chunks: list[Chunk]) -> None:
        """Generate embeddings and store chunks."""
        contents = [chunk.content for chunk in chunks]
        t0 = time.time()
        embeddings = await asyncio.to_thread(self.embedder.embed_batch, contents)
        log.debug(
            "embedding_completed",
            chunks=len(chunks),
            duration_ms=round((time.time() - t0) * 1000, 1),
        )

        items = [
            ChunkWithEmbedding(chunk=chunk, embedding=embedding)
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]

        t0 = time.time()
        await asyncio.to_thread(self.store.add_chunks, items)
        log.debug(
            "storage_completed",
            chunks=len(items),
            duration_ms=round((time.time() - t0) * 1000, 1),
        )

    def get_status(self, project_path: Path) -> IndexStatus:
        """Get the index status for a project.

        Args:
            project_path: Root directory of the project.

        Returns:
            IndexStatus with current state information.
        """
        project_path = project_path.resolve()
        cache_dir = self._resolve_cache_dir(project_path)

        # Check if index exists
        if not cache_dir.exists():
            return IndexStatus(
                is_indexed=False,
                last_updated=None,
                files_count=0,
                chunks_count=0,
                stale_files=[],
            )

        cache = FileChangeCache(cache_dir)

        # Get current files and check for staleness
        current_files = self.scan_files(project_path)
        stale_files = cache.get_stale_files(current_files)

        # Get indexed file list from injected store
        indexed_files = self.store.get_indexed_files()
        chunks_count = self.store.count()

        # Determine last updated from cache file mtime
        cache_file = cache_dir / CACHE_FILENAME
        last_updated = None
        if cache_file.exists():
            last_updated = datetime.fromtimestamp(cache_file.stat().st_mtime)

        return IndexStatus(
            is_indexed=len(indexed_files) > 0,
            last_updated=last_updated,
            files_count=len(indexed_files),
            chunks_count=chunks_count,
            stale_files=stale_files,
        )
