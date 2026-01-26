"""Indexing orchestration: scanning, chunking, embedding, storing."""

import asyncio
import fnmatch
import subprocess  # nosec B404
import time
from collections.abc import AsyncGenerator
from datetime import datetime
from pathlib import Path

import structlog

from semantic_code_mcp.config import Settings, get_index_path
from semantic_code_mcp.indexer.chunker import PythonChunker
from semantic_code_mcp.indexer.embedder import Embedder
from semantic_code_mcp.models import (
    Chunk,
    ChunkWithEmbedding,
    IndexProgress,
    IndexResult,
    IndexStatus,
)
from semantic_code_mcp.storage.cache import FileChangeCache
from semantic_code_mcp.storage.lancedb import VectorStore

log = structlog.get_logger()


class Indexer:
    """Orchestrates the indexing pipeline: scan, chunk, embed, store."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the indexer.

        Args:
            settings: Application settings.
        """
        self.settings = settings
        self.chunker = PythonChunker()
        self.embedder = Embedder(settings)

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

    async def index(
        self, project_path: Path, force: bool = False
    ) -> AsyncGenerator[IndexProgress | IndexResult]:
        """Index a project's codebase with progress updates.

        Yields IndexProgress updates during the operation, then yields
        the final IndexResult.

        Args:
            project_path: Root directory of the project.
            force: If True, re-index all files regardless of changes.

        Yields:
            IndexProgress updates, then final IndexResult.
        """
        start_time = time.time()
        project_path = project_path.resolve()

        log.info("indexing_started", project=str(project_path), force=force)

        yield IndexProgress(stage="init", message="Starting indexing...", percent=0)

        # Get index storage path
        index_path = get_index_path(self.settings, project_path)
        index_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        store = VectorStore(index_path)
        cache = FileChangeCache(index_path)

        yield IndexProgress(stage="scan", message="Scanning files...", percent=5)

        # Scan for Python files (run in thread to not block)
        current_files = await asyncio.to_thread(self.scan_files, project_path)

        yield IndexProgress(
            stage="scan", message=f"Found {len(current_files)} Python files", percent=10
        )

        # Determine which files need indexing
        if force:
            files_to_index = current_files
            files_to_delete: list[str] = []
            cache.clear()
            store.clear()
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

        if not files_to_index and not files_to_delete:
            yield IndexProgress(stage="done", message="No changes to index", percent=100)
            yield IndexResult(
                files_indexed=0,
                chunks_indexed=0,
                files_deleted=0,
                duration_seconds=time.time() - start_time,
            )
            return

        yield IndexProgress(
            stage="prepare",
            message=f"Indexing {len(files_to_index)} files...",
            percent=15,
        )

        # Delete chunks for removed/modified files
        for file_path in files_to_delete:
            store.delete_by_file(file_path)
            cache.remove_file(file_path)

        if not force:
            for file_path in files_to_index:
                store.delete_by_file(file_path)

        yield IndexProgress(stage="chunk", message="Chunking files...", percent=20)

        # Chunk files in parallel batches
        all_chunks: list[Chunk] = []
        batch_size = 20
        total_files = len(files_to_index)

        t0 = time.time()
        for batch_start in range(0, total_files, batch_size):
            batch_end = min(batch_start + batch_size, total_files)
            batch = files_to_index[batch_start:batch_end]

            # Process batch in parallel
            chunk_tasks = [
                asyncio.to_thread(self.chunker.chunk_file, file_path) for file_path in batch
            ]
            batch_results = await asyncio.gather(*chunk_tasks)
            for chunks in batch_results:
                all_chunks.extend(chunks)

            # Yield progress after each batch
            percent = 20 + int(30 * batch_end / total_files)
            yield IndexProgress(
                stage="chunk",
                message=f"Chunked {batch_end}/{total_files} files ({len(all_chunks)} chunks)",
                percent=percent,
            )

        log.debug(
            "chunking_completed",
            files=len(files_to_index),
            chunks=len(all_chunks),
            duration_ms=round((time.time() - t0) * 1000, 1),
        )

        # Embed chunks
        if all_chunks:
            yield IndexProgress(
                stage="embed",
                message=f"Embedding {len(all_chunks)} chunks...",
                percent=55,
            )

            # Load model if needed (this is the slow part on first run)
            if not self.embedder.is_loaded:
                yield IndexProgress(stage="embed", message="Loading embedding model...", percent=55)
                t0 = time.time()
                await asyncio.to_thread(self.embedder.load)
                log.debug("model_loaded", duration_ms=round((time.time() - t0) * 1000, 1))

            yield IndexProgress(
                stage="embed",
                message=f"Generating embeddings for {len(all_chunks)} chunks...",
                percent=60,
            )

            # Generate embeddings (run in thread as it's CPU-bound)
            contents = [chunk.content for chunk in all_chunks]
            t0 = time.time()
            embeddings = await asyncio.to_thread(self.embedder.embed_batch, contents)
            log.debug(
                "embedding_completed",
                chunks=len(all_chunks),
                duration_ms=round((time.time() - t0) * 1000, 1),
            )

            yield IndexProgress(stage="store", message="Storing in database...", percent=85)

            # Create ChunkWithEmbedding objects
            items = [
                ChunkWithEmbedding(chunk=chunk, embedding=embedding)
                for chunk, embedding in zip(all_chunks, embeddings, strict=True)
            ]

            # Store in vector database
            t0 = time.time()
            await asyncio.to_thread(store.add_chunks, items)
            log.debug(
                "storage_completed",
                chunks=len(items),
                duration_ms=round((time.time() - t0) * 1000, 1),
            )

        yield IndexProgress(stage="finalize", message="Finalizing...", percent=95)

        # Update cache with indexed files
        cache.update_files(files_to_index)

        duration = time.time() - start_time
        result = IndexResult(
            files_indexed=len(files_to_index),
            chunks_indexed=len(all_chunks),
            files_deleted=len(files_to_delete),
            duration_seconds=duration,
        )

        log.info(
            "indexing_completed",
            files_indexed=result.files_indexed,
            chunks_indexed=result.chunks_indexed,
            duration=f"{duration:.2f}s",
        )

        yield IndexProgress(stage="done", message="Indexing complete!", percent=100)
        yield result

    def get_status(self, project_path: Path) -> IndexStatus:
        """Get the index status for a project.

        Args:
            project_path: Root directory of the project.

        Returns:
            IndexStatus with current state information.
        """
        project_path = project_path.resolve()
        index_path = get_index_path(self.settings, project_path)

        # Check if index exists
        if not index_path.exists():
            return IndexStatus(
                is_indexed=False,
                last_updated=None,
                files_count=0,
                chunks_count=0,
                stale_files=[],
            )

        store = VectorStore(index_path)
        cache = FileChangeCache(index_path)

        # Get current files and check for staleness
        current_files = self.scan_files(project_path)
        stale_files = cache.get_stale_files(current_files)

        # Get indexed file list
        indexed_files = store.get_indexed_files()
        chunks_count = store.count()

        # Determine last updated from cache file mtime
        cache_file = index_path / "file_mtimes.json"
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
