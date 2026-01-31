"""File change detection cache for incremental indexing."""

import json
import tempfile
from pathlib import Path

import structlog

from semantic_code_mcp.models import FileChanges

log = structlog.get_logger()

CACHE_FILENAME = "file_mtimes.json"


class FileChangeCache:
    """Tracks file modification times to detect changes for incremental indexing."""

    def __init__(self, cache_dir: Path) -> None:
        """Initialize the file change cache.

        Args:
            cache_dir: Directory to store the cache file.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_path = self.cache_dir / CACHE_FILENAME
        self._mtimes: dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        """Load cached mtimes from disk."""
        if not self.cache_path.exists():
            log.debug("cache_file_not_found", path=str(self.cache_path))
            return

        try:
            with open(self.cache_path) as f:
                self._mtimes = json.load(f)
            log.debug("cache_loaded", files_count=len(self._mtimes))
        except (json.JSONDecodeError, OSError) as e:
            log.warning("cache_load_failed", error=str(e))
            self._mtimes = {}

    def _save(self) -> None:
        """Save cached mtimes to disk atomically (write to temp, then rename)."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
        try:
            with open(fd, "w") as f:
                json.dump(self._mtimes, f)
            Path(tmp_path).replace(self.cache_path)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise
        log.debug("cache_saved", files_count=len(self._mtimes))

    def get_tracked_files(self) -> list[str]:
        """Get list of all tracked file paths.

        Returns:
            List of file paths being tracked.
        """
        return list(self._mtimes.keys())

    def update_file(self, file_path: str) -> None:
        """Update the cached mtime for a single file.

        Args:
            file_path: Path to the file.
        """
        path = Path(file_path)
        if path.exists():
            self._mtimes[file_path] = path.stat().st_mtime
            self._save()

    def update_files(self, file_paths: list[str]) -> None:
        """Update cached mtimes for multiple files.

        Args:
            file_paths: List of file paths to update.
        """
        for file_path in file_paths:
            path = Path(file_path)
            if path.exists():
                self._mtimes[file_path] = path.stat().st_mtime
        self._save()

    def remove_file(self, file_path: str) -> None:
        """Remove a file from the cache.

        Args:
            file_path: Path to remove.
        """
        self._mtimes.pop(file_path, None)
        self._save()

    def remove_files(self, file_paths: list[str]) -> None:
        """Remove multiple files from the cache.

        Args:
            file_paths: List of file paths to remove.
        """
        for file_path in file_paths:
            self._mtimes.pop(file_path, None)
        self._save()

    def clear(self) -> None:
        """Clear all tracked files from the cache."""
        self._mtimes = {}
        self._save()

    def get_changes(self, current_files: list[str]) -> FileChanges:
        """Compare current files with cached state to find changes.

        Args:
            current_files: List of file paths that currently exist.

        Returns:
            FileChanges with new, modified, and deleted files.
        """
        current_set = set(current_files)
        cached_set = set(self._mtimes.keys())

        new_files: list[str] = []
        modified_files: list[str] = []
        deleted_files: list[str] = []

        # Find new and modified files
        for file_path in current_files:
            if file_path not in cached_set:
                new_files.append(file_path)
            else:
                path = Path(file_path)
                if path.exists():
                    current_mtime = path.stat().st_mtime
                    cached_mtime = self._mtimes[file_path]
                    if current_mtime != cached_mtime:
                        modified_files.append(file_path)

        # Find deleted files
        deleted_files = [f for f in cached_set if f not in current_set]

        return FileChanges(new=new_files, modified=modified_files, deleted=deleted_files)

    def has_changes(self, current_files: list[str]) -> bool:
        """Check if there are any changes without full comparison.

        Args:
            current_files: List of file paths that currently exist.

        Returns:
            True if there are new, modified, or deleted files.
        """
        return self.get_changes(current_files).has_changes

    def get_stale_files(self, current_files: list[str]) -> list[str]:
        """Get list of files that need re-indexing.

        Args:
            current_files: List of file paths that currently exist.

        Returns:
            List of file paths that are new or modified.
        """
        return self.get_changes(current_files).stale_files
