"""File change detection cache for incremental indexing."""

import json
import os
import tempfile
from pathlib import Path

import structlog

from semantic_code_mcp.domain.models import FileChanges

log = structlog.get_logger()

CACHE_FILENAME = "file_mtimes.json"
# Written beside the cache file, then renamed over it, so a crash never leaves half a file
TEMPORARY_SUFFIX = ".tmp"


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
            self._mtimes = json.loads(self.cache_path.read_text())
            log.debug("cache_loaded", files_count=len(self._mtimes))
        except (json.JSONDecodeError, OSError) as error:
            log.warning("cache_load_failed", error=str(error))
            self._mtimes = {}

    def _save(self) -> None:
        """Save cached mtimes to disk atomically (write to temp, then rename)."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(dir=self.cache_dir, suffix=TEMPORARY_SUFFIX)
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w") as temporary_file:
                json.dump(self._mtimes, temporary_file)
            temporary_path.replace(self.cache_path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
        log.debug("cache_saved", files_count=len(self._mtimes))

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

        deleted_files = [file_path for file_path in cached_set if file_path not in current_set]

        return FileChanges(new=new_files, modified=modified_files, deleted=deleted_files)

    def get_stale_files(self, current_files: list[str]) -> list[str]:
        """Get list of files that need re-indexing.

        Args:
            current_files: List of file paths that currently exist.

        Returns:
            List of file paths that are new or modified.
        """
        return self.get_changes(current_files).stale_files
