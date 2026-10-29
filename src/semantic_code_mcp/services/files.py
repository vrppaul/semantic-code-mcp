"""A project's files on disk now, for its stale check and its index runs: scanned each time, or
followed by the system's watcher between full scans (`followed_files.py`, ADR 010, O1): inotify
on Linux, FSEvents on macOS, ReadDirectoryChangesW on Windows.

A project is scanned at each look where no watch can tell what a scan would: on a system with no
watcher, on a file system whose changes another machine makes, outside a repository's own root
(a directory no repository holds, or one below a repository's root, where directories outside
the project decide what git lists), with submodules, whose own indexes decide their files, and
past the watcher's limits or wherever the system fails it, for the rest of the session. A
following that loses the files' thread starts again from a full scan at the same look.

Every CHECK_INTERVAL_SECONDS a look is checked against a full scan: what no event can tell (the
user's global git configuration and excludes file, a file system mounted over a directory
followed) is then seen within that time, and a difference is logged as a warning, the following
started again from the scan. A forced index run starts the following again too.
"""

import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import override

import structlog

from semantic_code_mcp.discovery import Repository, SourceDiscovery
from semantic_code_mcp.domain.interfaces import DirectoryWatcher, ProjectFiles
from semantic_code_mcp.domain.models import FileStamps
from semantic_code_mcp.services.followed_files import FollowedFiles, FollowingLostError
from semantic_code_mcp.services.timing import elapsed_milliseconds
from semantic_code_mcp.watching.systems import sees_every_change

log = structlog.get_logger()

# Makes the watcher a project is followed with
type WatcherOpener = Callable[[], DirectoryWatcher]
# The time in seconds on a clock that only moves on, as `time.monotonic` gives it
type Clock = Callable[[], float]

# The file that names a repository's submodules, whose own indexes decide their files
SUBMODULES_FILE = ".gitmodules"
# How long the files followed go without a full scan to check them against
CHECK_INTERVAL_SECONDS = 300.0
# How many of the paths that differ a warning names
PATHS_NAMED = 5


class ScannedFiles(ProjectFiles):
    """A project's files found by a full scan each time they are asked for."""

    def __init__(self, discovery: SourceDiscovery) -> None:
        self._discovery = discovery

    @override
    def current(self, project_root: Path) -> FileStamps:
        return self._discovery.discover(project_root)

    @override
    def rescanned(self, project_root: Path) -> FileStamps:
        return self._discovery.discover(project_root)

    @override
    def close(self) -> None:
        """Nothing follows the files between scans."""


class WatchedFiles(ProjectFiles):
    """A project's files followed by the system's watcher where a watch can tell what a scan
    would, else scanned each time. Its lock keeps two looks, and what they follow, one after
    another."""

    def __init__(
        self, discovery: SourceDiscovery, open_watcher: WatcherOpener, clock: Clock
    ) -> None:
        self._discovery = discovery
        self._open_watcher = open_watcher
        self._clock = clock
        self._lock = threading.Lock()
        self._followed: FollowedFiles | None = None
        self._next_check_at = 0.0
        # Set where following failed on the system's side, as at inotify's limit of watches:
        # the project is scanned at each look for the rest of the session
        self._following_failed = False

    @override
    def current(self, project_root: Path) -> FileStamps:
        root = project_root.resolve()
        with self._lock:
            if self._followed is not None and self._followed.root == root:
                try:
                    return self._checked(root, self._followed.look())
                except FollowingLostError as reason:
                    log.info("files_followed_again", project=str(root), reason=str(reason))
                except OSError as error:
                    self._give_up(root, error)
            return self._started(root)

    @override
    def rescanned(self, project_root: Path) -> FileStamps:
        with self._lock:
            return self._started(project_root.resolve())

    @override
    def close(self) -> None:
        with self._lock:
            self._stop_following()

    def _checked(self, root: Path, files: FileStamps) -> FileStamps:
        """The files a look read, checked against a full scan once the interval has passed:
        where they differ, the following starts again. Raises `FollowingLostError` then."""
        if self._clock() < self._next_check_at:
            return files
        self._next_check_at = self._clock() + CHECK_INTERVAL_SECONDS
        scanned = self._discovery.discover(root)
        if scanned == files:
            return files
        differing = sorted(
            path for path in files.keys() | scanned.keys() if files.get(path) != scanned.get(path)
        )
        log.warning(
            "files_followed_differ_from_a_scan",
            project=str(root),
            paths=len(differing),
            first_paths=differing[:PATHS_NAMED],
        )
        message = "the files followed differ from a scan"
        raise FollowingLostError(message)

    def _started(self, root: Path) -> FileStamps:
        """The files by a full scan, a following started first where one can tell them."""
        self._stop_following()
        self._next_check_at = self._clock() + CHECK_INTERVAL_SECONDS
        repository = self._discovery.listing_repository(root)
        if self._following_failed or not self._can_follow(root, repository):
            return self._discovery.scan(root, repository)
        started = time.perf_counter()
        try:
            watcher = self._open_watcher()
        except OSError as error:
            self._give_up(root, error)
            return self._discovery.scan(root, repository)
        try:
            self._followed = FollowedFiles.started(
                root=root, repository=repository, discovery=self._discovery, watcher=watcher
            )
        except FollowingLostError as reason:
            watcher.close()
            log.info("files_not_followed", project=str(root), reason=str(reason))
            return self._discovery.scan(root, repository)
        except OSError as error:
            watcher.close()
            self._give_up(root, error)
            return self._discovery.scan(root, repository)
        log.debug(
            "files_followed",
            project=str(root),
            files=len(self._followed.files),
            duration_ms=elapsed_milliseconds(started),
        )
        return self._followed.files

    def _can_follow(self, root: Path, repository: Repository | None) -> bool:
        """Whether a watch can tell what a scan of the root would: a walk's, as the settings
        ask, or git's at its repository's own root without submodules; on file systems whose
        every change the system's watcher sees."""
        if self._discovery.use_gitignore and (
            repository is None or repository.root != root or os.path.lexists(root / SUBMODULES_FILE)
        ):
            return False
        return sees_every_change(root)

    def _give_up(self, root: Path, error: OSError) -> None:
        """Scan the project at each look from now: following failed on the system's side."""
        self._stop_following()
        self._following_failed = True
        log.warning("files_not_followed", project=str(root), error=str(error))

    def _stop_following(self) -> None:
        if self._followed is not None:
            self._followed.close()
            self._followed = None
