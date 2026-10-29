"""A project's files on disk now, for its stale check and its index runs: scanned each time, or
followed by inotify between full scans on Linux (`followed_files.py`, ADR 010, O1).

A project is scanned at each look where no watch can tell what a scan would: off Linux, on a
file system whose changes another machine makes, outside a repository's own root (a directory
no repository holds, or one below a repository's root, where directories outside the project
decide what git lists), with submodules, whose own indexes decide their files, and past
inotify's limits, for the rest of the session. A following that loses the files' thread starts
again from a full scan at the same look.
"""

import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import override

import structlog

from semantic_code_mcp.discovery import Repository, SourceDiscovery
from semantic_code_mcp.domain.interfaces import ProjectFiles
from semantic_code_mcp.domain.models import FileStamps
from semantic_code_mcp.services.followed_files import (
    SUBMODULES_FILE,
    FollowedFiles,
    FollowingLostError,
)
from semantic_code_mcp.services.timing import elapsed_milliseconds
from semantic_code_mcp.watching import Inotify, sees_every_change

log = structlog.get_logger()

# Makes the inotify instance a project is followed with
type InotifyOpener = Callable[[], Inotify]


class ScannedFiles(ProjectFiles):
    """A project's files found by a full scan each time they are asked for."""

    def __init__(self, discovery: SourceDiscovery) -> None:
        self._discovery = discovery

    @override
    def current(self, project_root: Path) -> FileStamps:
        return self._discovery.discover(project_root)

    @override
    def close(self) -> None:
        """Nothing follows the files between scans."""


class WatchedFiles(ProjectFiles):
    """A project's files followed by inotify where a watch can tell what a scan would, else
    scanned each time. Its lock keeps two looks, and what they follow, one after another."""

    def __init__(self, discovery: SourceDiscovery, open_inotify: InotifyOpener) -> None:
        self._discovery = discovery
        self._open_inotify = open_inotify
        self._lock = threading.Lock()
        self._followed: FollowedFiles | None = None
        # Set where following met the system's limits, as of inotify's watches: the project is
        # scanned at each look for the rest of the session
        self._limit_reached = False

    @override
    def current(self, project_root: Path) -> FileStamps:
        root = project_root.resolve()
        with self._lock:
            if self._followed is not None and self._followed.root == root:
                try:
                    return self._followed.look()
                except FollowingLostError as reason:
                    log.info("files_followed_again", project=str(root), reason=str(reason))
                except OSError as error:
                    self._give_up(root, error)
            return self._started(root)

    @override
    def close(self) -> None:
        with self._lock:
            self._stop_following()

    def _started(self, root: Path) -> FileStamps:
        """The files by a full scan, a following started first where one can tell them."""
        self._stop_following()
        repository = self._discovery.listing_repository(root)
        if self._limit_reached or not self._can_follow(root, repository):
            return self._discovery.scan(root, repository)
        started = time.perf_counter()
        try:
            inotify = self._open_inotify()
        except OSError as error:
            self._give_up(root, error)
            return self._discovery.scan(root, repository)
        try:
            self._followed = FollowedFiles.started(
                root=root, repository=repository, discovery=self._discovery, inotify=inotify
            )
        except FollowingLostError as reason:
            inotify.close()
            log.info("files_not_followed", project=str(root), reason=str(reason))
            return self._discovery.scan(root, repository)
        except OSError as error:
            inotify.close()
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
        ask, or git's at its repository's own root without submodules; on a file system whose
        every change inotify sees."""
        if self._discovery.use_gitignore and (
            repository is None or repository.root != root or os.path.lexists(root / SUBMODULES_FILE)
        ):
            return False
        return sees_every_change(root)

    def _give_up(self, root: Path, error: OSError) -> None:
        """Scan the project at each look from now: following met the system's limits."""
        self._stop_following()
        self._limit_reached = True
        log.warning("files_not_followed_past_a_limit", project=str(root), error=str(error))

    def _stop_following(self) -> None:
        if self._followed is not None:
            self._followed.close()
            self._followed = None
