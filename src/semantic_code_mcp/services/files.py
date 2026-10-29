"""A project's files on disk now, for its stale check and its index runs: scanned each time,
or kept from a watcher's changes between full scans (ADR 010, O1).

A full scan starts a new watcher first, so what changes while it scans is the next read's. It
comes at the first look, when the watcher was lost or the changes ask for one, for a project
of another root, and at least every FULL_SCAN_INTERVAL_SECONDS: a net for what no change the
watcher reports shows, as an excludes file outside the project. A watcher that cannot start
is tried again with that net, the project scanned meanwhile.
"""

import threading
from collections.abc import Callable
from pathlib import Path
from typing import override

import structlog

from semantic_code_mcp.discovery import SourceDiscovery
from semantic_code_mcp.domain.interfaces import ProjectFiles
from semantic_code_mcp.domain.models import FileStamps
from semantic_code_mcp.watching import FileWatcher, WatcherLostError

log = structlog.get_logger()

# How long the files kept from a watcher go without a full scan
FULL_SCAN_INTERVAL_SECONDS = 300.0

# Starts a watcher of a project's root, resolved, and the directories under it to watch
type WatcherOpener = Callable[[Path, list[Path]], FileWatcher]
# The time in seconds on a clock that only moves on, as `time.monotonic` gives it
type Clock = Callable[[], float]


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
    """A project's files as the last full scan found them, with what a watcher saw change
    since read again by the scan's rules. Its lock keeps two reads, and the watcher with
    them, one after another."""

    def __init__(
        self, discovery: SourceDiscovery, open_watcher: WatcherOpener, clock: Clock
    ) -> None:
        self._discovery = discovery
        self._open_watcher = open_watcher
        self._clock = clock
        self._lock = threading.Lock()
        self._watcher: FileWatcher | None = None
        # The root the files and the watcher are of; None before the first scan
        self._root: Path | None = None
        self._files: FileStamps = {}
        self._next_scan_at = 0.0

    @override
    def current(self, project_root: Path) -> FileStamps:
        with self._lock:
            if self._root != project_root or self._clock() >= self._next_scan_at:
                files = self._scanned(project_root)
            elif self._watcher is None:
                files = self._discovery.discover(project_root)
            else:
                files = self._read_from(self._watcher, project_root)
            self._files = files
            # A copy, as the next read changes what this one keeps
            return dict(files)

    @override
    def close(self) -> None:
        with self._lock:
            self._close_watcher()

    def _read_from(self, watcher: FileWatcher, project_root: Path) -> FileStamps:
        """The files kept with what the watcher saw change; a full scan where the watcher was
        lost or the changes ask for one."""
        try:
            changed = watcher.changed()
        except WatcherLostError as error:
            log.warning("file_watcher_lost", project=str(project_root), reason=str(error))
            return self._scanned(project_root)
        refreshed = self._discovery.refresh(project_root, self._files, changed)
        if refreshed is None:
            log.debug("file_changes_need_a_scan", project=str(project_root), paths=len(changed))
            return self._scanned(project_root)
        return refreshed

    def _scanned(self, project_root: Path) -> FileStamps:
        """A full scan, a new watcher started before it."""
        self._close_watcher()
        self._watcher = self._started_watcher(project_root)
        self._root = project_root
        self._next_scan_at = self._clock() + FULL_SCAN_INTERVAL_SECONDS
        return self._discovery.discover(project_root)

    def _started_watcher(self, project_root: Path) -> FileWatcher | None:
        """A watcher of the directories a scan reads; None for a project no watcher can
        follow, or where one cannot start."""
        directories = self._discovery.watched_directories(project_root)
        if directories is None:
            log.debug("project_scanned_without_a_watcher", project=str(project_root))
            return None
        try:
            return self._open_watcher(project_root.resolve(), directories)
        except WatcherLostError as error:
            log.warning("file_watcher_not_started", project=str(project_root), reason=str(error))
            return None

    def _close_watcher(self) -> None:
        if self._watcher is not None:
            self._watcher.close()
            self._watcher = None
