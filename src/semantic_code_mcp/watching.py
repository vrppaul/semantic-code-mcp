"""Watching a project's files, so a search's stale check reads what changed instead of every
file (ADR 010, O1).

notify, under watchfiles, reads the system's queue of file events on a thread of its own, so a
change made just before a search may not have reached it yet. A read is therefore a barrier:
it touches a new marker in a directory the same watcher watches, then takes the watcher's
changes until the marker's comes. One inotify instance keeps one queue, and macOS's stream is
ordered by event, so every change made before the marker is in hand by then. Windows keeps a
queue for each watched directory, ordering nothing across them, and watchfiles polls on WSL,
where a poll may reach the marker before the directory that changed: neither is watched.

Each directory is watched on its own, not below it: a recursive watch takes an inotify watch
for every directory under the root, `.git`'s and `node_modules`' among them, from a limit the
user's editor shares.

The watcher is watchfiles' `RustNotify`, which its public `watch` wraps: that generator makes
its watcher at the first read, after the marker the read touched, so the first marker would
never come; this one is made before any marker is touched.
"""

import os
import platform
import shutil
import sys
import tempfile
import weakref
from pathlib import Path

from watchfiles._rust_notify import RustNotify, WatchfilesRustInternalError

from semantic_code_mcp.domain.models import ProjectPath

# Changes are taken as soon as a step passes with none new: nothing is held back to group them
NO_DEBOUNCE_MILLISECONDS = 0
STEP_MILLISECONDS = 1
# How long a read waits for its marker before the watcher counts as lost; a marker comes in
# about 2 ms
MARKER_DEADLINE_MILLISECONDS = 1_000
# How the watcher is made: events of the system, never a poll, which would order nothing
# against the marker; no debug lines; each directory alone; a directory it may not read left
# out, as a scan cannot read it either
WATCHER_DEBUG = False
WATCHER_POLLS = False
UNUSED_POLL_DELAY_MILLISECONDS = 0
WATCHED_RECURSIVELY = False
PERMISSION_DENIED_IGNORED = True
MARKERS_PREFIX = "semantic-code-mcp-markers-"
MARKER_NAME_FORMAT = "marker-{number}"
# The environment variable that makes watchfiles poll, and the release WSL's kernel names
FORCE_POLLING_VARIABLE = "WATCHFILES_FORCE_POLLING"
WSL_KERNEL_RELEASE = "microsoft-standard"
WINDOWS_PLATFORM = "win32"
PATH_SEPARATOR = "/"


class WatcherLostError(RuntimeError):
    """A watcher that could not start, reported an error, or whose marker did not come in
    time: it can no longer say what changed, and the project is scanned whole."""


def watching_supported() -> bool:
    """Whether a watcher's barrier holds here: not on Windows, nor where watchfiles polls."""
    if sys.platform == WINDOWS_PLATFORM or os.environ.get(FORCE_POLLING_VARIABLE):
        return False
    return WSL_KERNEL_RELEASE not in platform.uname().release.lower()


class FileWatcher:
    """Watches the directories of one project, each on its own, and says which paths under
    its root changed since it was last read.

    Not safe for two threads at once: its reader holds a lock around each read.
    """

    def __init__(self, root: Path, directories: list[Path]) -> None:
        """Start watching the directories, each under the root, which is resolved. Raises
        `WatcherLostError` where the watcher cannot start, as at inotify's limit of watches."""
        self._root_prefix = f"{root}{os.sep}"
        self._markers = Path(tempfile.mkdtemp(prefix=MARKERS_PREFIX)).resolve()
        # Removed when the watcher closes, or is collected, or the process ends without it
        self._remove_markers = weakref.finalize(
            self, shutil.rmtree, self._markers, ignore_errors=True
        )
        self._markers_made = 0
        try:
            self._notify = RustNotify(
                [str(directory) for directory in (*directories, self._markers)],
                WATCHER_DEBUG,
                WATCHER_POLLS,
                UNUSED_POLL_DELAY_MILLISECONDS,
                WATCHED_RECURSIVELY,
                PERMISSION_DENIED_IGNORED,
            )
        except (OSError, WatchfilesRustInternalError) as error:
            self._remove_markers()
            raise WatcherLostError(str(error)) from error
        # A first read shows the watcher answers; what changed while it started is the scan's
        try:
            self.changed()
        except WatcherLostError:
            self.close()
            raise

    def changed(self) -> list[ProjectPath]:
        """The paths under the root that changed since the last read, each from the root as
        POSIX text, sorted; raises `WatcherLostError` where the watcher can no longer say."""
        marker = self._markers / MARKER_NAME_FORMAT.format(number=self._markers_made)
        self._markers_made += 1
        try:
            marker.touch()
        except OSError as error:
            # Its directory gone, as a cleaner of temporary files may take it
            raise WatcherLostError(str(error)) from error
        marker_text = str(marker)
        changed: set[ProjectPath] = set()
        marker_seen = False
        while not marker_seen:
            paths = self._next_changed_paths()
            marker_seen = marker_text in paths
            changed |= {
                path.removeprefix(self._root_prefix).replace(os.sep, PATH_SEPARATOR)
                for path in paths
                if path.startswith(self._root_prefix)
            }
        marker.unlink(missing_ok=True)
        return sorted(changed)

    def close(self) -> None:
        """Stop watching and remove the markers' directory."""
        self._notify.close()
        self._remove_markers()

    def _next_changed_paths(self) -> set[str]:
        """The paths of the watcher's next changes, whatever their kind, as the system names
        them; none within the marker's deadline or an error the watcher reports, and it is
        lost. Its timeout clears what came in its last instant, which a lost watcher's full
        scan reads anyway."""
        try:
            changes = self._notify.watch(
                NO_DEBOUNCE_MILLISECONDS, STEP_MILLISECONDS, MARKER_DEADLINE_MILLISECONDS, None
            )
        except WatchfilesRustInternalError as error:
            raise WatcherLostError(str(error)) from error
        if isinstance(changes, str):
            message = f"no marker within {MARKER_DEADLINE_MILLISECONDS} ms: {changes}"
            raise WatcherLostError(message)
        return {changed_path for _, changed_path in changes}
