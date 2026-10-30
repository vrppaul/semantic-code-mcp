"""Directories watched one by one within trees a system watches whole (ADR 010, O1).

macOS's FSEvents and Windows' ReadDirectoryChangesW each tell every change below a directory,
by its path, where a project's files are followed one directory at a time, as inotify watches
them (`services/followed_files.py`). A watcher of trees tells each change as inotify would: by
the directory watched that holds it, and its name.

A tree is opened for a directory the first time one no tree holds is watched, and lives until
the watcher closes: one for the project's root, and one for each of git's directories outside
it. A tree opened above another takes its place, the other's changes handed out first; as it
is opened before the other is closed, no change goes untold. A change below no directory
watched is passed over, as no inotify watch would have told it.

The system says little of what it saw, and some of it folded together: FSEvents gives one set
of flags for several changes of one path, a directory's or a file's, and Windows names no kind
at all. So what lies at the path now says what changed there:
- an entry made, removed or renamed is what lay at its path gone, a directory watched there or a
  file, and what lies there now added, a directory walked again from its top: a file's name may
  have been taken by a directory, or a directory's by a file. What lies there is what its
  directory lists by that very spelling, as a volume that ignores case finds `Src` by `src`
  once `src` was renamed `Src`;
- an entry written is a file changed; a directory's own writing, as its entries change, is
  passed over, as their own changes tell them;
- an entry's attributes changed is a file changed, or a directory whose permissions may change
  what may be read below it, walked again.

The system may spell a path otherwise than it was watched by: another case of its letters on a
volume that ignores case, or through a firmlink of macOS. A path is matched to the directory
watched that holds it by its text, and else by where that directory lies, its device and inode.
"""

import errno
import os
import stat
from abc import ABC, abstractmethod
from enum import StrEnum, auto
from pathlib import Path
from typing import override

from pydantic import BaseModel, ConfigDict

from semantic_code_mcp.domain.interfaces import DirectoryWatcher
from semantic_code_mcp.domain.models import NO_WATCH, Place, WatchChange, WatchEvent

# The number of the first watch, as inotify gives it
FIRST_WATCH = 1
# The name of an event of the directory itself, or of changes lost
NO_NAME = ""


class TreeAction(StrEnum):
    """What the system says happened at a path of a tree it watches."""

    # An entry made, removed, or renamed from the path or to it; or several of these at once
    made_or_removed = auto()
    # An entry's content written
    written = auto()
    # An entry's attributes changed, as its permissions or its times
    attributes_changed = auto()
    # Changes were lost, or the tree itself moved: only a fresh start can tell the files
    lost = auto()


class TreeChange(BaseModel):
    """One change the system tells of a tree it watches."""

    model_config = ConfigDict(frozen=True)

    # Where, absolute, as the system spells it; empty for changes lost
    path: str
    action: TreeAction


# What a tree tells where it lost changes
CHANGES_LOST = TreeChange(path=NO_NAME, action=TreeAction.lost)


class Tree(ABC):
    """A directory the system watches whole, with every one below it."""

    def __init__(self, directory: str) -> None:
        # As it was watched by
        self.directory = directory

    @abstractmethod
    def changes(self) -> list[TreeChange]:
        """The changes since the last call, in their order: every change made before the call
        is among them."""

    @abstractmethod
    def close(self) -> None:
        """Stop watching, and let go of what the system holds for it."""


class WatchedDirectory(BaseModel):
    """A directory watched: its path, as it was watched by, and where it lay then."""

    model_config = ConfigDict(frozen=True)

    path: str
    place: Place


def is_plain_directory(path: str, status: os.stat_result) -> bool:
    """Whether what a path's own `lstat` read is a directory itself: no symlink, nor on Windows a
    junction, whose `lstat` reads as a plain directory."""
    return stat.S_ISDIR(status.st_mode) and not os.path.isjunction(path)


def is_plain_directory_now(path: str) -> bool:
    try:
        status = os.lstat(path)
    except OSError:
        return False
    return is_plain_directory(path, status)


def lies_in(path: str, directory: str) -> bool:
    """Whether a path is a directory or lies below it, both spelled alike."""
    return path == directory or path.startswith(directory.rstrip(os.sep) + os.sep)


def entry_event(watch: int, change: WatchChange, name: str, *, is_directory: bool) -> WatchEvent:
    return WatchEvent(watch=watch, change=change, name=name, is_directory=is_directory)


class LookCache(BaseModel):
    """What one look of a watcher of trees has read so far: the watch holding each directory
    met, by the system's spelling of it, and the names each directory watched lists."""

    holding: dict[str, int | None]
    names: dict[int, frozenset[str]]


class TreeWatcher(DirectoryWatcher):
    """Directories watched each on its own within trees the system watches whole, each opened by
    `open_tree`."""

    def __init__(self) -> None:
        self._trees: list[Tree] = []
        # What trees closed as a tree above them took their place told, handed out next
        self._untold: list[TreeChange] = []
        self._watched: dict[int, WatchedDirectory] = {}
        # Each directory watched by its path as the file system compares names, and by its place
        self._by_path: dict[str, int] = {}
        self._by_place: dict[Place, int] = {}
        self._next_watch = FIRST_WATCH

    @abstractmethod
    def open_tree(self, directory: str) -> Tree:
        """Watch a directory and every one below it, from the moment this returns. Raises the
        OSError the system gives."""

    @override
    def watch(self, directory: str) -> int:
        status = os.lstat(directory)
        if not is_plain_directory(directory, status):
            raise NotADirectoryError(errno.ENOTDIR, os.strerror(errno.ENOTDIR), directory)
        # As inotify, which watches only what it may read
        if not os.access(directory, os.R_OK):
            raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), directory)
        key = os.path.normcase(directory)
        known = self._by_path.get(key)
        if known is not None:
            return known
        self._hold_in_a_tree(directory)
        watch = self._next_watch
        self._next_watch += 1
        place = Place.of(status)
        self._watched[watch] = WatchedDirectory(path=directory, place=place)
        self._by_path[key] = watch
        self._by_place[place] = watch
        return watch

    @override
    def unwatch(self, watch: int) -> None:
        watched = self._watched.pop(watch, None)
        if watched is None:
            return
        key = os.path.normcase(watched.path)
        if self._by_path.get(key) == watch:
            del self._by_path[key]
        if self._by_place.get(watched.place) == watch:
            del self._by_place[watched.place]

    @override
    def events(self) -> list[WatchEvent]:
        changes = self._untold
        self._untold = []
        for tree in self._trees:
            changes += tree.changes()
        look = LookCache(holding={}, names={})
        events: list[WatchEvent] = []
        for change in changes:
            events += self._told(change, look)
        return events

    @override
    def close(self) -> None:
        for tree in self._trees:
            tree.close()
        self._trees = []

    def _hold_in_a_tree(self, directory: str) -> None:
        """Open a tree for a directory no tree holds; the trees below it closed once it is
        open, their changes kept."""
        key = os.path.normcase(directory)
        if any(lies_in(key, os.path.normcase(tree.directory)) for tree in self._trees):
            return
        opened = self.open_tree(directory)
        below = [tree for tree in self._trees if lies_in(os.path.normcase(tree.directory), key)]
        for tree in below:
            self._untold += tree.changes()
            tree.close()
        self._trees = [tree for tree in self._trees if tree not in below] + [opened]

    def _told(self, change: TreeChange, look: LookCache) -> list[WatchEvent]:
        """A change as inotify would tell it: none below no directory watched."""
        if change.action is TreeAction.lost:
            return [entry_event(NO_WATCH, WatchChange.changes_lost, NO_NAME, is_directory=False)]
        parent, name = os.path.split(os.path.normpath(change.path))
        if parent not in look.holding:
            look.holding[parent] = self._watch_holding(parent)
        watch = look.holding[parent]
        if watch is None:
            return []
        path = f"{self._watched[watch].path}{os.sep}{name}"
        directory_now = is_plain_directory_now(path)
        match change.action:
            case TreeAction.made_or_removed:
                if watch not in look.names:
                    look.names[watch] = self._names_in(watch)
                lies_there = name in look.names[watch]
                return self._made_or_removed(
                    watch, name, path, lies_there=lies_there, directory_now=directory_now
                )
            case TreeAction.written if directory_now:
                return []
            case _:
                return [
                    entry_event(watch, WatchChange.entry_changed, name, is_directory=directory_now)
                ]

    def _made_or_removed(
        self, watch: int, name: str, path: str, *, lies_there: bool, directory_now: bool
    ) -> list[WatchEvent]:
        """What lay at the path gone, a directory watched or a file, and what lies there now
        added: a file's name may have been taken by a directory, or a directory's by a file."""
        watched = os.path.normcase(path) in self._by_path
        gone = [entry_event(watch, WatchChange.entry_removed, name, is_directory=watched)]
        if not lies_there:
            return gone
        return [
            *gone,
            entry_event(watch, WatchChange.entry_added, name, is_directory=directory_now),
        ]

    def _names_in(self, watch: int) -> frozenset[str]:
        """The names a directory watched lists now, each as it is spelled there: none where it
        went or may no longer be read."""
        try:
            with os.scandir(self._watched[watch].path) as entries:
                return frozenset(entry.name for entry in entries)
        except OSError:
            return frozenset()

    def _watch_holding(self, directory: str) -> int | None:
        """The watch of a directory as the system spells it: by its text, else by where it lies;
        None where no directory watched lies there."""
        watch = self._by_path.get(os.path.normcase(directory))
        if watch is not None:
            return watch
        try:
            status = Path(directory).stat()
        except OSError:
            return None
        return self._by_place.get(Place.of(status))
