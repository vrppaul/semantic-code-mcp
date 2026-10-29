"""A project's files followed by inotify since a full scan, on Linux (ADR 010, O1): what each
look reads of what changed, so it finds what a scan would.

A followed project keeps what git listed and each listed file's record as last read, and at each
look reads inotify's queue to its end, which holds every change made before the look. A path
that changed is read again by the scan's rules; a new file git has not listed is asked of `git
check-ignore`; git's index rewritten lists the files again, git's part of a scan alone; a new
directory is watched, then listed, on the live instance, so nothing made meanwhile goes unseen;
one gone has its watches dropped and its files read again; a nested repository is looked into
only where git tracks files below it, as git does. What changes the rules themselves (a
`.gitignore`, an excludes file, a repository made or gone, submodules, the root or git's own
directories no longer where they were) and a queue that overflowed end the following: only a
fresh start and a full scan can tell the files then. What no watch covers is read at every look:
a symlink, whose target may lie anywhere, and a file git lists in a directory nothing watches
(one git ignores though it tracks the file, one a nested repository holds, one the server may
not read). The user's global git configuration and excludes file are read as a following starts.
"""

import os
from collections.abc import Collection, Iterable
from enum import StrEnum, auto
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from semantic_code_mcp.discovery import (
    PATH_SEPARATOR,
    ROOT_DIRECTORY,
    CandidateCheck,
    FoundFile,
    FoundFiles,
    Repository,
    SourceDiscovery,
    is_utf8,
    joined_path,
    kept_once,
)
from semantic_code_mcp.domain.models import FileStamps
from semantic_code_mcp.watching import Change, Inotify, WatchEvent

# Where git keeps its work, in a work tree or as a nested repository's mark; the files whose
# change can change which files git lists; and those of git's own directories that matter here
GIT_DIRECTORY_NAME = ".git"
GITIGNORE_NAME = ".gitignore"
SUBMODULES_FILE = ".gitmodules"
INDEX_NAME = "index"
INFO_DIRECTORY_NAME = "info"
EXCLUDE_NAME = "exclude"

# More new entries than this at one look are told apart by git's listing again, not asked of
# `git check-ignore` one by one: on Sentry it takes 0.3 ms a path, matched against 102
# patterns, and a listing 40 ms
MOST_ENTRIES_ASKED_OF_GIT = 100


class DirectoryRole(StrEnum):
    """Why a directory is watched, which says what its events mean."""

    # A directory a scan reads: a change of any entry is read again
    work_tree = auto()
    # A repository nested in the project, which git does not look into: only its `.git` counts
    nested_repository = auto()
    # Git's own directory, with the index
    git_directory = auto()
    # The directory a linked worktree's repository shares, with `info`
    common_directory = auto()
    # `info`, with the excludes file
    info_directory = auto()


# The entries of git's directories whose change lists the files again
NAMES_LISTING_AGAIN = {DirectoryRole.git_directory: frozenset({INDEX_NAME})}
# Those whose change starts the following again; a main work tree's git directory is its common
# directory too
NAMES_STARTING_AGAIN = {
    DirectoryRole.git_directory: frozenset({INFO_DIRECTORY_NAME}),
    DirectoryRole.common_directory: frozenset({INFO_DIRECTORY_NAME}),
    DirectoryRole.info_directory: frozenset({EXCLUDE_NAME}),
    DirectoryRole.nested_repository: frozenset({GIT_DIRECTORY_NAME}),
}
# The names in a work tree whose change starts the following again in a repository: a nested
# repository made or gone, or the project's own; the rules of what git ignores
WORK_TREE_NAMES_STARTING_AGAIN = frozenset({GIT_DIRECTORY_NAME, GITIGNORE_NAME})
# The names at the root itself that do so too: submodules, whose own indexes decide their files
ROOT_NAMES_STARTING_AGAIN = frozenset({SUBMODULES_FILE})
NO_NAMES = frozenset[str]()


class FollowingLostError(RuntimeError):
    """The files kept can no longer be trusted, as the queue overflowed, the root went, git
    failed or a rule changed: only a full scan can tell them."""


class WatchedDirectory(BaseModel):
    """A directory watched: why, and where, from the root for one in the work tree, else as
    the file system spells it."""

    model_config = ConfigDict(frozen=True)

    role: DirectoryRole
    path: str


class Place(BaseModel):
    """Where a directory lies: its device and inode, which its path keeps while nothing moves
    it or a directory above it."""

    model_config = ConfigDict(frozen=True)

    device: int
    inode: int


def place_of(directory: Path) -> Place | None:
    """Where the directory at a path lies now; None where none does."""
    try:
        status = directory.stat()
    except OSError:
        return None
    return Place(device=status.st_dev, inode=status.st_ino)


class LookChanges(BaseModel):
    """What the events of one look changed: the paths to read again, the directories to watch
    and list, and whether the directories followed or git's listing changed."""

    paths: set[str]
    directories: set[str]
    directories_changed: bool
    list_again: bool


def parent_of(path: str) -> str:
    """The directory a path lies in, from the root."""
    return path.rpartition(PATH_SEPARATOR)[0]


def lies_under(path: str, directory: str) -> bool:
    """Whether a path is a directory or lies below it, both from the root."""
    return path == directory or path.startswith(directory + PATH_SEPARATOR)


class FollowedFiles:
    """A project's files kept from inotify's events since a full scan: the work tree's
    directories watched each on its own, git's directories for what decides its listing, and
    the files git lists, each with its record as last read.

    Built by `started`, which watches, then scans; not safe for two threads at once.
    """

    def __init__(
        self,
        *,
        root: Path,
        repository: Repository | None,
        discovery: SourceDiscovery,
        inotify: Inotify,
    ) -> None:
        self.root = root
        # The root as text, which a path from it joins without a `Path` built per path
        self._root_text = str(root)
        # None where a walk lists the files, as the settings ask
        self._repository = repository
        self._discovery = discovery
        self._inotify = inotify
        self._rules = discovery.ignore_rules()
        self._directories: dict[int, WatchedDirectory] = {}
        # The work tree's directories followed, by their paths from the root
        self._watches: dict[str, int] = {}
        # What git lists, by the paths from the root; empty where a walk lists every file of a
        # directory followed
        self._listed: set[str] = set()
        self._found: FoundFiles = {}
        # The paths listed whose own path is a symlink
        self._links: set[str] = set()
        self._polled: frozenset[str] = frozenset()
        self._files: FileStamps = {}
        # Where the root and git's directories lay as the following started, by their paths: a
        # directory above one moved sends its watches nothing
        self._places: dict[Path, Place] = {}
        # The directories below the root holding a repository of their own, and those of them
        # git looks into, as it tracks files below them
        self._repositories_within: set[str] = set()
        self._looked_into: set[str] = set()

    @classmethod
    def started(
        cls,
        *,
        root: Path,
        repository: Repository | None,
        discovery: SourceDiscovery,
        inotify: Inotify,
    ) -> FollowedFiles:
        """A project followed from now: every directory watched before it is listed, then the
        files listed and read, so what changes meanwhile is the first look's. Raises
        `FollowingLostError` where it cannot be followed, and the OSError of a limit reached."""
        followed = cls(root=root, repository=repository, discovery=discovery, inotify=inotify)
        followed._begin()
        return followed

    @property
    def files(self) -> FileStamps:
        """The files as the last look read them."""
        return self._files

    def look(self) -> FileStamps:
        """The files now, with what changed since the last look read again. Raises
        `FollowingLostError` where only a full scan can tell, and the OSError of a limit
        reached."""
        for directory, place in self._places.items():
            if place_of(directory) != place:
                message = f"{directory} is no longer where the following found it"
                raise FollowingLostError(message)
        events = self._inotify.events()
        changes = LookChanges(
            paths=set(), directories=set(), directories_changed=False, list_again=False
        )
        for event in events:
            self._take(event, changes)
        if changes.directories:
            watched = len(self._directories)
            changes.paths |= self._follow(changes.directories)
            changes.directories_changed |= len(self._directories) != watched
        if changes.list_again:
            changes.paths |= self._list_again()
        paths = changes.paths | self._polled
        links = set(self._links)
        if paths and self._read_again(paths, changes):
            self._files = kept_once(self._found.values())
        if changes.directories_changed or changes.list_again or links != self._links:
            self._polled = self._polled_paths()
        return self._files

    def close(self) -> None:
        """Stop watching."""
        self._inotify.close()

    def _begin(self) -> None:
        self._hold_place(self.root)
        if self._repository is not None:
            self._hold_place(self._repository.git_directory)
            self._hold_place(self._repository.common_directory)
            self._watch_git_directories(self._repository)
        files = self._follow([ROOT_DIRECTORY])
        if ROOT_DIRECTORY not in self._watches:
            message = "the project's root cannot be watched"
            raise FollowingLostError(message)
        candidates: Iterable[str] = files
        if self._repository is not None:
            self._listed = self._git_listing()
            candidates = self._listed
        check = CandidateCheck(self.root)
        self._found = self._discovery.found_files(candidates, self._rules, check)
        self._links = set(check.links_met)
        self._polled = self._polled_paths()
        self._files = kept_once(self._found.values())

    def _hold_place(self, directory: Path) -> None:
        """Note where a directory lies, so each look finds it there."""
        place = place_of(directory)
        if place is None:
            message = f"{directory} is gone"
            raise FollowingLostError(message)
        self._places[directory] = place

    def _watch_git_directories(self, repository: Repository) -> None:
        """Watch where git keeps its index and excludes; its `info` only where it is, as its
        making is seen in the directory that holds it."""
        self._watch_git(repository.git_directory, DirectoryRole.git_directory)
        if repository.common_directory != repository.git_directory:
            self._watch_git(repository.common_directory, DirectoryRole.common_directory)
        info = repository.common_directory / INFO_DIRECTORY_NAME
        if info.is_dir():
            self._watch_git(info, DirectoryRole.info_directory)

    def _watch_git(self, directory: Path, role: DirectoryRole) -> None:
        try:
            watch = self._inotify.watch(str(directory))
        except (FileNotFoundError, NotADirectoryError, PermissionError) as error:
            message = f"git's directory {directory} cannot be watched: {error}"
            raise FollowingLostError(message) from error
        self._directories[watch] = WatchedDirectory(role=role, path=str(directory))

    def _take(self, event: WatchEvent, changes: LookChanges) -> None:
        """Note what one event changed."""
        if event.change is Change.queue_overflowed:
            message = "inotify's queue overflowed"
            raise FollowingLostError(message)
        # A watch dropped since, whose directory a walk lists again where it still lies:
        # inotify gives no watch's number twice until it wraps round
        directory = self._directories.get(event.watch)
        if directory is None:
            return
        if directory.role is DirectoryRole.work_tree:
            self._take_from_work_tree(event, directory.path, changes)
        else:
            self._take_from_git(event, directory, changes)

    def _take_from_work_tree(self, event: WatchEvent, directory: str, changes: LookChanges) -> None:
        if event.change is Change.directory_gone:
            if directory == ROOT_DIRECTORY:
                message = "the project's root went"
                raise FollowingLostError(message)
            # Deleted, moved or unmounted: whatever lies at its path now is listed again
            self._forget(directory, changes)
            changes.directories.add(directory)
            return
        # The directory's own attributes: those of each directory below the root come from
        # its parent, by its name
        if not event.name:
            return
        path = joined_path(directory, event.name)
        if self._repository is not None and (
            event.name in WORK_TREE_NAMES_STARTING_AGAIN
            or (directory == ROOT_DIRECTORY and event.name in ROOT_NAMES_STARTING_AGAIN)
        ):
            message = f"{path} changed"
            raise FollowingLostError(message)
        if not event.is_directory:
            changes.paths.add(path)
            return
        if event.change in (Change.entry_removed, Change.entry_changed):
            self._forget(path, changes)
        if event.change in (Change.entry_added, Change.entry_changed):
            changes.directories.add(path)

    def _take_from_git(
        self, event: WatchEvent, directory: WatchedDirectory, changes: LookChanges
    ) -> None:
        """Note what an event of git's own directories or a nested repository changed: git's
        directories going is found by their place at each look, a nested repository's by its
        parent's event."""
        if event.name in NAMES_STARTING_AGAIN.get(directory.role, NO_NAMES):
            message = f"{event.name} in {directory.path} changed"
            raise FollowingLostError(message)
        if event.name in NAMES_LISTING_AGAIN.get(directory.role, NO_NAMES):
            changes.list_again = True

    def _forget(self, directory: str, changes: LookChanges) -> None:
        """Stop watching a directory of the work tree and every one below it, its files read
        again, and no walk of them still to come."""
        for watch, watched in list(self._directories.items()):
            if watched.role in (
                DirectoryRole.work_tree,
                DirectoryRole.nested_repository,
            ) and lies_under(watched.path, directory):
                self._inotify.unwatch(watch)
                del self._directories[watch]
                self._watches.pop(watched.path, None)
        self._repositories_within = {
            path for path in self._repositories_within if not lies_under(path, directory)
        }
        self._looked_into = {path for path in self._looked_into if not lies_under(path, directory)}
        changes.paths |= {path for path in self._found if lies_under(path, directory)}
        changes.directories -= {path for path in changes.directories if lies_under(path, directory)}
        changes.directories_changed = True

    def _follow(self, directories: Iterable[str]) -> set[str]:
        """Watch each directory a scan reads, then list it, and so every one below it; the files
        met, by their paths from the root. Git is asked which of them it ignores whole before
        and again after, when every one is watched: a `.gitignore` changed between, in a
        directory not yet watched, sent no event, so a changed answer starts the following
        again. Raises the OSError of a limit reached."""
        starting = [directory for directory in directories if self._may_descend(directory)]
        if not starting:
            return set()
        ignored = self._ignored_within(starting)
        files: set[str] = set()
        pending = [directory for directory in starting if directory not in ignored]
        while pending:
            repositories = self._walk(pending, ignored, files)
            # Those git looks into are listed as the work tree they are
            pending = sorted(self._holding_tracked(repositories))
            self._looked_into |= set(pending)
        if self._ignored_within(starting) != ignored:
            message = "what git ignores changed while its directories were watched"
            raise FollowingLostError(message)
        return files

    def _walk(self, pending: list[str], ignored: frozenset[str], files: set[str]) -> list[str]:
        """Watch, then list, each directory pending and those below it, adding the files met;
        the directories met that hold a repository of their own, watched alone until git says
        whether it looks into them."""
        repositories: list[str] = []
        while pending:
            directory = pending.pop()
            try:
                watch = self._inotify.watch(self._absolute(directory))
            # Its parent's event tells of its going; what git lists in one the server may not
            # read is read at every look
            except FileNotFoundError, NotADirectoryError, PermissionError:
                continue
            entries = self._entries(directory)
            if self._holds_repository(directory, entries):
                self._repositories_within.add(directory)
                if directory not in self._looked_into:
                    self._directories[watch] = WatchedDirectory(
                        role=DirectoryRole.nested_repository, path=directory
                    )
                    repositories.append(directory)
                    continue
            self._directories[watch] = WatchedDirectory(
                role=DirectoryRole.work_tree, path=directory
            )
            self._watches[directory] = watch
            for entry in entries:
                path = joined_path(directory, entry.name)
                if not entry.is_dir(follow_symlinks=False):
                    files.add(path)
                elif self._may_descend(path) and path not in ignored:
                    pending.append(path)
        return repositories

    def _holding_tracked(self, directories: Collection[str]) -> frozenset[str]:
        """The directories holding a repository of their own that git looks into, as it tracks
        files below them."""
        if not directories:
            return frozenset()
        holding = self._discovery.directories_holding_tracked(self.root, directories)
        if holding is None:
            message = "git could not say which directories it tracks files in"
            raise FollowingLostError(message)
        return holding

    def _ignored_within(self, directories: list[str]) -> frozenset[str]:
        """The directories git ignores whole within those given, or the whole work tree where
        the root is among them; none where a walk lists the files. Of those git names, only
        those a walk would go into are asked of `git check-ignore`."""
        if self._repository is None:
            return frozenset()
        under = [] if ROOT_DIRECTORY in directories else directories
        named = self._discovery.directories_git_may_ignore(self.root, under)
        if named is None:
            message = "git could not name the directories it ignores"
            raise FollowingLostError(message)
        asked = [directory + PATH_SEPARATOR for directory in named if self._may_descend(directory)]
        ignored = self._discovery.ignored_by_git(self.root, asked)
        if ignored is None:
            message = "git could not say which directories it ignores"
            raise FollowingLostError(message)
        return frozenset(directory.removesuffix(PATH_SEPARATOR) for directory in ignored)

    def _entries(self, directory: str) -> list[os.DirEntry[str]]:
        """A directory's entries; none where it went since it was watched, which its parent's
        event says."""
        try:
            with os.scandir(self._absolute(directory)) as entries:
                return list(entries)
        except OSError:
            return []

    def _holds_repository(self, directory: str, entries: list[os.DirEntry[str]]) -> bool:
        """Whether a directory below the root holds a repository of its own, which git does not
        look into unless it tracks files below it."""
        return (
            self._repository is not None
            and directory != ROOT_DIRECTORY
            and any(entry.name == GIT_DIRECTORY_NAME for entry in entries)
        )

    def _may_descend(self, directory: str) -> bool:
        """Whether a scan could keep a file below a directory, git aside: it is not git's own,
        no pattern covers it and the index can hold its name."""
        name = directory.rpartition(PATH_SEPARATOR)[2]
        if self._repository is not None and name == GIT_DIRECTORY_NAME:
            return False
        return is_utf8(directory) and not self._rules.covers_directory(directory)

    def _list_again(self) -> set[str]:
        """Ask git for its listing again; the paths it lists now or listed before but not
        both. Which nested repositories git looks into may have changed with its index, and
        only a fresh start follows that."""
        if self._repositories_within and self._holding_tracked(self._repositories_within) != (
            self._looked_into
        ):
            message = "git looks into another set of nested repositories"
            raise FollowingLostError(message)
        listed = self._git_listing()
        changed = listed ^ self._listed
        self._listed = listed
        return changed

    def _git_listing(self) -> set[str]:
        listed = self._discovery.git_listing(self.root)
        if listed is None:
            message = "git could not list the files"
            raise FollowingLostError(message)
        return set(listed)

    def _read_again(self, paths: Collection[str], changes: LookChanges) -> bool:
        """Read the paths again by the scan's rules, an entry git has not listed yet asked of
        git first; whether a file's record changed."""
        check = CandidateCheck(self.root)
        unlisted: dict[str, FoundFile | None] = {}
        records_changed = False
        for path in paths:
            record = None
            if self._is_candidate(path):
                record = check.found(path)
                if not self._is_listed(path) and self._exists(path, record):
                    unlisted[path] = record
                    continue
            records_changed |= self._keep(path, record, is_link=path in check.links_met)
        for path, record in self._admitted(unlisted, changes).items():
            records_changed |= self._keep(path, record, is_link=path in check.links_met)
        return records_changed

    def _exists(self, path: str, record: FoundFile | None) -> bool:
        """Whether an entry lies at a path: a file read, or any other, as git lists a symlink
        whatever it leads to."""
        return record is not None or os.path.lexists(self._absolute(path))

    def _admitted(
        self, unlisted: dict[str, FoundFile | None], changes: LookChanges
    ) -> dict[str, FoundFile | None]:
        """The entries git lists among those it had not, added to its listing, each with its
        record; those it ignores with none. Many at once are told by git's listing again, what
        else it changed read at the look its index's event comes."""
        if len(unlisted) > MOST_ENTRIES_ASKED_OF_GIT:
            self._list_again()
            changes.list_again = True
        else:
            ignored = self._discovery.ignored_by_git(self.root, unlisted.keys())
            if ignored is None:
                message = "git could not say which new files it ignores"
                raise FollowingLostError(message)
            self._listed |= unlisted.keys() - ignored
        return {path: record if path in self._listed else None for path, record in unlisted.items()}

    def _is_candidate(self, path: str) -> bool:
        """Whether a scan could keep the path: one a scan wants, which git lists or may list as
        it lies in a directory followed."""
        return self._discovery.is_wanted(path, self._rules) and (
            path in self._listed or parent_of(path) in self._watches
        )

    def _is_listed(self, path: str) -> bool:
        """Whether a scan's listing holds a candidate: git's, or a walk's of every file of a
        directory followed."""
        return path in self._listed if self._repository is not None else True

    def _keep(self, path: str, record: FoundFile | None, *, is_link: bool) -> bool:
        """Keep a path's record as read, and whether it is a symlink of the listing; whether
        its record changed."""
        if is_link and self._is_listed(path):
            self._links.add(path)
        else:
            self._links.discard(path)
        if record is None:
            return self._found.pop(path, None) is not None
        if self._found.get(path) == record:
            return False
        self._found[path] = record
        return True

    def _polled_paths(self) -> frozenset[str]:
        """What no watch covers, read at every look: the symlinks listed, whose targets may lie
        anywhere, and what git lists in a directory nothing follows."""
        unfollowed = {
            path
            for path in self._listed
            if parent_of(path) not in self._watches and self._discovery.is_wanted(path, self._rules)
        }
        return frozenset(self._links | unfollowed)

    def _absolute(self, path: str) -> str:
        """A path from the root as the file system takes it."""
        return self._root_text if path == ROOT_DIRECTORY else f"{self._root_text}{os.sep}{path}"
