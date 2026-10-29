"""A project's files followed by the system's watcher of directories since a full scan (ADR 010,
O1): what each look reads of what changed, so it finds what a scan would.

A followed project keeps git's last listing and each listed file's record as last read, and at
each look reads the watcher's events, which hold every change its watches saw before the
look. A path that changed is read again by the scan's rules. Git is asked for its listing again,
never patched, where it may have changed: its index rewritten, or a name it has not listed made
in a directory followed; the paths whose listing changed are read again with the rest. A new
directory is watched, then listed, on the live instance, so nothing made meanwhile goes unseen;
the directories gone in one look are forgotten in one pass, their watches dropped and their
files read again; a nested repository is looked into only where git tracks files below it as it
is walked, as git does, and whatever git's listing says of it since holds. What changes the
rules themselves (a `.gitignore`, an excludes file, git's configuration, a repository made or
gone, the root or git's own directories no longer where they were) and a queue that overflowed
end the following: only a fresh start and a full scan can tell the files then. What no watch's
events tell is read at every look: a symlink, whose target may lie anywhere, a file of more than
one name, edited through another, and a file git lists in a directory nothing watches (one git
ignores though it tracks the file, one a nested repository holds, one the server may not read).
What no event and no reading tells (the user's global git configuration and excludes file, a
second name a file got since it was read, a file system mounted or unmounted below the root)
`files.py`'s check against a scan finds now and then.
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
from semantic_code_mcp.domain.interfaces import DirectoryWatcher
from semantic_code_mcp.domain.models import FileStamps, Place, WatchChange, WatchEvent

# Where git keeps its work, in a work tree or as a nested repository's mark; the files whose
# change can change which files git lists; and those of git's own directories that matter here
GIT_DIRECTORY_NAME = ".git"
GITIGNORE_NAME = ".gitignore"
INDEX_NAME = "index"
INFO_DIRECTORY_NAME = "info"
EXCLUDE_NAME = "exclude"
CONFIGURATION_NAME = "config"
WORKTREE_CONFIGURATION_NAME = "config.worktree"


class WorkTreeRole(StrEnum):
    """Why a directory of the work tree is watched, which says what its events mean."""

    # A directory a scan reads: a change of any entry is read again
    followed = auto()
    # A repository nested in the project, which git does not look into: only its `.git` counts
    nested_repository = auto()


class GitRole(StrEnum):
    """Which of git's own directories is watched."""

    # Its own directory, with the index and a linked worktree's configuration
    git_directory = auto()
    # The directory a linked worktree's repository shares, with `info` and the configuration
    common_directory = auto()
    # `info`, with the excludes file
    info_directory = auto()


# The entries of git's directories whose change lists the files again
NAMES_LISTING_AGAIN = {GitRole.git_directory: frozenset({INDEX_NAME})}
# Those whose change starts the following again; a main work tree's git directory is its common
# directory too
NAMES_STARTING_AGAIN = {
    GitRole.git_directory: frozenset(
        {INFO_DIRECTORY_NAME, CONFIGURATION_NAME, WORKTREE_CONFIGURATION_NAME}
    ),
    GitRole.common_directory: frozenset({INFO_DIRECTORY_NAME, CONFIGURATION_NAME}),
    GitRole.info_directory: frozenset({EXCLUDE_NAME}),
}
# The names in a work tree whose change starts the following again in a repository: a nested
# repository made or gone, or the project's own; the rules of what git ignores. Submodules made
# or made active write git's configuration, which does so too
WORK_TREE_NAMES_STARTING_AGAIN = frozenset({GIT_DIRECTORY_NAME, GITIGNORE_NAME})
NO_NAMES = frozenset[str]()


class FollowingLostError(RuntimeError):
    """The files kept can no longer be trusted, as the queue overflowed, the root went, git
    failed or a rule changed: only a full scan can tell them."""


class WorkTreeDirectory(BaseModel):
    """A directory of the work tree watched: why, and its path from the root."""

    model_config = ConfigDict(frozen=True)

    role: WorkTreeRole
    path: str


class GitDirectory(BaseModel):
    """One of git's own directories watched: which, and where."""

    model_config = ConfigDict(frozen=True)

    role: GitRole
    path: Path


class LookChanges(BaseModel):
    """What the events of one look changed: the paths to read again, the directories to forget
    and those to watch and list, and whether git is asked for its listing again."""

    paths: set[str]
    gone: set[str]
    added: set[str]
    list_again: bool


def place_of(directory: Path) -> Place | None:
    """Where the directory at a path lies now; None where none does."""
    try:
        status = directory.stat()
    except OSError:
        return None
    return Place(device=status.st_dev, inode=status.st_ino)


def parent_of(path: str) -> str:
    """The directory a path lies in, from the root."""
    return path.rpartition(PATH_SEPARATOR)[0]


def lies_under_any(path: str, directories: Collection[str]) -> bool:
    """Whether a path from the root is one of the directories or lies below one: its own text
    and each directory above it looked up, so a look's gone directories cost one pass."""
    while path != ROOT_DIRECTORY:
        if path in directories:
            return True
        path = parent_of(path)
    return False


class FollowedFiles:
    """A project's files kept from a watcher's events since a full scan: the work tree's
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
        watcher: DirectoryWatcher,
    ) -> None:
        self.root = root
        # The root as text, which a path from it joins without a `Path` built per path
        self._root_text = str(root)
        # None where a walk lists the files, as the settings ask
        self._repository = repository
        self._discovery = discovery
        self._watcher = watcher
        self._rules = discovery.ignore_rules()
        self._work_tree: dict[int, WorkTreeDirectory] = {}
        self._git: dict[int, GitDirectory] = {}
        # The work tree's directories followed, by their paths from the root
        self._watches: dict[str, int] = {}
        # Git's last listing, by the paths from the root; empty where a walk lists every file of
        # a directory followed
        self._listed: set[str] = set()
        self._found: FoundFiles = {}
        # The paths listed whose changes no event of their directory tells: symlinks, and files
        # of more than one name
        self._changed_elsewhere: set[str] = set()
        self._polled: frozenset[str] = frozenset()
        self._files: FileStamps = {}
        # Where the root and git's directories lay as the following started, by their paths: a
        # directory above one moved sends its watches nothing
        self._places: dict[Path, Place] = {}
        # The directories below the root holding a repository of their own that git looks into,
        # as it tracks files below them
        self._looked_into: set[str] = set()

    @classmethod
    def started(
        cls,
        *,
        root: Path,
        repository: Repository | None,
        discovery: SourceDiscovery,
        watcher: DirectoryWatcher,
    ) -> FollowedFiles:
        """A project followed from now: every directory watched before it is listed, then the
        files listed and read, so what changes meanwhile is the first look's. Raises
        `FollowingLostError` where it cannot be followed, and the OSError the system gives,
        as at its limits."""
        followed = cls(root=root, repository=repository, discovery=discovery, watcher=watcher)
        followed._begin()
        return followed

    @property
    def files(self) -> FileStamps:
        """The files as the last look read them."""
        return self._files

    def look(self) -> FileStamps:
        """The files now, with what changed since the last look read again. Raises
        `FollowingLostError` where only a full scan can tell, and the OSError the system
        gives."""
        for directory, place in self._places.items():
            if place_of(directory) != place:
                message = f"{directory} is no longer where the following found it"
                raise FollowingLostError(message)
        changes = LookChanges(paths=set(), gone=set(), added=set(), list_again=False)
        for event in self._watcher.events():
            self._take(event, changes)
        if changes.gone:
            self._forget(changes.gone, changes)
        if changes.added:
            met = self._follow(changes.added)
            changes.paths |= met
            changes.list_again |= self._lists_anew(met)
        if changes.list_again:
            changes.paths |= self._list_again()
        changed_elsewhere = set(self._changed_elsewhere)
        paths = changes.paths | self._polled
        if paths and self._read_again(paths):
            self._files = kept_once(self._found.values())
        if (
            changes.gone
            or changes.added
            or changes.list_again
            or changed_elsewhere != self._changed_elsewhere
        ):
            self._polled = self._polled_paths()
        return self._files

    def close(self) -> None:
        """Stop watching."""
        self._watcher.close()

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
        self._changed_elsewhere = set(check.changed_elsewhere)
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
        """Watch where git keeps its index, excludes and configuration; its `info` only where
        it is, as its making is seen in the directory that holds it."""
        self._watch_git(repository.git_directory, GitRole.git_directory)
        if repository.common_directory != repository.git_directory:
            self._watch_git(repository.common_directory, GitRole.common_directory)
        info = repository.common_directory / INFO_DIRECTORY_NAME
        if info.is_dir():
            self._watch_git(info, GitRole.info_directory)

    def _watch_git(self, directory: Path, role: GitRole) -> None:
        try:
            watch = self._watcher.watch(str(directory))
        except (FileNotFoundError, NotADirectoryError, PermissionError) as error:
            message = f"git's directory {directory} cannot be watched: {error}"
            raise FollowingLostError(message) from error
        self._git[watch] = GitDirectory(role=role, path=directory)

    def _take(self, event: WatchEvent, changes: LookChanges) -> None:
        """Note what one event changed. An event of a directory forgotten in this look keeps
        its path as it was: what it names is read where that path leads now, which a walk of
        the directory's new place reads too."""
        if event.change is WatchChange.queue_overflowed:
            message = "the watcher's queue overflowed"
            raise FollowingLostError(message)
        git = self._git.get(event.watch)
        if git is not None:
            self._take_from_git(event, git, changes)
            return
        # A watch dropped since, whose directory a walk lists again where it still lies:
        # a watcher gives no watch's number twice until it wraps round
        directory = self._work_tree.get(event.watch)
        if directory is None:
            return
        if directory.role is WorkTreeRole.nested_repository:
            if event.name == GIT_DIRECTORY_NAME:
                message = f"the repository in {directory.path} changed"
                raise FollowingLostError(message)
            return
        self._take_from_work_tree(event, directory.path, changes)

    def _take_from_work_tree(self, event: WatchEvent, directory: str, changes: LookChanges) -> None:
        # A directory below the root deleted or moved is told by its parent, by its name; one
        # unmounted, by its own watch alone, is the check against a scan's to find
        if event.change is WatchChange.directory_gone:
            if directory == ROOT_DIRECTORY:
                message = "the project's root went"
                raise FollowingLostError(message)
            return
        # The directory's own attributes: those of each directory below the root come from
        # its parent, by its name
        if not event.name:
            return
        path = joined_path(directory, event.name)
        if self._repository is not None and event.name in WORK_TREE_NAMES_STARTING_AGAIN:
            message = f"{path} changed"
            raise FollowingLostError(message)
        if event.is_directory:
            # A directory's attributes changed, as its permissions: what may be read of it now
            # is listed again
            if event.change in (WatchChange.entry_removed, WatchChange.entry_changed):
                changes.gone.add(path)
            if event.change in (WatchChange.entry_added, WatchChange.entry_changed):
                changes.added.add(path)
            return
        changes.paths.add(path)
        if event.change is WatchChange.entry_added:
            changes.list_again |= self._lists_anew([path])

    def _take_from_git(
        self, event: WatchEvent, directory: GitDirectory, changes: LookChanges
    ) -> None:
        """Note what an event of git's own directories changed: their going is found by their
        place at each look."""
        if event.name in NAMES_STARTING_AGAIN.get(directory.role, NO_NAMES):
            message = f"{event.name} in {directory.path} changed"
            raise FollowingLostError(message)
        if event.name in NAMES_LISTING_AGAIN.get(directory.role, NO_NAMES):
            changes.list_again = True

    def _lists_anew(self, paths: Iterable[str]) -> bool:
        """Whether git may list a file it has not among the entries made: one a scan wants,
        where git lists the files."""
        return self._repository is not None and any(
            path not in self._listed and self._discovery.is_wanted(path, self._rules)
            for path in paths
        )

    def _forget(self, gone: Collection[str], changes: LookChanges) -> None:
        """Stop watching the directories gone and every one below them, their files read
        again: in one pass over what is watched and kept, however many went."""
        for watch, watched in list(self._work_tree.items()):
            if lies_under_any(watched.path, gone):
                self._watcher.unwatch(watch)
                del self._work_tree[watch]
                self._watches.pop(watched.path, None)
        self._looked_into = {path for path in self._looked_into if not lies_under_any(path, gone)}
        changes.paths |= {path for path in self._found if lies_under_any(parent_of(path), gone)}

    def _follow(self, directories: Iterable[str]) -> set[str]:
        """Watch each directory a scan reads, then list it, and so every one below it; the files
        met, by their paths from the root. Git is asked which of them it ignores whole before
        and again after, when every one is watched: a `.gitignore` changed between, in a
        directory not yet watched, sent no event, so a changed answer starts the following
        again. A directory gone meanwhile is passed over. Raises the OSError the system gives,
        as at its limit of watches."""
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
                watch = self._watcher.watch(self._absolute(directory))
            # Its parent's event tells of its going; what git lists in one the server may not
            # read is read at every look
            except FileNotFoundError, NotADirectoryError, PermissionError:
                continue
            entries = self._entries(directory)
            if self._holds_repository(directory, entries) and directory not in self._looked_into:
                self._work_tree[watch] = WorkTreeDirectory(
                    role=WorkTreeRole.nested_repository, path=directory
                )
                repositories.append(directory)
                continue
            self._work_tree[watch] = WorkTreeDirectory(role=WorkTreeRole.followed, path=directory)
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
        """A directory's entries; none where it went or may no longer be read since it was
        watched, which its parent's event says."""
        try:
            with os.scandir(self._absolute(directory)) as entries:
                return list(entries)
        except FileNotFoundError, NotADirectoryError, PermissionError:
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
        both."""
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

    def _read_again(self, paths: Collection[str]) -> bool:
        """Read the paths again by the scan's rules; whether a file's record changed."""
        check = CandidateCheck(self.root)
        records_changed = False
        for path in paths:
            record = check.found(path) if self._is_candidate(path) else None
            records_changed |= self._keep(path, record)
            if path in check.changed_elsewhere:
                self._changed_elsewhere.add(path)
            else:
                self._changed_elsewhere.discard(path)
        return records_changed

    def _is_candidate(self, path: str) -> bool:
        """Whether a scan reads the path: one it wants, in git's listing or, where a walk lists
        the files, in a directory followed."""
        listed = (
            path in self._listed
            if self._repository is not None
            else parent_of(path) in self._watches
        )
        return listed and self._discovery.is_wanted(path, self._rules)

    def _keep(self, path: str, record: FoundFile | None) -> bool:
        """Keep a path's record as read; whether it changed."""
        if record is None:
            return self._found.pop(path, None) is not None
        if self._found.get(path) == record:
            return False
        self._found[path] = record
        return True

    def _polled_paths(self) -> frozenset[str]:
        """What no watch's events tell, read at every look: the paths listed whose changes come
        from elsewhere, and what git lists in a directory nothing follows."""
        unfollowed = {
            path
            for path in self._listed
            if parent_of(path) not in self._watches and self._discovery.is_wanted(path, self._rules)
        }
        return frozenset(self._changed_elsewhere | unfollowed)

    def _absolute(self, path: str) -> str:
        """A path from the root as the file system takes it."""
        return self._root_text if path == ROOT_DIRECTORY else f"{self._root_text}{os.sep}{path}"
