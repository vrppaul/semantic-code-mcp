"""Tests of a project's files followed by the system's watcher between full scans: what they
hold is what a scan finds, a look after an edit reads what changed without asking git, and a
project no watch can tell is scanned. Each runs on every system with a watcher: inotify on Linux,
FSEvents on macOS, ReadDirectoryChangesW on Windows."""

import contextlib
import ctypes
import os
import shutil
import stat
import sys
import tempfile
from collections.abc import Callable, Collection
from functools import partial
from pathlib import Path
from typing import override

import pytest
from hypothesis import given, settings, strategies
from pydantic import BaseModel

from semantic_code_mcp.app import (
    build_chunker,
    create_app,
    project_registry_lifespan,
    store_opener,
)
from semantic_code_mcp.config import Settings
from semantic_code_mcp.discovery import SourceDiscovery
from semantic_code_mcp.domain.interfaces import DirectoryWatcher
from semantic_code_mcp.domain.models import FileStamps, WatchEvent
from semantic_code_mcp.services.files import CHECK_INTERVAL_SECONDS, WatchedFiles
from semantic_code_mcp.watching import directory_changes
from semantic_code_mcp.watching.directory_changes import BUFFER_SIZE, NAME_ENCODING, RECORD_HEADER
from semantic_code_mcp.watching.fsevents import MARKS_PREFIX
from semantic_code_mcp.watching.systems import (
    LINUX_PLATFORM,
    MACOS_PLATFORM,
    WINDOWS_PLATFORM,
    system_watcher,
)
from tests.fakes import FakeEmbedder
from tests.helpers import commit_all, discovery_for, indexing_service, make_repository, run_git
from tests.indexing import NO_PROGRESS
from tests.stores import IDENTITY, StoreClass, open_store

pytestmark = pytest.mark.skipif(system_watcher() is None, reason="this system has no watcher")
ONLY_ON_LINUX = pytest.mark.skipif(sys.platform != LINUX_PLATFORM, reason="Linux's alone")
ONLY_ON_MACOS = pytest.mark.skipif(sys.platform != MACOS_PLATFORM, reason="macOS's alone")
ONLY_ON_WINDOWS = pytest.mark.skipif(sys.platform != WINDOWS_PLATFORM, reason="Windows' alone")
# APFS and NTFS hold no name that is not UTF-8
NAMES_NOT_UTF8 = ONLY_ON_LINUX
NAMES_UTF8_ALONE = pytest.mark.skipif(
    sys.platform == LINUX_PLATFORM, reason="run with names that are not UTF-8"
)

# A name a module and a directory take in turn
SHIFTING_NAME = "shifting.py"
# An mtime well after any file the test wrote
LATER_MTIME_SECONDS = 4_000_000_000
SOURCE = "x = 1\n"
MORE_SOURCE = "y = 2\n"
TEXT = "notes\n"
OPERATION_EXAMPLES = 30
MOST_OPERATIONS = 8
EDITS_LOOKED_AT = 100
READS_OF_EVERY_FILE = 20_000
# How many events inotify's queue holds before it overflows, as Linux says it
QUEUE_LIMIT_FILE = Path("/proc/sys/fs/inotify/max_queued_events")
# Where the clock of these tests starts
START_OF_TIME = 0.0
# More new modules than a look usually meets, made at once
MANY_MODULES = 150
# renameat2's flag that swaps two paths, and the directory a relative path is read from; and
# the flag of macOS's renamex_np that does the same
RENAME_EXCHANGE = 2
CURRENT_DIRECTORY = -100
RENAME_SWAP = 2
# The error inotify gives at the user's limit of watches
LIMIT_OF_WATCHES_ERROR = OSError(28, "No space left on device")
# A name of 100 characters, whose records fill ReadDirectoryChangesW's buffer soon: each a header
# of 12 bytes and the name in UTF-16, one as it is made and one as it is deleted
LONG_NAME = "n" * 100 + ".py"
RECORD_BYTES = RECORD_HEADER.size + len(LONG_NAME.encode(NAME_ENCODING))
RECORDS_A_FILE = 2
# The files made and deleted: enough to fill the buffer twice over
BUFFERS_FILLED = 2
FILES_FILLING_THE_BUFFER = BUFFERS_FILLED * BUFFER_SIZE // (RECORDS_A_FILE * RECORD_BYTES)
# What a tree keeps between looks in a test of what it keeps no more: as many modules made each
# tell more changes than that
FEW_CHANGES_KEPT = 50
EXECUTE_ONLY = 0o311
READABLE = 0o755
# This process's open files
OPEN_DESCRIPTORS = Path("/proc/self/fd")
# A directory name that is not UTF-8, as bytes: decoded where a test runs, as Windows cannot
# decode it and this module is imported there too
NOT_UTF8_NAME = b"caf\xe9"


class RecordingDiscovery(SourceDiscovery):
    """The real discovery, keeping how often it asked git for its listing and what it asked
    git to check."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(
            extensions=build_chunker().supported_extensions,
            ignore_patterns=settings.ignore_patterns,
            use_gitignore=settings.use_gitignore,
            git_timeout_seconds=settings.git_timeout_seconds,
            home_directory=Path.home(),
        )
        self.listings = 0
        self.checked: list[str] = []

    @override
    def git_listing(self, root: Path) -> list[str] | None:
        self.listings += 1
        return super().git_listing(root)

    @override
    def ignored_by_git(self, root: Path, paths: Collection[str]) -> frozenset[str] | None:
        self.checked += sorted(paths)
        return super().ignored_by_git(root, paths)


class DirectoryMadeAsListed(RecordingDiscovery):
    """The real discovery, which makes a directory as git lists the files, after the walk."""

    def __init__(self, settings: Settings, late_directory: Path) -> None:
        super().__init__(settings)
        self._late_directory = late_directory

    @override
    def git_listing(self, root: Path) -> list[str] | None:
        self._late_directory.mkdir(exist_ok=True)
        return super().git_listing(root)


class IgnoreRulesChangedAsAsked(RecordingDiscovery):
    """The real discovery, whose project stops ignoring what it ignored right after git first
    says which directories it ignores, before any is watched."""

    def __init__(self, settings: Settings, gitignore: Path) -> None:
        super().__init__(settings)
        # The files to empty, once
        self._pending = [gitignore]

    @override
    def ignored_by_git(self, root: Path, paths: Collection[str]) -> frozenset[str] | None:
        ignored = super().ignored_by_git(root, paths)
        for gitignore in self._pending:
            gitignore.write_text("")
        self._pending.clear()
        return ignored


def open_watcher() -> DirectoryWatcher:
    """A watcher of this system's own."""
    watcher = system_watcher()
    assert watcher is not None
    return watcher()


class WatcherAround(DirectoryWatcher):
    """A watcher of this system's own, each call handed on to it."""

    def __init__(self) -> None:
        self._watcher = open_watcher()

    @override
    def watch(self, directory: str) -> int:
        return self._watcher.watch(directory)

    @override
    def unwatch(self, watch: int) -> None:
        self._watcher.unwatch(watch)

    @override
    def events(self) -> list[WatchEvent]:
        return self._watcher.events()

    @override
    def close(self) -> None:
        self._watcher.close()


class DirectoryMadeAsWatched(WatcherAround):
    """A watcher in whose directories a directory is made just before each is watched, after
    nothing listed it yet."""

    def __init__(self, made_at: dict[str, list[Path]]) -> None:
        super().__init__()
        # The directories made just before the first watch of a directory, by that directory
        self._made_at = made_at

    @override
    def watch(self, directory: str) -> int:
        for made in self._made_at.pop(directory, []):
            made.mkdir()
        return super().watch(directory)


class RecordingWatcher(WatcherAround):
    """A watcher keeping every directory it was asked to watch, as the file system spells it."""

    def __init__(self) -> None:
        super().__init__()
        self.watched: set[str] = set()

    @override
    def watch(self, directory: str) -> int:
        self.watched.add(os.path.normpath(directory))
        return super().watch(directory)


class SetClock:
    """A clock that reads what the test set it to."""

    def __init__(self) -> None:
        self.now = START_OF_TIME

    def __call__(self) -> float:
        return self.now


def exchanged_by_renameat2(first: Path, second: Path) -> int:
    """Swap two paths in one call, as Linux's `mv --exchange` does: each directory's parent sees
    one leave and the other come under one name. What the call returned, 0 where it swapped."""
    library = ctypes.CDLL(None, use_errno=True)
    return library.renameat2(
        CURRENT_DIRECTORY,
        os.fsencode(first),
        CURRENT_DIRECTORY,
        os.fsencode(second),
        RENAME_EXCHANGE,
    )


def exchanged_by_renamex_np(first: Path, second: Path) -> int:
    """Swap two paths in one call, as macOS's `renamex_np` with RENAME_SWAP does."""
    library = ctypes.CDLL(None, use_errno=True)
    return library.renamex_np(os.fsencode(first), os.fsencode(second), RENAME_SWAP)


class CountedWatchers:
    """Makes watchers of this system's own, counting them."""

    def __init__(self) -> None:
        self.opened = 0

    def __call__(self) -> DirectoryWatcher:
        self.opened += 1
        return open_watcher()


class WatcherAtItsLimit(WatcherAround):
    """A watcher at the system's limit of watches."""

    @override
    def watch(self, directory: str) -> int:
        raise LIMIT_OF_WATCHES_ERROR


class CountedWatchersAtTheirLimit(CountedWatchers):
    @override
    def __call__(self) -> DirectoryWatcher:
        self.opened += 1
        return WatcherAtItsLimit()


def writable_again(remove: Callable[[str], object], path: str, _error: BaseException) -> None:
    """Remove what Windows refused to, once it may be written: git keeps its objects read-only."""
    Path(path).chmod(stat.S_IWRITE)
    remove(path)


def remove_tree(directory: Path) -> None:
    """Remove a directory and all in it, as gone already where it is."""
    with contextlib.suppress(FileNotFoundError):
        shutil.rmtree(directory, onexc=writable_again)


def scan_of(project: Path, settings: Settings) -> FileStamps:
    """What a scan of the project finds now."""
    return discovery_for(settings, build_chunker()).discover(project)


def write_source(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SOURCE)


def write_top_module(project: Path) -> None:
    write_source(project / "a.py")


def write_deep_module(project: Path) -> None:
    write_source(project / "src" / "deep" / "c.py")


def write_document(project: Path) -> None:
    write_source(project / "docs" / "guide.md")


def write_in_a_new_directory(project: Path) -> None:
    write_source(project / "fresh" / "new.py")


def write_into_build(project: Path) -> None:
    write_source(project / "build" / "generated.py")


def edit_top_module(project: Path) -> None:
    write_top_module(project)
    with (project / "a.py").open("a") as module:
        module.write(MORE_SOURCE)


def touch_top_module(project: Path) -> None:
    write_top_module(project)
    os.utime(project / "a.py", (LATER_MTIME_SECONDS, LATER_MTIME_SECONDS))


def delete_top_module(project: Path) -> None:
    (project / "a.py").unlink(missing_ok=True)


def rename_a_module(project: Path) -> None:
    write_source(project / "src" / "b.py")
    (project / "src" / "b.py").replace(project / "src" / "renamed.py")


def rename_a_directory(project: Path) -> None:
    write_deep_module(project)
    shutil.rmtree(project / "moved", ignore_errors=True)
    (project / "src").replace(project / "moved")


def delete_a_tree(project: Path) -> None:
    shutil.rmtree(project / "src", ignore_errors=True)


def move_a_directory_out(project: Path) -> None:
    """Out of the project: its files send nothing, the directory's going is all there is."""
    write_deep_module(project)
    shutil.rmtree(project.parent / "taken", ignore_errors=True)
    (project / "src").replace(project.parent / "taken")


def move_a_directory_in(project: Path) -> None:
    """Into the project, with its files: none was written where a watch saw it."""
    write_source(project.parent / "arriving" / "deeper" / "d.py")
    shutil.rmtree(project / "arrived", ignore_errors=True)
    (project.parent / "arriving").replace(project / "arrived")


def ignore_directories_and_names(project: Path) -> None:
    (project / ".gitignore").write_text("deep/\nskipped.py\n*.log\n")


def write_skipped_module(project: Path) -> None:
    write_source(project / "skipped.py")


def write_a_log(project: Path) -> None:
    """A directory whose files git ignores, though it does not ignore the directory."""
    write_source(project / "logs" / "run.log")


def write_a_module_beside_the_logs(project: Path) -> None:
    write_source(project / "logs" / "parser.py")


def clear(path: Path) -> None:
    """Whatever lies at a path removed, a file or a directory, as nothing lies there already."""
    with contextlib.suppress(FileNotFoundError, IsADirectoryError, PermissionError):
        path.unlink()
    remove_tree(path)


def make_a_name_a_package(project: Path) -> None:
    """The name of a file, if one lay there, taken by a directory holding a module."""
    clear(project / SHIFTING_NAME)
    write_source(project / SHIFTING_NAME / "inner.py")


def make_a_name_a_module(project: Path) -> None:
    """The name of a directory, if one lay there, taken by a module."""
    clear(project / SHIFTING_NAME)
    write_source(project / SHIFTING_NAME)


def link_to_a_module(project: Path) -> None:
    write_top_module(project)
    (project / "alias.py").unlink(missing_ok=True)
    (project / "alias.py").symlink_to("a.py")


def link_outside(project: Path) -> None:
    write_source(project.parent / "outside.py")
    (project / "away.py").unlink(missing_ok=True)
    (project / "away.py").symlink_to(project.parent / "outside.py")


def link_to_a_text_file(project: Path) -> None:
    """A link whose target no scan keeps of its own, nor is edited where its link lies."""
    (project / "notes.txt").write_text(TEXT)
    (project / "notes.py").unlink(missing_ok=True)
    (project / "notes.py").symlink_to("notes.txt")


def edit_the_text_file(project: Path) -> None:
    with (project / "notes.txt").open("a") as notes:
        notes.write(TEXT)


def link_to_a_text_file_not_yet_written(project: Path) -> None:
    """A link git lists though it leads nowhere yet: its target, made later, is no change
    where the link lies, and no file a scan keeps of its own."""
    (project / "early.py").unlink(missing_ok=True)
    (project / "early.py").symlink_to("later.txt")


def write_the_later_text_file(project: Path) -> None:
    (project / "later.txt").write_text(TEXT)


def name_a_directory_not_utf8(project: Path) -> None:
    write_source(project / os.fsdecode(NOT_UTF8_NAME) / "inside.py")


def commit_anything(repository: Path) -> None:
    """A commit of all there is, even nothing new, which rewrites git's index."""
    run_git(repository, "add", "--all")
    run_git(repository, "commit", "--quiet", "--allow-empty", "--message", "snapshot")


def nest_a_repository(project: Path) -> None:
    write_source(project / "nested" / "inner.py")
    commit_anything(make_repository(project / "nested"))


def remove_the_nested_repository_s_git(project: Path) -> None:
    remove_tree(project / "nested" / ".git")


def commit_everything(project: Path) -> None:
    commit_anything(project)


def force_add_everything(project: Path) -> None:
    """What git ignores, tracked now: its index alone changes, and git lists those files."""
    run_git(project, "add", "--all", "--force")


def refresh_git_s_index(project: Path) -> None:
    """A status after a file was touched, which rewrites the index's record of it."""
    run_git(project, "status", "--porcelain")


# A change of a project, made in place
type Operation = Callable[[Path], None]

WALKED_OPERATIONS: list[Operation] = [
    write_top_module,
    write_deep_module,
    write_document,
    write_in_a_new_directory,
    write_into_build,
    edit_top_module,
    touch_top_module,
    delete_top_module,
    rename_a_module,
    rename_a_directory,
    delete_a_tree,
    move_a_directory_out,
    move_a_directory_in,
    write_skipped_module,
    write_a_log,
    write_a_module_beside_the_logs,
    link_to_a_module,
    link_outside,
    link_to_a_text_file,
    edit_the_text_file,
    link_to_a_text_file_not_yet_written,
    write_the_later_text_file,
    make_a_name_a_package,
    make_a_name_a_module,
]
REPOSITORY_OPERATIONS = [
    *WALKED_OPERATIONS,
    ignore_directories_and_names,
    nest_a_repository,
    remove_the_nested_repository_s_git,
    commit_everything,
    force_add_everything,
    refresh_git_s_index,
]
WALKED = Settings(use_gitignore=False)


def plain_directory(directory: Path) -> Path:
    project = directory / "project"
    project.mkdir()
    return project


def repository(directory: Path) -> Path:
    return make_repository(directory / "project")


class LookedAt(BaseModel):
    """What the files followed held and what a scan found, after each change of a project."""

    kept: list[FileStamps]
    scanned: list[FileStamps]


def looked_at(
    make_project: Callable[[Path], Path], operations: list[Operation], project_settings: Settings
) -> LookedAt:
    """A project made in a directory of its own and changed by each operation in turn."""
    with tempfile.TemporaryDirectory() as directory:
        project = make_project(Path(directory))
        files = WatchedFiles(
            discovery_for(project_settings, build_chunker()), open_watcher, SetClock()
        )
        files.current(project)
        looked = LookedAt(kept=[], scanned=[])
        for operation in operations:
            operation(project)
            looked.kept.append(files.current(project))
            looked.scanned.append(scan_of(project, project_settings))
        files.close()
    return looked


@pytest.mark.parametrize(
    ("make_project", "operations", "project_settings"),
    [
        pytest.param(
            repository,
            [*REPOSITORY_OPERATIONS, name_a_directory_not_utf8],
            Settings(),
            id="listed by git",
            marks=NAMES_NOT_UTF8,
        ),
        pytest.param(
            plain_directory,
            [*WALKED_OPERATIONS, name_a_directory_not_utf8],
            WALKED,
            id="walked, as set",
            marks=NAMES_NOT_UTF8,
        ),
        pytest.param(
            repository,
            REPOSITORY_OPERATIONS,
            Settings(),
            id="listed by git, names in UTF-8",
            marks=NAMES_UTF8_ALONE,
        ),
        pytest.param(
            plain_directory,
            WALKED_OPERATIONS,
            WALKED,
            id="walked, as set, names in UTF-8",
            marks=NAMES_UTF8_ALONE,
        ),
    ],
)
@settings(max_examples=OPERATION_EXAMPLES, deadline=None)
@given(data=strategies.data())
def test_the_files_followed_are_those_a_scan_finds(
    make_project: Callable[[Path], Path],
    operations: list[Operation],
    project_settings: Settings,
    data: strategies.DataObject,
):
    """A change missed would leave the index stale for good; one invented, a file indexed that
    no scan lists."""
    done = data.draw(
        strategies.lists(strategies.sampled_from(operations), max_size=MOST_OPERATIONS)
    )

    looked = looked_at(make_project, done, project_settings)

    assert looked.kept == looked.scanned


def worktree_of_a_repository_ignoring_a_module(directory: Path) -> Path:
    """A linked worktree, whose `.git` is a file: its index and excludes lie outside it."""
    main = repository(directory)
    write_top_module(main)
    (main / ".gitignore").write_text("skipped.py\n")
    commit_all(main)
    run_git(main, "worktree", "add", "--quiet", str(directory / "worktree"))
    write_skipped_module(directory / "worktree")
    return directory / "worktree"


def add_the_skipped_module_by_force(project: Path) -> None:
    run_git(project, "add", "--force", "skipped.py")


def ignore_the_top_module_too(project: Path) -> None:
    (project / ".gitignore").write_text("skipped.py\na.py\n")


def untrack_the_top_module(project: Path) -> None:
    run_git(project, "rm", "--cached", "--quiet", "a.py")


def repository_with_a_branch_of_other_files(directory: Path) -> Path:
    project = repository(directory)
    write_top_module(project)
    write_source(project / "lib" / "c.py")
    commit_all(project)
    run_git(project, "checkout", "--quiet", "-b", "other")
    write_source(project / "src" / "only_on_other.py")
    (project / "lib" / "c.py").unlink()
    edit_top_module(project)
    commit_all(project)
    run_git(project, "checkout", "--quiet", "-")
    return project


def check_out_the_other_branch(project: Path) -> None:
    run_git(project, "checkout", "--quiet", "other")


def check_out_the_branch_before(project: Path) -> None:
    run_git(project, "checkout", "--quiet", "-")


def committed_repository(directory: Path) -> Path:
    project = repository(directory)
    write_top_module(project)
    write_source(project / "lib" / "c.py")
    commit_all(project)
    return project


def edit_and_add_a_module(project: Path) -> None:
    edit_top_module(project)
    write_source(project / "lib" / "untracked.py")


def stash_with_what_git_does_not_track(project: Path) -> None:
    run_git(project, "stash", "--quiet", "--include-untracked")


def pop_the_stash(project: Path) -> None:
    run_git(project, "stash", "pop", "--quiet")


def close_the_library_but_to_its_owner_s_passing(project: Path) -> None:
    """Execute only: its files can be read by name, never listed, nor watched."""
    (project / "lib").chmod(EXECUTE_ONLY)


def edit_the_library_s_module(project: Path) -> None:
    with (project / "lib" / "c.py").open("a") as module:
        module.write(MORE_SOURCE)


def open_the_library_again(project: Path) -> None:
    (project / "lib").chmod(READABLE)


def make_the_root_again(project: Path) -> None:
    remove_tree(project)
    write_top_module(project)


def write_beside_in_the_root_made_again(project: Path) -> None:
    write_source(project / "brand_new.py")


def repository_without_info(directory: Path) -> Path:
    project = committed_repository(directory)
    shutil.rmtree(project / ".git" / "info")
    return project


def write_an_untracked_module_beside(project: Path) -> None:
    write_source(project / "beside.py")


def exclude_the_module_beside_from_info(project: Path) -> None:
    """An excludes file made, which git reads for what it does not track."""
    info = project / ".git" / "info"
    info.mkdir()
    (info / "exclude").write_text("beside.py\n")


def make_a_repository_in_the_library(project: Path) -> None:
    """In a directory already followed: what git does not track in it is no longer listed."""
    make_repository(project / "lib")


def make_a_repository_in_the_fresh_directory(project: Path) -> None:
    """In a directory followed that holds only what git does not track: git no longer looks
    into it."""
    make_repository(project / "fresh")


def untrack_the_library(project: Path) -> None:
    """Git looked into the repository in it for what it tracked there, and no longer does."""
    run_git(project, "rm", "-r", "--cached", "--quiet", "lib")


def write_a_module_in_the_library(project: Path) -> None:
    write_source(project / "lib" / "more.py")


def repository_of_compiled_files(directory: Path) -> Path:
    """A package whose directory holds only files git ignores so far, as after a checkout."""
    project = repository(directory)
    (project / ".gitignore").write_text("*.pyc\n")
    write_source(project / "package" / "module.pyc")
    commit_all(project)
    return project


def write_the_package_s_module(project: Path) -> None:
    write_source(project / "package" / "module.py")


def repository_nesting_a_repository(directory: Path) -> Path:
    project = committed_repository(directory)
    nest_a_repository(project)
    return project


def link_to_the_skipped_module(project: Path) -> None:
    """A link git lists to a file it ignores: only the link is kept, read through it."""
    (project / ".gitignore").write_text("skipped.py\n")
    write_skipped_module(project)
    (project / "visible.py").symlink_to("skipped.py")


def edit_the_skipped_module(project: Path) -> None:
    with (project / "skipped.py").open("a") as module:
        module.write(MORE_SOURCE)


def repository_beside_a_library(directory: Path) -> Path:
    """A repository, and beside it another to add as its submodule."""
    library = make_repository(directory / "library")
    write_source(library / "library.py")
    commit_all(library)
    return committed_repository(directory)


def add_the_library_as_a_submodule(project: Path) -> None:
    run_git(
        project,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "--quiet",
        str(project.parent / "library"),
        "library",
    )


def add_a_module_to_the_submodule(project: Path) -> None:
    """Its own index changes, which lies in the repository's git directory, unwatched."""
    write_source(project / "library" / "more.py")
    run_git(project / "library", "add", "more.py")


def worktree_of_a_repository_without_info(directory: Path) -> Path:
    """A linked worktree whose repository has no `info` yet, where its excludes would lie."""
    worktree = worktree_of_a_repository_ignoring_a_module(directory)
    shutil.rmtree(directory / "project" / ".git" / "info", ignore_errors=True)
    return worktree


def exclude_the_module_beside_from_the_main_repository_s_info(project: Path) -> None:
    info = project.parent / "project" / ".git" / "info"
    info.mkdir()
    (info / "exclude").write_text("beside.py\n")


def move_the_main_repository_away(project: Path) -> None:
    """Its git directory moved, nothing changed in the worktree, which git no longer reads."""
    (project.parent / "project").replace(project.parent / "moved_away")


def repository_with_a_twin_outside(directory: Path) -> Path:
    """A module that has a second name outside the project, a hard link."""
    project = committed_repository(directory)
    os.link(project / "a.py", directory / "twin.py")
    return project


def edit_the_top_module_through_its_twin(project: Path) -> None:
    """An event goes to the directory of the name written through alone."""
    with (project.parent / "twin.py").open("a") as twin:
        twin.write(MORE_SOURCE)


def exclude_beside_by_the_repository_s_configuration(project: Path) -> None:
    excludes = project.parent / "repository-excludes"
    excludes.write_text("beside.py\n")
    run_git(project, "config", "core.excludesFile", str(excludes))


def rename_the_library_in_case(project: Path) -> None:
    """On a volume that ignores case, as macOS' and Windows' do, its old name still finds it."""
    (project / "lib").rename(project / "Lib")


def edit_the_library_s_module_by_its_new_case(project: Path) -> None:
    with (project / "Lib" / "c.py").open("a") as module:
        module.write(MORE_SOURCE)


def edit_the_nested_module(project: Path) -> None:
    with (project / "nested" / "inner.py").open("a") as module:
        module.write(MORE_SOURCE)


class Case(BaseModel):
    """A project, and the changes made to it, each followed by a look."""

    make_project: Callable[[Path], Path]
    steps: list[Operation]


def looked_at_case(case: Case, directory: Path) -> LookedAt:
    project = case.make_project(directory)
    files = WatchedFiles(discovery_for(Settings(), build_chunker()), open_watcher, SetClock())
    files.current(project)
    looked = LookedAt(kept=[], scanned=[])
    for step in case.steps:
        step(project)
        looked.kept.append(files.current(project))
        looked.scanned.append(scan_of(project, Settings()))
    files.close()
    return looked


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            Case(
                make_project=repository,
                steps=[ignore_directories_and_names, write_skipped_module],
            ),
            id="a new file git ignores",
        ),
        pytest.param(
            Case(
                make_project=repository,
                steps=[write_deep_module, ignore_directories_and_names, force_add_everything],
            ),
            id="a file git tracks since, nothing written",
        ),
        pytest.param(
            Case(
                make_project=worktree_of_a_repository_ignoring_a_module,
                steps=[add_the_skipped_module_by_force],
            ),
            id="a linked worktree's index",
        ),
        pytest.param(
            Case(
                make_project=worktree_of_a_repository_ignoring_a_module,
                steps=[ignore_the_top_module_too, untrack_the_top_module],
            ),
            id="a tracked module of a linked worktree untracked and ignored",
        ),
        pytest.param(
            Case(
                make_project=repository_with_a_branch_of_other_files,
                steps=[check_out_the_other_branch, check_out_the_branch_before],
            ),
            id="a branch checked out and back",
        ),
        pytest.param(
            Case(
                make_project=committed_repository,
                steps=[edit_and_add_a_module, stash_with_what_git_does_not_track, pop_the_stash],
            ),
            id="a stash with what git does not track, and back",
        ),
        pytest.param(
            Case(
                make_project=committed_repository,
                steps=[
                    close_the_library_but_to_its_owner_s_passing,
                    edit_the_library_s_module,
                    open_the_library_again,
                    edit_the_library_s_module,
                ],
            ),
            id="a directory nothing may watch",
        ),
        pytest.param(
            Case(
                make_project=committed_repository,
                steps=[make_the_root_again, write_beside_in_the_root_made_again],
            ),
            id="the root made again",
        ),
        pytest.param(
            Case(
                make_project=repository_without_info,
                steps=[write_an_untracked_module_beside, exclude_the_module_beside_from_info],
            ),
            id="git's info made since, with an exclude",
        ),
        pytest.param(
            Case(
                make_project=repository_of_compiled_files,
                steps=[write_the_package_s_module],
            ),
            id="a directory of ignored files only gets a module",
        ),
        pytest.param(
            Case(
                make_project=repository_nesting_a_repository,
                steps=[remove_the_nested_repository_s_git, edit_the_nested_module],
            ),
            id="a nested repository's .git removed",
        ),
        pytest.param(
            Case(
                make_project=committed_repository,
                steps=[link_to_the_skipped_module, edit_the_skipped_module],
            ),
            id="a link's target git ignores edited",
        ),
        pytest.param(
            Case(
                make_project=repository_beside_a_library,
                steps=[add_the_library_as_a_submodule, add_a_module_to_the_submodule],
            ),
            id="a submodule added, and a module to it",
        ),
        pytest.param(
            Case(
                make_project=worktree_of_a_repository_ignoring_a_module,
                steps=[move_the_main_repository_away],
            ),
            id="a linked worktree's main repository moved",
            marks=pytest.mark.skipif(
                sys.platform == WINDOWS_PLATFORM,
                reason="Windows refuses to move a directory holding one watched",
            ),
        ),
        pytest.param(
            Case(
                make_project=worktree_of_a_repository_without_info,
                steps=[
                    write_an_untracked_module_beside,
                    exclude_the_module_beside_from_the_main_repository_s_info,
                ],
            ),
            id="a linked worktree's excludes made since",
        ),
        pytest.param(
            Case(
                make_project=committed_repository,
                steps=[make_a_repository_in_the_library, write_a_module_in_the_library],
            ),
            id="a repository made in a directory followed",
        ),
        pytest.param(
            Case(
                make_project=committed_repository,
                steps=[write_in_a_new_directory, make_a_repository_in_the_fresh_directory],
            ),
            id="a repository made in a directory of untracked files",
        ),
        pytest.param(
            Case(
                make_project=committed_repository,
                steps=[
                    make_a_repository_in_the_library,
                    untrack_the_library,
                    write_a_module_in_the_library,
                ],
            ),
            id="a repository git looked into untracked",
        ),
        pytest.param(
            Case(
                make_project=committed_repository,
                steps=[link_to_a_text_file_not_yet_written, write_the_later_text_file],
            ),
            id="a link's target written after it",
        ),
        pytest.param(
            Case(
                make_project=repository_with_a_twin_outside,
                steps=[edit_the_top_module_through_its_twin],
            ),
            id="a module edited through its other name",
        ),
        pytest.param(
            Case(
                make_project=committed_repository,
                steps=[
                    write_an_untracked_module_beside,
                    exclude_beside_by_the_repository_s_configuration,
                ],
            ),
            id="an excludes file named in the repository's configuration",
        ),
        pytest.param(
            Case(
                make_project=committed_repository,
                steps=[rename_the_library_in_case, edit_the_library_s_module_by_its_new_case],
            ),
            id="a directory renamed only in case",
        ),
    ],
)
def test_what_git_and_the_file_system_decide_the_files_followed_follow(case: Case, tmp_path: Path):
    """Changes a scan sees only by asking git or the file system, too rare in a sequence drawn
    at random."""
    looked = looked_at_case(case, tmp_path)

    assert looked.kept == looked.scanned


@pytest.mark.parametrize(
    ("make_project", "project_settings"),
    [
        pytest.param(repository, Settings(), id="listed by git"),
        pytest.param(plain_directory, WALKED, id="walked, as set"),
    ],
)
def test_a_name_taken_by_another_kind_is_followed(
    make_project: Callable[[Path], Path], project_settings: Settings
):
    """A module's name taken by a directory and back: a tree's system tells the name alone."""
    looked = looked_at(
        make_project,
        [make_a_name_a_module, make_a_name_a_package, make_a_name_a_module],
        project_settings,
    )

    assert looked.kept == looked.scanned


def test_a_look_after_an_edit_reads_what_changed_without_asking_git(tmp_path: Path):
    project = committed_repository(tmp_path)
    discovery = RecordingDiscovery(Settings())
    files = WatchedFiles(discovery, open_watcher, SetClock())
    files.current(project)
    asked_at_start = list(discovery.checked)

    edit_top_module(project)
    kept = files.current(project)
    files.close()

    assert (discovery.listings, discovery.checked, kept) == (
        1,
        asked_at_start,
        scan_of(project, Settings()),
    )


def test_git_rewriting_its_index_lists_the_files_again_without_watching_them_again(
    tmp_path: Path,
):
    """As `git status` does after an edit: listing them again is git's part of a scan alone."""
    project = committed_repository(tmp_path)
    discovery = RecordingDiscovery(Settings())
    instances = CountedWatchers()
    files = WatchedFiles(discovery, instances, SetClock())
    files.current(project)

    write_skipped_module(project)
    run_git(project, "add", "skipped.py")
    kept = files.current(project)
    files.close()

    assert (instances.opened, discovery.listings, kept) == (1, 2, scan_of(project, Settings()))


def test_new_modules_at_once_are_told_by_git_s_listing_again_once(tmp_path: Path):
    """However many: git's listing is never patched one path at a time."""
    project = committed_repository(tmp_path)
    discovery = RecordingDiscovery(Settings())
    files = WatchedFiles(discovery, open_watcher, SetClock())
    files.current(project)

    for number in range(MANY_MODULES):
        write_source(project / f"generated_{number}.py")
    kept = files.current(project)
    files.close()

    assert (discovery.listings, kept) == (2, scan_of(project, Settings()))


def test_what_git_ignores_changed_as_the_following_starts_is_followed(tmp_path: Path):
    """Changed in a directory not yet watched, it sends no event."""
    project = committed_repository(tmp_path)
    (project / ".gitignore").write_text("generated/\n")
    write_source(project / "generated" / "old.py")
    files = WatchedFiles(
        IgnoreRulesChangedAsAsked(Settings(), project / ".gitignore"), open_watcher, SetClock()
    )
    files.current(project)
    files.current(project)

    write_source(project / "generated" / "new.py")
    kept = files.current(project)
    files.close()

    assert kept == scan_of(project, Settings())


def test_a_walked_project_s_root_made_again_is_followed_again(tmp_path: Path):
    """Where no git directory goes with it, the root's own going is all there is."""
    project = plain_directory(tmp_path)
    write_top_module(project)
    files = WatchedFiles(discovery_for(WALKED, build_chunker()), open_watcher, SetClock())
    files.current(project)

    make_the_root_again(project)
    files.current(project)
    write_beside_in_the_root_made_again(project)
    kept = files.current(project)
    files.close()

    assert kept == scan_of(project, WALKED)


def test_a_walked_project_s_directory_moved_out_keeps_none_of_its_files(tmp_path: Path):
    """Its files send nothing once it is gone: only its going tells."""
    project = plain_directory(tmp_path)
    write_deep_module(project)
    files = WatchedFiles(discovery_for(WALKED, build_chunker()), open_watcher, SetClock())
    files.current(project)

    move_a_directory_out(project)
    kept = files.current(project)
    files.close()

    assert kept == scan_of(project, WALKED)


def test_a_link_git_ignores_is_listed_once(tmp_path: Path):
    """Not at every look, as a link git lists is read."""
    project = committed_repository(tmp_path)
    (project / ".gitignore").write_text("hidden.py\n")
    discovery = RecordingDiscovery(Settings())
    files = WatchedFiles(discovery, open_watcher, SetClock())
    files.current(project)
    (project / "hidden.py").symlink_to("a.py")
    files.current(project)

    edit_top_module(project)
    files.current(project)
    files.close()

    assert discovery.listings == 2


@pytest.mark.parametrize(
    "exchange",
    [
        pytest.param(exchanged_by_renameat2, marks=ONLY_ON_LINUX),
        pytest.param(exchanged_by_renamex_np, marks=ONLY_ON_MACOS),
    ],
)
def test_two_directories_swapped_are_followed_where_they_lie(
    exchange: Callable[[Path, Path], int], tmp_path: Path
):
    """Each one's going and the other's coming arrive under one name at once."""
    project = committed_repository(tmp_path)
    write_deep_module(project)
    files = WatchedFiles(discovery_for(Settings(), build_chunker()), open_watcher, SetClock())
    files.current(project)

    swapped = exchange(project / "lib", project / "src")
    files.current(project)
    write_source(project / "lib" / "after.py")
    write_source(project / "src" / "after.py")
    kept = files.current(project)
    files.close()

    assert (swapped, kept) == (0, scan_of(project, Settings()))


def test_a_renamed_directory_is_watched_no_more_once_followed_where_it_lies(tmp_path: Path):
    """Its old watch is forgotten with its record, so no later look watches anything again."""
    project = committed_repository(tmp_path)
    watcher = RecordingWatcher()
    files = WatchedFiles(discovery_for(Settings(), build_chunker()), lambda: watcher, SetClock())
    files.current(project)
    (project / "lib").replace(project / "library")
    files.current(project)
    watcher.watched.clear()

    for _ in range(EDITS_LOOKED_AT):
        files.current(project)
    files.close()

    assert watcher.watched == set()


def test_files_no_event_tells_of_are_found_by_the_check_against_a_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """As the user's global excludes, changed: the follower sees it once the interval passed."""
    excludes = exclude_by_the_global_configuration(tmp_path, monkeypatch)
    project = committed_repository(tmp_path)
    write_source(project / "beside.py")
    clock = SetClock()
    files = WatchedFiles(discovery_for(Settings(), build_chunker()), open_watcher, clock)
    files.current(project)

    excludes.write_text("beside.py\n")
    clock.now = START_OF_TIME + CHECK_INTERVAL_SECONDS
    kept = files.current(project)
    files.close()

    assert kept == scan_of(project, Settings())


def exclude_by_the_global_configuration(directory: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The user's global excludes file, empty for now, named by a configuration of their own:
    outside every project, no event of it reaches a following."""
    excludes = directory / "global-excludes"
    excludes.write_text("")
    configuration = directory / "global-configuration"
    configuration.write_text(f"[core]\n\texcludesFile = {excludes.as_posix()}\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(configuration))
    return excludes


async def test_a_forced_index_run_reads_the_files_by_a_full_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, store_class: StoreClass
):
    excludes = exclude_by_the_global_configuration(tmp_path, monkeypatch)
    project = committed_repository(tmp_path)
    write_source(project / "beside.py")
    indexing = indexing_service(
        open_store(store_class, tmp_path / "index", IDENTITY), Settings(), build_chunker()
    )
    await indexing.index(project, force=False, progress=NO_PROGRESS)

    excludes.write_text("beside.py\n")
    report = await indexing.index(project, force=True, progress=NO_PROGRESS)

    assert report.files_deleted == 1


def test_a_rescan_starts_the_following_again(tmp_path: Path):
    """As a forced index run asks, for what no event tells."""
    project = committed_repository(tmp_path)
    instances = CountedWatchers()
    files = WatchedFiles(discovery_for(Settings(), build_chunker()), instances, SetClock())
    files.current(project)

    kept = files.rescanned(project)
    files.close()

    assert (instances.opened, kept) == (2, scan_of(project, Settings()))


def test_every_edit_made_before_a_look_is_in_it(tmp_path: Path):
    """However soon the look comes, as a search right after an edit does."""
    project = committed_repository(tmp_path)
    files = WatchedFiles(discovery_for(Settings(), build_chunker()), open_watcher, SetClock())
    files.current(project)

    looks = []
    for _ in range(EDITS_LOOKED_AT):
        edit_top_module(project)
        looks.append(files.current(project) == scan_of(project, Settings()))
    files.close()

    assert looks == [True] * EDITS_LOOKED_AT


def test_reading_every_file_queues_nothing(tmp_path: Path):
    """As `rg` over a repository does: its reads would fill the queue a search reads."""
    write_top_module(tmp_path)
    watcher = open_watcher()
    watcher.watch(str(tmp_path))
    # What FSEvents may still hand of the module's writing, made before the watch
    watcher.events()

    for _ in range(READS_OF_EVERY_FILE):
        (tmp_path / "a.py").read_text()
    events = watcher.events()
    watcher.close()

    assert events == []


@ONLY_ON_LINUX
def test_a_queue_that_overflowed_starts_the_following_again(tmp_path: Path):
    """Filled by writes to two files in turn, which inotify cannot fold into one event each,
    then a module made, whose event the full queue drops."""
    project = committed_repository(tmp_path)
    instances = CountedWatchers()
    files = WatchedFiles(discovery_for(Settings(), build_chunker()), instances, SetClock())
    files.current(project)

    with (project / "a.py").open("a") as first, (project / "lib" / "c.py").open("a") as second:
        for _ in range(int(QUEUE_LIMIT_FILE.read_text())):
            for module in (first, second):
                module.write(MORE_SOURCE)
                module.flush()
    write_source(project / "after_the_flood.py")
    kept = files.current(project)
    files.close()

    assert (instances.opened, kept) == (2, scan_of(project, Settings()))


@ONLY_ON_WINDOWS
def test_changes_past_the_buffer_are_kept_between_looks(tmp_path: Path):
    """Files of long names made and deleted, more than the buffer between two reads holds, as a
    build's burst under `target/`, then a module made: the tree's thread keeps them all."""
    project = committed_repository(tmp_path)
    instances = CountedWatchers()
    files = WatchedFiles(discovery_for(Settings(), build_chunker()), instances, SetClock())
    files.current(project)

    for _ in range(FILES_FILLING_THE_BUFFER):
        (project / LONG_NAME).write_text(SOURCE)
        (project / LONG_NAME).unlink()
    write_source(project / "after_the_flood.py")
    kept = files.current(project)
    files.close()

    assert (instances.opened, kept) == (1, scan_of(project, Settings()))


@ONLY_ON_WINDOWS
def test_changes_past_what_a_tree_keeps_start_the_following_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """As an install's flood: no cheaper to follow than to scan again."""
    monkeypatch.setattr(directory_changes, "MOST_CHANGES_KEPT", FEW_CHANGES_KEPT)
    project = committed_repository(tmp_path)
    instances = CountedWatchers()
    files = WatchedFiles(discovery_for(Settings(), build_chunker()), instances, SetClock())
    files.current(project)

    for number in range(FEW_CHANGES_KEPT):
        write_source(project / f"flood_{number}.py")
    kept = files.current(project)
    files.close()

    assert (instances.opened, kept) == (2, scan_of(project, Settings()))


@ONLY_ON_MACOS
def test_a_mark_the_system_will_not_make_starts_the_following_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """As with the watcher's directory of marks removed: a following started again makes its
    marks anew, where one given up would scan for the rest of the session."""
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    # The watcher's marks in the test's own directory, apart from a server's that runs
    monkeypatch.setattr(tempfile, "tempdir", str(temporary))
    project = committed_repository(tmp_path)
    instances = CountedWatchers()
    files = WatchedFiles(discovery_for(Settings(), build_chunker()), instances, SetClock())
    files.current(project)

    for marks in temporary.glob(f"{MARKS_PREFIX}*"):
        remove_tree(marks)
    files.current(project)
    edit_top_module(project)
    kept = files.current(project)
    files.close()

    assert (instances.opened, kept) == (2, scan_of(project, Settings()))


def test_a_directory_made_as_the_following_starts_is_followed(tmp_path: Path):
    """Made after the walk listed its parent, before the files were listed."""
    project = committed_repository(tmp_path)
    files = WatchedFiles(
        DirectoryMadeAsListed(Settings(), project / "late"), open_watcher, SetClock()
    )
    files.current(project)

    write_source(project / "late" / "new.py")
    kept = files.current(project)
    files.close()

    assert kept == scan_of(project, Settings())


def test_a_directory_made_as_its_parent_is_watched_is_followed(tmp_path: Path):
    """Made after its parent's watch was asked for, before its parent was listed."""
    project = committed_repository(tmp_path)
    late = project / "late"
    files = WatchedFiles(
        discovery_for(Settings(), build_chunker()),
        partial(DirectoryMadeAsWatched, {str(project.resolve()): [late]}),
        SetClock(),
    )
    files.current(project)

    write_source(late / "new.py")
    kept = files.current(project)
    files.close()

    assert kept == scan_of(project, Settings())


@pytest.mark.parametrize(
    "unwatched",
    [
        pytest.param(["node_modules/package", "build/out", "generated/out"], id="names in UTF-8"),
        pytest.param(
            ["node_modules/package", "build/out", "generated/out", NOT_UTF8_NAME],
            id="a name not in UTF-8",
            marks=NAMES_NOT_UTF8,
        ),
    ],
)
def test_only_the_directories_a_scan_reads_and_git_s_own_are_watched(
    unwatched: list[str | bytes], tmp_path: Path
):
    """Each watch comes from a limit the user's editor shares: none for a directory a pattern
    covers, git ignores, whose name the index cannot hold, inside git's or a nested repository's
    own."""
    project = committed_repository(tmp_path)
    (project / ".gitignore").write_text("generated/\ncache/\n")
    for directory in ["src/deep", *unwatched]:
        write_source(project / os.fsdecode(directory) / "module.py")
    nest_a_repository(project)
    (project / "nested" / "inside").mkdir()
    watcher = RecordingWatcher()
    files = WatchedFiles(discovery_for(Settings(), build_chunker()), lambda: watcher, SetClock())
    files.current(project)

    write_source(project / "src" / "cache" / "entries" / "entry.py")
    write_source(project / "src" / "__pycache__" / "module.py")
    files.current(project)
    files.close()

    root = project.resolve()
    assert watcher.watched == {
        str(root / directory)
        for directory in ["", "lib", "src", "src/deep", "nested", ".git", ".git/info"]
    }


def test_a_project_past_the_limit_of_watches_is_scanned_without_trying_again(tmp_path: Path):
    project = committed_repository(tmp_path)
    instances = CountedWatchersAtTheirLimit()
    files = WatchedFiles(discovery_for(Settings(), build_chunker()), instances, SetClock())
    files.current(project)

    edit_top_module(project)
    kept = files.current(project)
    files.close()

    assert (instances.opened, kept) == (1, scan_of(project, Settings()))


def repository_s_directory(directory: Path) -> Path:
    """A directory below a repository's root, whose listing directories above it decide."""
    project = committed_repository(directory)
    return project / "lib"


def repository_with_submodules(directory: Path) -> Path:
    project = committed_repository(directory)
    (project / ".gitmodules").write_text("")
    return project


@pytest.mark.parametrize(
    "make_project",
    [
        pytest.param(plain_directory, id="outside a repository"),
        pytest.param(repository_s_directory, id="below a repository's root"),
        pytest.param(repository_with_submodules, id="with submodules"),
    ],
)
def test_a_project_no_watch_can_tell_is_scanned_at_each_look(
    make_project: Callable[[Path], Path], tmp_path: Path
):
    project = make_project(tmp_path)
    instances = CountedWatchers()
    files = WatchedFiles(discovery_for(Settings(), build_chunker()), instances, SetClock())
    files.current(project)

    write_source(project / "new.py")
    kept = files.current(project)

    assert (instances.opened, kept) == (0, scan_of(project, Settings()))


def open_descriptors() -> int:
    """How many files this process holds open."""
    return len(list(OPEN_DESCRIPTORS.iterdir()))


@ONLY_ON_WINDOWS
async def test_a_session_ended_lets_go_of_its_projects_directories(tmp_path: Path):
    """Windows refuses to move a directory holding one watched: once the session ends, the
    directory holding the project moves."""
    holder = tmp_path / "holder"
    holder.mkdir()
    project = committed_repository(holder)
    settings = Settings(cache_dir=tmp_path / "cache")

    async with project_registry_lifespan(
        create_app(settings, FakeEmbedder()),
        settings=settings,
        embedder=FakeEmbedder(),
        open_store=store_opener(settings),
    ) as registry:
        registry.services_for(project).files.current(project)
    holder.rename(tmp_path / "moved")

    assert (tmp_path / "moved" / "project").is_dir()


@ONLY_ON_LINUX
async def test_a_session_ended_stops_following_its_projects(tmp_path: Path):
    project = committed_repository(tmp_path)
    settings = Settings(cache_dir=tmp_path / "cache")
    before = open_descriptors()

    async with project_registry_lifespan(
        create_app(settings, FakeEmbedder()),
        settings=settings,
        embedder=FakeEmbedder(),
        open_store=store_opener(settings),
    ) as registry:
        registry.services_for(project).files.current(project)
        following = open_descriptors()

    assert (following - before, open_descriptors() - before) == (1, 0)
