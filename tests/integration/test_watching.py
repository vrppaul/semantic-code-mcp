"""Tests of a project's files kept from a watcher's changes between full scans: what they hold
is what a scan finds, a look after an edit reads only what changed, and a watcher that cannot
answer leaves the project scanned."""

import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import override

import pytest
from hypothesis import given, settings, strategies
from pydantic import BaseModel

from semantic_code_mcp.app import build_chunker
from semantic_code_mcp.config import Settings
from semantic_code_mcp.discovery import SourceDiscovery
from semantic_code_mcp.domain.models import FileStamps
from semantic_code_mcp.services.files import FULL_SCAN_INTERVAL_SECONDS, WatchedFiles
from semantic_code_mcp.watching import (
    MARKERS_PREFIX,
    FileWatcher,
    WatcherLostError,
    watching_supported,
)
from tests.helpers import discovery_for, make_repository, run_git

pytestmark = pytest.mark.skipif(
    not watching_supported(), reason="a watcher's barrier holds where the system orders events"
)

# An mtime well after any file the test wrote
LATER_MTIME_SECONDS = 4_000_000_000
SOURCE = "x = 1\n"
MORE_SOURCE = "y = 2\n"
OPERATION_EXAMPLES = 30
MOST_OPERATIONS = 8
EDITS_READ = 100
# Where the clock of these tests starts
START_OF_TIME = 0.0


class RecordingDiscovery(SourceDiscovery):
    """The real discovery, keeping each root it was asked to scan whole."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(
            extensions=build_chunker().supported_extensions,
            ignore_patterns=settings.ignore_patterns,
            use_gitignore=settings.use_gitignore,
            git_timeout_seconds=settings.git_timeout_seconds,
            home_directory=Path.home(),
        )
        self.scanned: list[Path] = []

    @override
    def discover(self, project_root: Path) -> FileStamps:
        self.scanned.append(project_root)
        return super().discover(project_root)


class SetClock:
    """A clock that reads what the test set it to."""

    def __init__(self) -> None:
        self.now = START_OF_TIME

    def __call__(self) -> float:
        return self.now


def no_watcher(_root: Path, _directories: list[Path]) -> FileWatcher:
    """A watcher that cannot start, as at inotify's limit of watches."""
    message = "the limit of watches is reached"
    raise WatcherLostError(message)


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


def ignore_a_directory_and_a_name(project: Path) -> None:
    (project / ".gitignore").write_text("deep/\nskipped.py\n")


def write_skipped_module(project: Path) -> None:
    write_source(project / "skipped.py")


def link_to_a_module(project: Path) -> None:
    write_top_module(project)
    (project / "alias.py").unlink(missing_ok=True)
    (project / "alias.py").symlink_to("a.py")


def link_outside(project: Path) -> None:
    write_source(project.parent / "outside.py")
    (project / "away.py").unlink(missing_ok=True)
    (project / "away.py").symlink_to(project.parent / "outside.py")


def commit_anything(repository: Path) -> None:
    """A commit of all there is, even nothing new, which rewrites git's index."""
    run_git(repository, "add", "--all")
    run_git(repository, "commit", "--quiet", "--allow-empty", "--message", "snapshot")


def nest_a_repository(project: Path) -> None:
    write_source(project / "nested" / "inner.py")
    commit_anything(make_repository(project / "nested"))


def commit_everything(project: Path) -> None:
    commit_anything(project)


def force_add_everything(project: Path) -> None:
    """What git ignores, tracked now: its index alone changes, and git lists those files."""
    run_git(project, "add", "--all", "--force")


# A change of a project, made in place
type Operation = Callable[[Path], None]

PLAIN_OPERATIONS: list[Operation] = [
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
    ignore_a_directory_and_a_name,
    write_skipped_module,
    link_to_a_module,
    link_outside,
    nest_a_repository,
]
REPOSITORY_OPERATIONS = [*PLAIN_OPERATIONS, commit_everything, force_add_everything]


def plain_directory(directory: Path) -> Path:
    project = directory / "project"
    project.mkdir()
    return project


def repository(directory: Path) -> Path:
    return make_repository(directory / "project")


class LookedAt(BaseModel):
    """What a watcher's files held and what a scan found, after each change of a project."""

    kept: list[FileStamps]
    scanned: list[FileStamps]


def looked_at(make_project: Callable[[Path], Path], operations: list[Operation]) -> LookedAt:
    """A project made in a directory of its own and changed by each operation in turn."""
    with tempfile.TemporaryDirectory() as directory:
        project = make_project(Path(directory))
        discovery = discovery_for(Settings(), build_chunker())
        files = WatchedFiles(discovery, FileWatcher, SetClock())
        files.current(project)
        looked = LookedAt(kept=[], scanned=[])
        for operation in operations:
            operation(project)
            looked.kept.append(files.current(project))
            looked.scanned.append(discovery.discover(project))
        files.close()
    return looked


@pytest.mark.parametrize(
    ("make_project", "operations"),
    [
        pytest.param(repository, REPOSITORY_OPERATIONS, id="in a repository"),
        pytest.param(plain_directory, PLAIN_OPERATIONS, id="outside one"),
    ],
)
@settings(max_examples=OPERATION_EXAMPLES, deadline=None)
@given(data=strategies.data())
def test_the_files_a_watcher_keeps_are_those_a_scan_finds(
    make_project: Callable[[Path], Path], operations: list[Operation], data: strategies.DataObject
):
    """A change it misses would leave the index stale for good; one it invents, a file
    indexed that no scan lists."""
    done = data.draw(
        strategies.lists(strategies.sampled_from(operations), max_size=MOST_OPERATIONS)
    )

    looked = looked_at(make_project, done)

    assert looked.kept == looked.scanned


@pytest.mark.parametrize(
    "operations",
    [
        pytest.param(
            [ignore_a_directory_and_a_name, write_skipped_module], id="a new file git ignores"
        ),
        pytest.param(
            [write_deep_module, ignore_a_directory_and_a_name, force_add_everything],
            id="a file git tracks since, nothing written",
        ),
    ],
)
def test_what_git_decides_the_files_a_watcher_keeps_follow(operations: list[Operation]):
    """Changes a scan sees only by asking git, too rare in a sequence drawn at random."""
    looked = looked_at(repository, operations)

    assert looked.kept == looked.scanned


def test_a_look_after_an_edit_reads_what_changed_without_a_scan(tmp_path: Path):
    project = repository(tmp_path)
    write_top_module(project)
    discovery = RecordingDiscovery(Settings())
    files = WatchedFiles(discovery, FileWatcher, SetClock())
    files.current(project)

    edit_top_module(project)
    kept = files.current(project)
    files.close()

    assert (discovery.scanned, kept) == (
        [project],
        discovery_for(Settings(), build_chunker()).discover(project),
    )


def test_every_edit_made_before_a_read_is_in_it(tmp_path: Path):
    """However soon the read comes, as a search right after an edit does."""
    project = plain_directory(tmp_path)
    write_top_module(project)
    watcher = FileWatcher(project.resolve(), [project.resolve()])

    reads = []
    for _ in range(EDITS_READ):
        edit_top_module(project)
        reads.append("a.py" in watcher.changed())
    watcher.close()

    assert reads == [True] * EDITS_READ


def test_a_watcher_that_cannot_start_leaves_the_project_scanned(tmp_path: Path):
    project = repository(tmp_path)
    write_top_module(project)
    discovery = RecordingDiscovery(Settings())
    files = WatchedFiles(discovery, no_watcher, SetClock())
    files.current(project)

    edit_top_module(project)
    kept = files.current(project)

    assert (discovery.scanned, kept) == (
        [project, project],
        discovery_for(Settings(), build_chunker()).discover(project),
    )


def test_a_watcher_whose_markers_are_gone_is_lost_and_the_project_scanned_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A cleaner of temporary files may take its directory of markers."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    project = repository(tmp_path)
    write_top_module(project)
    discovery = RecordingDiscovery(Settings())
    files = WatchedFiles(discovery, FileWatcher, SetClock())
    files.current(project)
    [markers] = tmp_path.glob(f"{MARKERS_PREFIX}*")
    shutil.rmtree(markers)

    edit_top_module(project)
    kept = files.current(project)
    files.close()

    assert (discovery.scanned, kept) == (
        [project, project],
        discovery_for(Settings(), build_chunker()).discover(project),
    )


def test_the_files_are_scanned_whole_again_once_the_interval_has_passed(tmp_path: Path):
    """A net for what no change a watcher reports shows, as an excludes file outside it."""
    project = repository(tmp_path)
    write_top_module(project)
    discovery = RecordingDiscovery(Settings())
    clock = SetClock()
    files = WatchedFiles(discovery, FileWatcher, clock)
    files.current(project)

    clock.now = START_OF_TIME + FULL_SCAN_INTERVAL_SECONDS
    files.current(project)
    files.close()

    assert discovery.scanned == [project, project]
