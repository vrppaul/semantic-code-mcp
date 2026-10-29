"""Tests of directories watched within trees a system watches whole: each change a tree tells is
told as inotify would tell it, by the directory watched that holds it and its name, and by what
lies at its path now; a watch is refused where inotify would refuse it."""

import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from semantic_code_mcp.domain.models import NO_WATCH, WatchChange, WatchEvent
from semantic_code_mcp.watching.systems import WINDOWS_PLATFORM
from semantic_code_mcp.watching.trees import TreeAction, TreeChange
from tests.fakes import ScriptedTrees

# A directory its owner may pass through but not list
EXECUTE_ONLY = 0o311
READABLE = 0o755


def watched_project(directory: Path) -> Path:
    """A project of a module at its root and one in `src`, by its real path."""
    project = directory.resolve() / "project"
    (project / "src").mkdir(parents=True)
    (project / "a.py").write_text("")
    (project / "src" / "b.py").write_text("")
    return project


def told(watcher: ScriptedTrees, tree: Path, path: Path, action: TreeAction) -> list[WatchEvent]:
    """What the watcher tells once a tree told one change of a path."""
    watcher.trees[str(tree)].told.append(TreeChange(path=str(path), action=action))
    return watcher.events()


def event(watch: int, change: WatchChange, name: str, *, is_directory: bool) -> WatchEvent:
    return WatchEvent(watch=watch, change=change, name=name, is_directory=is_directory)


def leave_it(_project: Path) -> None:
    """Nothing changed since the tree told it."""


def remove_the_source_directory(project: Path) -> None:
    (project / "src" / "b.py").unlink()
    (project / "src").rmdir()


def make_a_package(project: Path) -> None:
    (project / "package").mkdir()


def write_a_module(project: Path) -> None:
    (project / "c.py").write_text("")


def test_a_change_is_told_by_the_directory_watched_that_holds_it(tmp_path: Path):
    project = watched_project(tmp_path)
    watcher = ScriptedTrees()
    watcher.watch(str(project))
    source = watcher.watch(str(project / "src"))

    events = told(watcher, project, project / "src" / "b.py", TreeAction.written)

    assert events == [event(source, WatchChange.entry_changed, "b.py", is_directory=False)]


def test_a_change_below_no_directory_watched_is_passed_over(tmp_path: Path):
    """As no inotify watch would have told it: in a directory a scan does not read."""
    project = watched_project(tmp_path)
    watcher = ScriptedTrees()
    watcher.watch(str(project))

    events = told(watcher, project, project / "src" / "b.py", TreeAction.written)

    assert events == []


def test_changes_lost_are_told_as_a_queue_overflowed(tmp_path: Path):
    project = watched_project(tmp_path)
    watcher = ScriptedTrees()
    watcher.watch(str(project))

    events = told(watcher, project, Path(), TreeAction.lost)

    assert events == [event(NO_WATCH, WatchChange.queue_overflowed, "", is_directory=False)]


@pytest.mark.parametrize(
    ("change_since", "expected"),
    [
        pytest.param(
            leave_it,
            [WatchChange.entry_removed, WatchChange.entry_added],
            id="a directory there now, walked again",
        ),
        pytest.param(remove_the_source_directory, [WatchChange.entry_removed], id="none there"),
    ],
)
def test_a_directory_watched_made_or_removed_is_gone(
    change_since: Callable[[Path], None], expected: list[WatchChange], tmp_path: Path
):
    """Removed and made again, swapped, or moved away and back, it may hold what no change
    below it told."""
    project = watched_project(tmp_path)
    watcher = ScriptedTrees()
    root = watcher.watch(str(project))
    watcher.watch(str(project / "src"))
    change_since(project)

    events = told(watcher, project, project / "src", TreeAction.made_or_removed)

    assert events == [event(root, change, "src", is_directory=True) for change in expected]


@pytest.mark.parametrize(
    ("change_since", "name", "expected", "is_directory"),
    [
        pytest.param(write_a_module, "c.py", WatchChange.entry_added, False, id="a file there"),
        pytest.param(
            make_a_package, "package", WatchChange.entry_added, True, id="a directory there"
        ),
        pytest.param(leave_it, "gone.py", WatchChange.entry_removed, False, id="nothing there"),
    ],
)
def test_an_entry_made_or_removed_is_what_lies_at_its_path_now(
    change_since: Callable[[Path], None],
    name: str,
    expected: WatchChange,
    is_directory: bool,
    tmp_path: Path,
):
    """Whatever the system folded together: made then removed, removed then made, renamed
    from or to the path."""
    project = watched_project(tmp_path)
    watcher = ScriptedTrees()
    root = watcher.watch(str(project))
    change_since(project)

    events = told(watcher, project, project / name, TreeAction.made_or_removed)

    assert events == [event(root, expected, name, is_directory=is_directory)]


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        pytest.param(TreeAction.written, [], id="written, as its entries changed"),
        pytest.param(
            TreeAction.attributes_changed,
            [WatchChange.entry_changed],
            id="its attributes changed, walked again",
        ),
    ],
)
def test_a_directory_s_own_change_is_told_only_where_it_may_change_what_is_read(
    action: TreeAction, expected: list[WatchChange], tmp_path: Path
):
    """Its entries' changes tell themselves; its permissions may change what may be read below
    it."""
    project = watched_project(tmp_path)
    watcher = ScriptedTrees()
    root = watcher.watch(str(project))
    watcher.watch(str(project / "src"))

    events = told(watcher, project, project / "src", action)

    assert events == [event(root, change, "src", is_directory=True) for change in expected]


def test_a_path_spelled_otherwise_is_found_by_where_its_directory_lies(tmp_path: Path):
    """As another case of its letters, or a firmlink of macOS, spells it: here a symlink."""
    project = watched_project(tmp_path)
    (tmp_path / "spelled").symlink_to(project / "src", target_is_directory=True)
    watcher = ScriptedTrees()
    watcher.watch(str(project))
    source = watcher.watch(str(project / "src"))

    events = told(watcher, project, tmp_path / "spelled" / "b.py", TreeAction.written)

    assert events == [event(source, WatchChange.entry_changed, "b.py", is_directory=False)]


def test_a_tree_opened_above_another_takes_its_place_its_changes_told_first(tmp_path: Path):
    """As the project's root above its own `.git`, watched first."""
    project = watched_project(tmp_path)
    watcher = ScriptedTrees()
    source = watcher.watch(str(project / "src"))
    watcher.trees[str(project / "src")].told.append(
        TreeChange(path=str(project / "src" / "b.py"), action=TreeAction.written)
    )
    watcher.watch(str(project))

    events = told(watcher, project, project / "a.py", TreeAction.written)

    assert (sorted(watcher.trees), watcher.trees[str(project / "src")].closed, events) == (
        sorted([str(project), str(project / "src")]),
        True,
        [
            event(source, WatchChange.entry_changed, "b.py", is_directory=False),
            event(source + 1, WatchChange.entry_changed, "a.py", is_directory=False),
        ],
    )


def test_a_directory_a_tree_holds_opens_none_of_its_own(tmp_path: Path):
    project = watched_project(tmp_path)
    watcher = ScriptedTrees()
    watcher.watch(str(project))
    watcher.watch(str(project / "src"))

    assert list(watcher.trees) == [str(project)]


def link_to_the_sources(project: Path) -> Path:
    (project / "linked").symlink_to(project / "src", target_is_directory=True)
    return project / "linked"


def the_module(project: Path) -> Path:
    return project / "a.py"


@pytest.mark.parametrize(
    "not_a_directory", [pytest.param(link_to_the_sources), pytest.param(the_module)]
)
def test_what_is_no_directory_itself_is_not_watched(
    not_a_directory: Callable[[Path], Path], tmp_path: Path
):
    """As inotify watches no symlink to a directory, nor a file."""
    project = watched_project(tmp_path)
    watcher = ScriptedTrees()

    with pytest.raises(NotADirectoryError):
        watcher.watch(str(not_a_directory(project)))


@pytest.mark.skipif(sys.platform == WINDOWS_PLATFORM, reason="a directory of Windows lists always")
def test_a_directory_that_may_not_be_listed_is_not_watched(tmp_path: Path):
    """As inotify refuses it: what git lists in it is read at every look."""
    project = watched_project(tmp_path)
    (project / "src").chmod(EXECUTE_ONLY)
    watcher = ScriptedTrees()

    with pytest.raises(PermissionError):
        watcher.watch(str(project / "src"))
    (project / "src").chmod(READABLE)
