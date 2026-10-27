"""Tests for source file discovery."""

import os
import tempfile
from pathlib import Path

import pytest
from hypothesis import example, given, settings, strategies

from semantic_code_mcp.discovery import ProjectRoot, SourceDiscovery
from semantic_code_mcp.domain.models import SourceFile
from tests.helpers import commit_all, make_repository, run_git

PYTHON_EXTENSIONS = (".py",)
NO_IGNORE_PATTERNS: tuple[str, ...] = ()
SHORT_GIT_TIMEOUT_SECONDS = 0.2
GIT_TIMEOUT_SECONDS = 10.0
# A home directory no repository of a test lies at or above
HOME_OF_NO_TEST = Path("/home-of-no-test")


def make_discovery(
    ignore_patterns: tuple[str, ...] = NO_IGNORE_PATTERNS,
    use_gitignore: bool = True,
    home_directory: Path = HOME_OF_NO_TEST,
) -> SourceDiscovery:
    return SourceDiscovery(
        extensions=PYTHON_EXTENSIONS,
        ignore_patterns=ignore_patterns,
        use_gitignore=use_gitignore,
        git_timeout_seconds=GIT_TIMEOUT_SECONDS,
        home_directory=home_directory,
    )


def relative_names(files: list[SourceFile]) -> set[str]:
    """The paths discovery gives, each relative to the project root it was asked about."""
    return {file.path.as_posix() for file in files}


class TestGitDiscovery:
    """A project inside a git work tree."""

    def test_finds_tracked_and_untracked_files(self, tmp_path: Path):
        repository = make_repository(tmp_path / "repo")
        (repository / "tracked.py").write_text("x = 1\n")
        commit_all(repository)
        (repository / "untracked.py").write_text("y = 2\n")

        files = make_discovery().discover(repository)

        assert relative_names(files) == {"tracked.py", "untracked.py"}

    def test_skips_gitignored_files(self, tmp_path: Path):
        repository = make_repository(tmp_path / "repo")
        (repository / ".gitignore").write_text("generated/\n*_pb2.py\n")
        (repository / "code.py").write_text("x = 1\n")
        (repository / "message_pb2.py").write_text("x = 1\n")
        (repository / "generated").mkdir()
        (repository / "generated" / "model.py").write_text("x = 1\n")

        files = make_discovery().discover(repository)

        assert relative_names(files) == {"code.py"}

    def test_returns_non_ascii_file_names_as_existing_paths(self, tmp_path: Path):
        repository = make_repository(tmp_path / "repo")
        (repository / "тест.py").write_text("x = 1\n")
        commit_all(repository)

        files = make_discovery().discover(repository)

        assert relative_names(files) == {"тест.py"}
        assert all((repository / file.path).is_file() for file in files)

    def test_subdirectory_of_a_repository_honours_gitignore(self, tmp_path: Path):
        repository = make_repository(tmp_path / "repo")
        (repository / ".gitignore").write_text("scratch.py\n")
        package = repository / "package"
        package.mkdir()
        (package / "module.py").write_text("x = 1\n")
        (package / "scratch.py").write_text("x = 1\n")
        (repository / "outside.py").write_text("x = 1\n")

        files = make_discovery().discover(package)

        assert relative_names(files) == {"module.py"}

    def test_linked_worktree_honours_gitignore(self, tmp_path: Path):
        repository = make_repository(tmp_path / "repo")
        (repository / ".gitignore").write_text("scratch.py\n")
        (repository / "module.py").write_text("x = 1\n")
        commit_all(repository)
        worktree = tmp_path / "worktree"
        run_git(repository, "worktree", "add", "--quiet", str(worktree))
        (worktree / "scratch.py").write_text("x = 1\n")

        files = make_discovery().discover(worktree)

        assert relative_names(files) == {"module.py"}

    def test_ignore_patterns_apply_to_git_results(self, tmp_path: Path):
        repository = make_repository(tmp_path / "repo")
        (repository / "code.py").write_text("x = 1\n")
        (repository / "docs").mkdir()
        (repository / "docs" / "conf.py").write_text("x = 1\n")
        commit_all(repository)

        files = make_discovery(ignore_patterns=("docs",)).discover(repository)

        assert relative_names(files) == {"code.py"}

    def test_without_gitignore_finds_ignored_files(self, tmp_path: Path):
        """An allowlist .gitignore (`/*` plus exceptions) must not hide source files."""
        repository = make_repository(tmp_path / "repo")
        (repository / ".gitignore").write_text("/*\n!/.gitignore\n!/listed.py\n")
        (repository / "listed.py").write_text("x = 1\n")
        (repository / "unlisted.py").write_text("x = 1\n")

        files = make_discovery(use_gitignore=False).discover(repository)

        assert relative_names(files) == {"listed.py", "unlisted.py"}

    def test_project_ignored_by_an_enclosing_repository_is_walked(self, tmp_path: Path):
        """A home directory under git for dotfiles must not hide the projects inside it."""
        home = make_repository(tmp_path / "home")
        (home / ".gitignore").write_text("*\n!.gitignore\n")
        project = home / "projects" / "app"
        project.mkdir(parents=True)
        (project / "code.py").write_text("x = 1\n")

        files = make_discovery().discover(project)

        assert relative_names(files) == {"code.py"}

    def test_ignored_subdirectory_of_a_repository_is_walked(self, tmp_path: Path):
        repository = make_repository(tmp_path / "repo")
        (repository / ".gitignore").write_text("scratch/\n")
        experiment = repository / "scratch" / "experiment"
        experiment.mkdir(parents=True)
        (experiment / "trial.py").write_text("x = 1\n")

        files = make_discovery().discover(experiment)

        assert relative_names(files) == {"trial.py"}

    def test_files_inside_a_submodule_are_found(self, tmp_path: Path):
        library = make_repository(tmp_path / "library")
        (library / "lib.py").write_text("x = 1\n")
        commit_all(library)
        application = make_repository(tmp_path / "application")
        (application / "app.py").write_text("x = 1\n")
        run_git(
            application,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "--quiet",
            str(library),
            "vendor/library",
        )
        commit_all(application)

        files = make_discovery().discover(application)

        assert relative_names(files) == {"app.py", "vendor/library/lib.py"}

    def test_symlink_to_a_file_inside_the_project_is_listed_once(self, tmp_path: Path):
        repository = make_repository(tmp_path / "repo")
        (repository / "real.py").write_text("x = 1\n")
        (repository / "alias.py").symlink_to(repository / "real.py")
        commit_all(repository)

        files = make_discovery().discover(repository)

        assert relative_names(files) == {"real.py"}

    def test_git_variables_in_the_environment_do_not_redirect_the_listing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        other = make_repository(tmp_path / "other")
        (other / "foreign.py").write_text("x = 1\n")
        commit_all(other)
        repository = make_repository(tmp_path / "repo")
        (repository / "code.py").write_text("x = 1\n")
        monkeypatch.setenv("GIT_DIR", str(other / ".git"))
        monkeypatch.setenv("GIT_WORK_TREE", str(other))

        files = make_discovery().discover(repository)

        assert relative_names(files) == {"code.py"}


class TestGitUnavailable:
    """Discovery falls back to walking the file system when git cannot answer."""

    def test_missing_git_binary_falls_back_to_walk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        repository = make_repository(tmp_path / "repo")
        (repository / "code.py").write_text("x = 1\n")
        empty_bin = tmp_path / "empty-bin"
        empty_bin.mkdir()
        monkeypatch.setenv("PATH", str(empty_bin))

        files = make_discovery().discover(repository)

        assert relative_names(files) == {"code.py"}

    def test_hanging_git_falls_back_to_walk(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        repository = make_repository(tmp_path / "repo")
        (repository / "code.py").write_text("x = 1\n")
        fake_bin = tmp_path / "fake-bin"
        fake_bin.mkdir()
        fake_git = fake_bin / "git"
        fake_git.write_text("#!/bin/sh\nexec sleep 5\n")
        fake_git.chmod(0o755)
        monkeypatch.setenv("PATH", str(fake_bin))
        discovery = SourceDiscovery(
            extensions=PYTHON_EXTENSIONS,
            ignore_patterns=NO_IGNORE_PATTERNS,
            use_gitignore=True,
            git_timeout_seconds=SHORT_GIT_TIMEOUT_SECONDS,
            home_directory=HOME_OF_NO_TEST,
        )

        files = discovery.discover(repository)

        assert relative_names(files) == {"code.py"}


class TestWalkDiscovery:
    """The file-system walk, which `use_gitignore=False` asks for."""

    def test_outside_a_work_tree_discovery_walks(self, tmp_path: Path):
        project = tmp_path / "project"
        project.mkdir()
        (project / "code.py").write_text("x = 1\n")

        files = make_discovery().discover(project)

        assert relative_names(files) == {"code.py"}

    def test_finds_files_with_supported_extensions(self, tmp_path: Path):
        project = tmp_path / "project"
        (project / "package").mkdir(parents=True)
        (project / "main.py").write_text("x = 1\n")
        (project / "package" / "module.py").write_text("x = 1\n")
        (project / "notes.txt").write_text("text\n")

        files = make_discovery(use_gitignore=False).discover(project)

        assert relative_names(files) == {"main.py", "package/module.py"}

    def test_name_pattern_matches_directories_at_any_depth(self, tmp_path: Path):
        project = tmp_path / "project"
        nested = project / "frontend" / "node_modules" / "lib"
        nested.mkdir(parents=True)
        (nested / "vendored.py").write_text("x = 1\n")
        (project / "code.py").write_text("x = 1\n")

        files = make_discovery(ignore_patterns=("node_modules",), use_gitignore=False).discover(
            project
        )

        assert relative_names(files) == {"code.py"}

    def test_path_pattern_is_anchored_at_the_project_root(self, tmp_path: Path):
        project = tmp_path / "project"
        (project / "build").mkdir(parents=True)
        (project / "tools" / "build").mkdir(parents=True)
        (project / "build" / "artifact.py").write_text("x = 1\n")
        (project / "tools" / "build" / "script.py").write_text("x = 1\n")

        files = make_discovery(ignore_patterns=("build/**",), use_gitignore=False).discover(project)

        assert relative_names(files) == {"tools/build/script.py"}

    def test_does_not_follow_symlinked_directories(self, tmp_path: Path):
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "foreign.py").write_text("x = 1\n")
        project = tmp_path / "project"
        project.mkdir()
        (project / "code.py").write_text("x = 1\n")
        (project / "linked").symlink_to(elsewhere, target_is_directory=True)

        files = make_discovery(use_gitignore=False).discover(project)

        assert relative_names(files) == {"code.py"}

    def test_pattern_spellings_follow_gitignore(self, tmp_path: Path):
        """A leading `/` or `./` anchors at the root, a trailing `/` is dropped, and a
        leading `**/` matches at the root too."""
        project = tmp_path / "project"
        for directory in (
            "build",
            "tools/build",
            "dist",
            "lib/dist",
            "docs",
            "node_modules",
            "web/node_modules",
        ):
            (project / directory).mkdir(parents=True)
            (project / directory / "module.py").write_text("x = 1\n")
        (project / "keep.py").write_text("x = 1\n")
        patterns = ("/build", "./dist", "docs/", "**/node_modules/**")

        files = make_discovery(ignore_patterns=patterns, use_gitignore=False).discover(project)

        assert relative_names(files) == {
            "keep.py",
            "tools/build/module.py",
            "lib/dist/module.py",
        }


@pytest.mark.parametrize("use_gitignore", [True, False], ids=["asking git", "walking"])
def test_a_file_name_that_is_not_utf8_is_skipped(tmp_path: Path, use_gitignore: bool):
    """The index cannot store such a name; one of them must not stop the rest."""
    repository = make_repository(tmp_path / "repo")
    (repository / "kept.py").write_text("x = 1\n")
    (repository / os.fsdecode(b"caf\xe9.py")).write_text("x = 1\n")

    files = make_discovery(use_gitignore=use_gitignore).discover(repository)

    assert relative_names(files) == {"kept.py"}


# What a project in the discovery property holds: files at a few depths, one in a directory
# its .gitignore names, and links to a file inside, to one outside, to nothing and to a
# directory, one of them named as a source; any of them tracked, and files deleted after the
# commit
PROJECT_FILES = ("a.py", "src/b.py", "src/deep/c.py", "ignored/d.py", "notes.txt")
# The sources every way of discovery finds while they are on disk
PLAIN_SOURCES = frozenset({"a.py", "src/b.py", "src/deep/c.py"})
LINK_TARGETS = {
    "inside.py": "a.py",
    "outside.py": "../outside.py",
    "nowhere.py": "missing.py",
    "linked_src": "src",
    "linked_dir.py": "src",
}
GITIGNORE = ".gitignore"
IGNORED_DIRECTORY = "ignored/\n"
OUTSIDE_FILE = "outside.py"
DISCOVERY_EXAMPLES = 30


def write_source(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x = 1\n")


@pytest.mark.parametrize("use_gitignore", [True, False], ids=["asking git", "walking"])
@settings(max_examples=DISCOVERY_EXAMPLES, deadline=None)
@given(
    written=strategies.sets(strategies.sampled_from(PROJECT_FILES)),
    linked=strategies.sets(strategies.sampled_from(sorted(LINK_TARGETS))),
    tracked=strategies.sets(strategies.sampled_from((*PROJECT_FILES, *LINK_TARGETS))),
    deleted=strategies.sets(strategies.sampled_from(PROJECT_FILES)),
)
@example(
    written={"a.py", "src/b.py"}, linked=set(), tracked={"a.py", "src/b.py"}, deleted={"src/b.py"}
).via("a tracked file deleted from disk")
@example(
    written={"a.py"}, linked={"outside.py"}, tracked={"a.py", "outside.py"}, deleted=set()
).via("a tracked link to a file outside the project")
def test_discovery_returns_every_source_on_disk_and_only_files_inside_the_project(
    use_gitignore: bool,
    written: set[str],
    linked: set[str],
    tracked: set[str],
    deleted: set[str],
):
    """A path that is gone, or that leads outside, would be read, chunked or indexed; a
    source on disk left out would never be found."""
    with tempfile.TemporaryDirectory() as directory:
        write_source(Path(directory) / OUTSIDE_FILE)
        project = make_repository(Path(directory) / "project")
        (project / GITIGNORE).write_text(IGNORED_DIRECTORY)
        for name in written:
            write_source(project / name)
        for name in linked:
            (project / name).symlink_to(LINK_TARGETS[name])
        run_git(project, "add", "--force", "--", GITIGNORE, *sorted(tracked & (written | linked)))
        run_git(project, "commit", "--quiet", "--message", "snapshot")
        for name in deleted & written:
            (project / name).unlink()

        files = make_discovery(use_gitignore=use_gitignore).discover(project)

        returned = [project / file.path for file in files]
        every_one = [True] * len(returned)
        assert [path.is_file() for path in returned] == every_one
        assert [path.resolve().is_relative_to(project.resolve()) for path in returned] == every_one
        assert PLAIN_SOURCES & (written - deleted) <= {file.path.as_posix() for file in files}


@pytest.fixture
def directories(tmp_path: Path) -> Path:
    """A repository with a package two levels down, a directory outside any repository, a
    repository that ignores the project inside it, a large repository that hides untracked
    files for speed, a home directory under git for dotfiles and a link to it, and a
    repository that holds a home directory below its root."""
    repository = make_repository(tmp_path / "repo")
    (repository / "pkg" / "sub").mkdir(parents=True)
    (tmp_path / "plain").mkdir()
    ignoring = make_repository(tmp_path / "ignoring")
    (ignoring / ".gitignore").write_text("*\n!.gitignore\n")
    (ignoring / "projects" / "app").mkdir(parents=True)
    large = make_repository(tmp_path / "large")
    run_git(large, "config", "status.showUntrackedFiles", "no")
    (large / "pkg").mkdir()
    dotfiles = make_repository(tmp_path / "dotfiles")
    run_git(dotfiles, "config", "status.showUntrackedFiles", "no")
    (dotfiles / ".bashrc").write_text("export EDITOR=vi\n")
    run_git(dotfiles, "add", ".bashrc")
    run_git(dotfiles, "commit", "--quiet", "--message", "dotfiles")
    (dotfiles / "projects" / "tool").mkdir(parents=True)
    (dotfiles / "projects" / "tool" / "main.py").write_text("x = 1\n")
    (dotfiles / "projects" / "kept").mkdir(parents=True)
    (dotfiles / "projects" / "kept" / ".editorconfig").write_text("root = true\n")
    run_git(dotfiles, "add", "projects/kept/.editorconfig")
    run_git(dotfiles, "commit", "--quiet", "--message", "editor settings")
    (tmp_path / "linked-home").symlink_to(dotfiles)
    system = make_repository(tmp_path / "system")
    (system / "home" / "user" / "projects" / "app").mkdir(parents=True)
    (system / "srv" / "app").mkdir(parents=True)
    return tmp_path


@pytest.mark.parametrize(
    ("asked", "root", "is_repository"),
    [
        pytest.param("repo/pkg/sub", "repo", True, id="a directory inside a repository"),
        pytest.param("repo", "repo", True, id="a repository's root"),
        pytest.param("plain", "plain", False, id="a directory outside any repository"),
        pytest.param(
            "ignoring/projects/app",
            "ignoring/projects/app",
            False,
            id="a project the repository ignores",
        ),
        pytest.param("large/pkg", "large", True, id="a repository that hides untracked files"),
    ],
)
def test_a_project_is_the_whole_repository_a_directory_lies_in(
    directories: Path, asked: str, root: str, *, is_repository: bool
):
    """One index per repository: a directory inside one is a part of its project. Outside
    one, the directory is a project of its own, which no repository names."""
    found = make_discovery().project_root_of((directories / asked).resolve())

    assert found == ProjectRoot(path=(directories / root).resolve(), is_repository=is_repository)


@pytest.mark.parametrize(
    ("asked", "home"),
    [
        pytest.param("dotfiles/projects/tool", "dotfiles", id="a project it tracks nothing of"),
        pytest.param("dotfiles/projects/kept", "dotfiles", id="a project it tracks a file of"),
        pytest.param("dotfiles/projects/tool", "linked-home", id="a home through a symlink"),
        pytest.param(
            "system/home/user/projects/app", "system/home/user", id="a project in the home"
        ),
        pytest.param("system/srv/app", "system/home/user", id="a project beside the home"),
        pytest.param("dotfiles", "dotfiles", id="the home itself"),
    ],
)
def test_a_repository_at_or_above_the_home_directory_holds_projects_apart(
    directories: Path, asked: str, home: str
):
    """A home directory under git for its dotfiles would otherwise index every project of
    the user's as one, however its owner keeps untracked files out of its status."""
    discovery = make_discovery(home_directory=directories / home)

    found = discovery.project_root_of((directories / asked).resolve())

    assert found == ProjectRoot(path=(directories / asked).resolve(), is_repository=False)


def test_a_user_s_setting_that_hides_untracked_files_keeps_a_repository_whole(
    directories: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Set for speed in large repositories, it says nothing of who owns a directory."""
    user_configuration = tmp_path / "gitconfig"
    user_configuration.write_text("[status]\n\tshowUntrackedFiles = no\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(user_configuration))

    found = make_discovery().project_root_of((directories / "repo" / "pkg" / "sub").resolve())

    assert found == ProjectRoot(path=(directories / "repo").resolve(), is_repository=True)
