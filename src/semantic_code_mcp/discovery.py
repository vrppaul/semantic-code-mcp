"""Source file discovery — which files of a project get indexed."""

import os
import subprocess  # nosec B404
from collections.abc import Sequence
from fnmatch import fnmatchcase
from pathlib import Path

import structlog
from pydantic import BaseModel, ConfigDict

log = structlog.get_logger()

# Git normally answers in milliseconds; past this, discovery walks the file system instead.
DEFAULT_GIT_TIMEOUT_SECONDS = 10.0
# `git ls-files -z` separates paths with NUL, so any file name survives unquoted.
GIT_PATH_SEPARATOR = b"\0"
SUCCESS_EXIT_CODE = 0
# `git check-ignore` exits 0 when the path it was given is ignored.
CHECK_IGNORE_MATCHED_EXIT_CODE = 0
# Variables that point git at another repository than the one around the project.
GIT_REDIRECTING_VARIABLES = frozenset(
    {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR", "GIT_OBJECT_DIRECTORY"}
)

# How ignore patterns are spelled: see SourceDiscovery.
PATH_SEPARATOR = "/"
CURRENT_DIRECTORY_PREFIX = "./"
ANY_DEPTH_PREFIX = "**/"


class IgnorePatterns(BaseModel):
    """Ignore patterns sorted by how they match."""

    model_config = ConfigDict(frozen=True)

    # Matched against each path component: a directory or a file name at any depth
    component: tuple[str, ...]
    # Matched against the path from the project root and each of its leading directories
    path: tuple[str, ...]


def parse_ignore_patterns(raw_patterns: Sequence[str]) -> IgnorePatterns:
    """Sort patterns by how they match, after the spellings .gitignore allows.

    A leading `./` and a trailing `/` are dropped; a leading `/` anchors the pattern at
    the project root; a leading `**/` also matches at the root itself.
    """
    component: list[str] = []
    path: list[str] = []
    for raw_pattern in raw_patterns:
        pattern = raw_pattern.removeprefix(CURRENT_DIRECTORY_PREFIX).rstrip(PATH_SEPARATOR)
        if pattern.startswith(PATH_SEPARATOR):
            path.append(pattern.removeprefix(PATH_SEPARATOR))
        elif PATH_SEPARATOR not in pattern:
            component.append(pattern)
        else:
            path.append(pattern)
            if pattern.startswith(ANY_DEPTH_PREFIX):
                path.append(pattern.removeprefix(ANY_DEPTH_PREFIX))
    return IgnorePatterns(component=tuple(component), path=tuple(path))


class SourceDiscovery:
    """Finds the files of a project that the chunkers can index.

    With `use_gitignore`, asks git for tracked files (submodules included) and for
    untracked files that .gitignore does not exclude. It walks the file system instead
    without `use_gitignore`, outside a git work tree, when an enclosing repository
    ignores the project, or when git is missing or fails. Either way the ignore patterns
    apply, symlinked directories are not followed, files that resolve outside the
    project root are skipped, and a file reached twice through a symlink is listed once.
    An untracked repository nested inside the project is not looked into.

    Ignore patterns follow .gitignore in spirit (`parse_ignore_patterns`): a pattern
    without a slash matches any path component (`node_modules` skips every directory of
    that name, at any depth); a pattern with a slash matches the path from the project
    root or one of its leading directories (`build/**` skips the top-level build
    directory only). As in `fnmatch`, `*` matches across `/`.
    """

    def __init__(
        self,
        *,
        extensions: Sequence[str],
        ignore_patterns: Sequence[str],
        use_gitignore: bool,
        git_timeout_seconds: float = DEFAULT_GIT_TIMEOUT_SECONDS,
    ) -> None:
        self._extensions = frozenset(extensions)
        self._patterns = parse_ignore_patterns(ignore_patterns)
        self._use_gitignore = use_gitignore
        self._git_timeout_seconds = git_timeout_seconds

    def discover(self, project_root: Path) -> list[Path]:
        """List the indexable files under a project root, as sorted absolute paths."""
        root = project_root.resolve()
        candidates = self._list_with_git(root) if self._use_gitignore else None
        if candidates is None:
            candidates = self._walk(root)
        return self._indexable(candidates, root)

    def _list_with_git(self, root: Path) -> list[Path] | None:
        """Ask git which files to index; None when git cannot or should not answer."""
        toplevel = self._run_git(["rev-parse", "--show-toplevel"], root)
        if toplevel is None or toplevel.returncode != SUCCESS_EXIT_CODE:
            log.debug("not_a_git_work_tree", project=str(root))
            return None

        repository_root = Path(os.fsdecode(toplevel.stdout.strip())).resolve()
        if repository_root != root and self._is_ignored_by_git(root):
            log.info(
                "project_ignored_by_enclosing_repository",
                project=str(root),
                repository=str(repository_root),
            )
            return None

        pathspecs = [f"*{extension}" for extension in sorted(self._extensions)]
        tracked = self._run_git(
            ["ls-files", "-z", "--cached", "--recurse-submodules", "--", *pathspecs], root
        )
        untracked = self._run_git(
            ["ls-files", "-z", "--others", "--exclude-standard", "--", *pathspecs], root
        )
        listings = [tracked, untracked]
        if any(listing is None or listing.returncode != SUCCESS_EXIT_CODE for listing in listings):
            log.warning("git_listing_failed", project=str(root))
            return None

        entries = b"".join(listing.stdout for listing in listings if listing is not None)
        return [root / os.fsdecode(entry) for entry in entries.split(GIT_PATH_SEPARATOR) if entry]

    def _is_ignored_by_git(self, root: Path) -> bool:
        """Whether the repository around root ignores root itself."""
        check = self._run_git(["check-ignore", "--quiet", "."], root)
        return check is not None and check.returncode == CHECK_IGNORE_MATCHED_EXIT_CODE

    def _run_git(
        self, arguments: list[str], root: Path
    ) -> subprocess.CompletedProcess[bytes] | None:
        """Run git in root, its answer whatever the exit code; None when git could not run.

        Variables that would point git at another repository are left out of its environment.
        """
        environment = {
            name: value
            for name, value in os.environ.items()
            if name not in GIT_REDIRECTING_VARIABLES
        }
        try:
            return subprocess.run(  # nosec B603, B607
                ["git", *arguments],
                cwd=root,
                env=environment,
                capture_output=True,
                timeout=self._git_timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            log.warning("git_unavailable", project=str(root), error=str(error))
            return None

    def _walk(self, root: Path) -> list[Path]:
        """List every file under root, pruning the directories a pattern ignores.

        `Path.walk` does not descend into symlinked directories.
        """
        found: list[Path] = []
        for directory, subdirectory_names, file_names in root.walk():
            subdirectory_names[:] = [
                name
                for name in subdirectory_names
                if not self._is_ignored_directory(directory / name, root)
            ]
            found.extend(directory / name for name in file_names)
        return found

    def _indexable(self, candidates: list[Path], root: Path) -> list[Path]:
        """The candidates worth indexing, each file once, sorted.

        A file reached both directly and through a symlink keeps its direct path.
        """
        # A stable sort by "is a symlink" keeps the path order within direct paths and links
        direct_paths_first = sorted(sorted(set(candidates)), key=Path.is_symlink)
        kept_by_target: dict[Path, Path] = {}
        for path in direct_paths_first:
            target = self._indexable_target(path, root)
            if target is not None:
                kept_by_target.setdefault(target, path)
        return sorted(kept_by_target.values())

    def _indexable_target(self, path: Path, root: Path) -> Path | None:
        """The file a candidate resolves to, when it is supported, not ignored, existing
        and inside root; None otherwise. The cheap checks run first."""
        if path.suffix not in self._extensions:
            return None
        if self._is_ignored(path.relative_to(root).as_posix()):
            return None
        if not path.is_file():
            return None
        target = path.resolve()
        if not target.is_relative_to(root):
            log.debug("skipped_file_outside_project", file_path=str(path), project=str(root))
            return None
        return target

    def _is_ignored(self, relative_path: str) -> bool:
        """Whether an ignore pattern covers a file's path relative to the project root."""
        parts = relative_path.split(PATH_SEPARATOR)
        if any(self._matches_component(part) for part in parts):
            return True
        leading_paths = [PATH_SEPARATOR.join(parts[:depth]) for depth in range(1, len(parts) + 1)]
        return any(
            fnmatchcase(leading_path, pattern)
            for leading_path in leading_paths
            for pattern in self._patterns.path
        )

    def _is_ignored_directory(self, directory: Path, root: Path) -> bool:
        """Whether a directory met during the walk is ignored as a whole, so it is pruned.

        Its parents were already checked on the way down, so its own path is enough.
        """
        if self._matches_component(directory.name):
            return True
        relative_path = directory.relative_to(root).as_posix()
        return any(
            fnmatchcase(relative_path, pattern)
            or fnmatchcase(relative_path + PATH_SEPARATOR, pattern)
            for pattern in self._patterns.path
        )

    def _matches_component(self, name: str) -> bool:
        """Whether a single path component matches a pattern without a slash."""
        return any(fnmatchcase(name, pattern) for pattern in self._patterns.component)
