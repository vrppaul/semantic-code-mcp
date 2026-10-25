"""Source file discovery — which files of a project get indexed."""

import os
import subprocess  # nosec B404
from collections.abc import Sequence
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath

import structlog

log = structlog.get_logger()

# Git normally answers in milliseconds; past this, discovery walks the file system instead.
DEFAULT_GIT_TIMEOUT_SECONDS = 10.0

# `git ls-files -z` separates paths with NUL, so any file name survives unquoted.
GIT_PATH_SEPARATOR = b"\0"

# An ignore pattern containing this separator is matched against the path from the
# project root; one without it is matched against each path component.
PATTERN_PATH_SEPARATOR = "/"


class SourceDiscovery:
    """Finds the files of a project that the chunkers can index.

    With `use_gitignore`, asks git for tracked files and for untracked files that
    .gitignore does not exclude. Without it, outside a git work tree, or when git is
    missing or fails, walks the file system. Either way the ignore patterns apply,
    symlinked directories are not followed, and files that resolve outside the
    project root are skipped.

    Ignore patterns follow .gitignore in spirit: a pattern without a slash matches
    any path component (`node_modules` skips every directory of that name, at any
    depth); a pattern with a slash matches the path from the project root or one of
    its leading directories (`build/**` skips the top-level build directory only).
    """

    def __init__(
        self,
        extensions: Sequence[str],
        ignore_patterns: Sequence[str],
        use_gitignore: bool,
        git_timeout_seconds: float = DEFAULT_GIT_TIMEOUT_SECONDS,
    ) -> None:
        self._extensions = frozenset(extensions)
        normalized_patterns = [
            pattern.rstrip(PATTERN_PATH_SEPARATOR) for pattern in ignore_patterns
        ]
        self._component_patterns = [
            pattern for pattern in normalized_patterns if PATTERN_PATH_SEPARATOR not in pattern
        ]
        self._path_patterns = [
            pattern for pattern in normalized_patterns if PATTERN_PATH_SEPARATOR in pattern
        ]
        self._use_gitignore = use_gitignore
        self._git_timeout_seconds = git_timeout_seconds

    def discover(self, project_root: Path) -> list[Path]:
        """List the indexable files under a project root, as sorted absolute paths."""
        root = project_root.resolve()
        candidates = self._list_with_git(root) if self._use_gitignore else None
        if candidates is None:
            candidates = self._walk(root)
        return sorted(path for path in candidates if self._is_indexable(path, root))

    def _list_with_git(self, root: Path) -> list[Path] | None:
        """Ask git for tracked and non-ignored untracked files; None when git cannot answer."""
        pathspecs = [f"*{extension}" for extension in sorted(self._extensions)]
        command = [
            "git",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--deduplicate",
            "--",
            *pathspecs,
        ]
        try:
            completed = subprocess.run(  # nosec B603, B607
                command,
                cwd=root,
                capture_output=True,
                timeout=self._git_timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            log.info("git_listing_unavailable", project=str(root), error=str(error))
            return None

        if completed.returncode != 0:
            log.debug(
                "git_listing_failed",
                project=str(root),
                returncode=completed.returncode,
                stderr=completed.stderr.decode(errors="replace").strip(),
            )
            return None

        entries = completed.stdout.split(GIT_PATH_SEPARATOR)
        return [root / os.fsdecode(entry) for entry in entries if entry]

    def _walk(self, root: Path) -> list[Path]:
        """List every file under root, pruning directories that a name pattern ignores.

        `Path.walk` does not descend into symlinked directories.
        """
        found: list[Path] = []
        for directory, subdirectory_names, file_names in root.walk():
            subdirectory_names[:] = [
                name for name in subdirectory_names if not self._matches_component(name)
            ]
            found.extend(directory / name for name in file_names)
        return found

    def _is_indexable(self, path: Path, root: Path) -> bool:
        """Whether a candidate is a supported, existing file inside root that no pattern
        ignores."""
        if path.suffix not in self._extensions or not path.is_file():
            return False
        if not path.resolve().is_relative_to(root):
            log.debug("skipped_file_outside_project", file_path=str(path), project=str(root))
            return False
        return not self._is_ignored(PurePosixPath(path.relative_to(root).as_posix()))

    def _is_ignored(self, relative_path: PurePosixPath) -> bool:
        """Whether an ignore pattern covers a path relative to the project root."""
        if any(self._matches_component(part) for part in relative_path.parts):
            return True
        leading_paths = (
            PurePosixPath(*relative_path.parts[:depth]).as_posix()
            for depth in range(1, len(relative_path.parts) + 1)
        )
        return any(
            fnmatchcase(leading_path, pattern)
            for leading_path in leading_paths
            for pattern in self._path_patterns
        )

    def _matches_component(self, name: str) -> bool:
        """Whether a single path component matches a pattern without a slash."""
        return any(fnmatchcase(name, pattern) for pattern in self._component_patterns)
