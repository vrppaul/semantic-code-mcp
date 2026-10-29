"""Source file discovery — which files of a project get indexed.

It runs before every search, to see what changed, so its cost is per file and kept to the
file system's: each directory is judged and resolved once for all its files, the ignore
patterns are one expression of each kind, and a path is resolved whole only where it is a
symlink (ADR 009, D4).
"""

import os
import re
import stat
import subprocess  # nosec B404
from collections.abc import Sequence
from fnmatch import translate
from pathlib import Path

import structlog
from pydantic import BaseModel, ConfigDict

from semantic_code_mcp.domain.models import FileStamp, FileStamps

log = structlog.get_logger()

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
# Joins the expressions of several patterns into one that matches where any of them does
PATTERN_ALTERNATIVE = "|"
# An expression that matches nothing, for no patterns: a lookahead of nothing never fails,
# so its negation always does
NO_PATTERN = "(?!)"
# The path from the project root of the root itself, the directory of its top-level files
ROOT_DIRECTORY = ""
# What starts a file's extension, and the extension of a name without one
EXTENSION_DOT = "."
NO_EXTENSION = ""


class IgnorePatterns(BaseModel):
    """Ignore patterns sorted by how they match."""

    model_config = ConfigDict(frozen=True)

    # Matched against each path component: a directory or a file name at any depth
    component: tuple[str, ...]
    # Matched against the path from the project root and each of its leading directories
    path: tuple[str, ...]


def is_utf8(name: str) -> bool:
    """Whether a name decoded from the file system is UTF-8; one that is not holds the
    surrogates `os.fsdecode` puts for its bytes, which no index column can store."""
    try:
        name.encode()
    except UnicodeEncodeError:
        return False
    return True


def joined_pattern(patterns: Sequence[str]) -> re.Pattern[str]:
    """One expression that matches a whole name or path where any of the patterns does, as
    `fnmatchcase` reads each; for no patterns, one that matches nothing."""
    expressions = [translate(pattern) for pattern in patterns] or [NO_PATTERN]
    return re.compile(PATTERN_ALTERNATIVE.join(expressions))


def extension_of(relative_path: str) -> str:
    """A file's extension as `Path.suffix` gives it, read off its path without building a
    `Path`: from the last dot of its name once the dots it starts with are set aside, so
    `.bashrc` and `..py` have none."""
    name = relative_path.rpartition(PATH_SEPARATOR)[2].lstrip(EXTENSION_DOT)
    _, dot, extension = name.rpartition(EXTENSION_DOT)
    return dot + extension if dot else NO_EXTENSION


def joined_path(directory: str, name: str) -> str:
    """The path from the project root of a name in a directory given by its own."""
    return name if directory == ROOT_DIRECTORY else f"{directory}{PATH_SEPARATOR}{name}"


class FoundFile(BaseModel):
    """A candidate that is a file inside the project: where it really lives, whether its own
    path is a symlink, its path from the project root as POSIX text, and its stamp."""

    model_config = ConfigDict(frozen=True)

    # Its real path as text, which two paths to one file share
    target: str
    through_link: bool
    path: str
    stamp: FileStamp


class IgnoreRules:
    """The ignore patterns as one discovery reads them (`SourceDiscovery` says how): a file is
    covered when a pattern covers its name, its path or a directory it lies in, and each
    directory's verdict is worked out once, for every file and directory under it."""

    def __init__(self, component_pattern: re.Pattern[str], path_pattern: re.Pattern[str]) -> None:
        self._component_pattern = component_pattern
        self._path_pattern = path_pattern
        self._directory_verdicts: dict[str, bool] = {}

    def covers_file(self, relative_path: str) -> bool:
        """Whether a pattern covers a file, by its path from the project root."""
        directory, _, name = relative_path.rpartition(PATH_SEPARATOR)
        return (
            self._component_pattern.match(name) is not None
            or self._path_pattern.match(relative_path) is not None
            or self.covers_directory(directory)
        )

    def covers_directory(self, relative_directory: str) -> bool:
        """Whether a pattern covers a directory: its name, its path, its path with a slash
        after it, which `build/**` matches, or a directory it lies in."""
        if relative_directory == ROOT_DIRECTORY:
            return False
        verdict = self._directory_verdicts.get(relative_directory)
        if verdict is None:
            parent, _, name = relative_directory.rpartition(PATH_SEPARATOR)
            verdict = (
                self._component_pattern.match(name) is not None
                or self._path_pattern.match(relative_directory) is not None
                or self._path_pattern.match(relative_directory + PATH_SEPARATOR) is not None
                or self.covers_directory(parent)
            )
            self._directory_verdicts[relative_directory] = verdict
        return verdict


class CandidateCheck:
    """What one discovery finds of its candidates on disk, each given by its path from the
    project root: a directory is resolved once, for all the files in it, from where its parent
    really lives."""

    def __init__(self, root: Path) -> None:
        self._root = root
        # The root as text, which a candidate's path joins without a `Path` built per file
        self._root_text = str(root)
        # Each directory met as it really lives, as text; None for one outside the root
        self._real_directories: dict[str, str | None] = {}

    def found(self, relative_path: str) -> FoundFile | None:
        """A candidate that is a file and resolves inside the root, with its record as of now."""
        path = f"{self._root_text}{os.sep}{relative_path}"
        try:
            own_status = os.lstat(path)
            through_link = stat.S_ISLNK(own_status.st_mode)
            status = Path(path).stat() if through_link else own_status
        except OSError:
            return None
        if not stat.S_ISREG(status.st_mode):
            return None
        target = self._link_target(path) if through_link else self._directory_target(relative_path)
        if target is None:
            log.debug("skipped_file_outside_project", file_path=path, project=self._root_text)
            return None
        return FoundFile(
            target=target,
            through_link=through_link,
            path=relative_path,
            stamp=FileStamp(mtime_ns=status.st_mtime_ns, size_bytes=status.st_size),
        )

    def _link_target(self, path: str) -> str | None:
        """Where a symlink leads, resolved whole; None outside the root."""
        target = Path(path).resolve()
        return str(target) if target.is_relative_to(self._root) else None

    def _directory_target(self, relative_path: str) -> str | None:
        """Where a file that is no symlink really lives: in its directory as resolved once for
        all its files, as git lists the tracked files of a directory a symlink has since
        replaced; None in a directory that resolves outside the root."""
        directory, _, name = relative_path.rpartition(PATH_SEPARATOR)
        real_directory = self._real_directory(directory)
        return None if real_directory is None else f"{real_directory}{os.sep}{name}"

    def _real_directory(self, directory: str) -> str | None:
        """Where a directory under the root really lives, as text: inside where its parent
        really lives, resolved whole only where it is a symlink, so a directory costs one
        `lstat` where resolving its path costs one a component; None outside the root, or
        where it cannot be read."""
        if directory == ROOT_DIRECTORY:
            return self._root_text
        if directory not in self._real_directories:
            parent, _, name = directory.rpartition(PATH_SEPARATOR)
            real_parent = self._real_directory(parent)
            self._real_directories[directory] = (
                None if real_parent is None else self._real_entry(f"{real_parent}{os.sep}{name}")
            )
        return self._real_directories[directory]

    def _real_entry(self, path: str) -> str | None:
        """Where an entry of a directory that really lives where its path says lives itself:
        there, unless it is a symlink; None where it leads outside the root or cannot be read."""
        try:
            through_link = stat.S_ISLNK(os.lstat(path).st_mode)
        except OSError:
            return None
        return self._link_target(path) if through_link else path


class ProjectRoot(BaseModel):
    """The root of the project a directory belongs to, and whether a git repository names
    it: a work tree's root that does not hold the home directory. Otherwise it is the
    directory itself, which nothing says is one project rather than a folder of many."""

    model_config = ConfigDict(frozen=True)

    path: Path
    is_repository: bool


def parse_ignore_patterns(raw_patterns: Sequence[str]) -> IgnorePatterns:
    """Sort patterns by how they match, after the spellings .gitignore allows.

    A trailing `/` is dropped; a leading `/` or `./` anchors the pattern at the project
    root; a leading `**/` also matches at the root itself. These are the user's settings,
    read as `fnmatch` reads them, classes and all; a search's globs, which agents write,
    are another grammar on purpose (`domain/globs.py`).
    """
    component: list[str] = []
    path: list[str] = []
    for raw_pattern in raw_patterns:
        pattern = raw_pattern.rstrip(PATH_SEPARATOR)
        if pattern.startswith(CURRENT_DIRECTORY_PREFIX):
            pattern = PATH_SEPARATOR + pattern.removeprefix(CURRENT_DIRECTORY_PREFIX)
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
        git_timeout_seconds: float,
        home_directory: Path,
    ) -> None:
        self._extensions = frozenset(extensions)
        patterns = parse_ignore_patterns(ignore_patterns)
        self._component_pattern = joined_pattern(patterns.component)
        self._path_pattern = joined_pattern(patterns.path)
        self._use_gitignore = use_gitignore
        self._git_timeout_seconds = git_timeout_seconds
        # Resolved as git resolves the work tree roots it is compared with
        self._home_directory = home_directory.resolve()

    def discover(self, project_root: Path) -> FileStamps:
        """The indexable files under a project root, by their paths from it as POSIX text and
        in those paths' order, each with its stamp as found now."""
        root = project_root.resolve()
        rules = IgnoreRules(self._component_pattern, self._path_pattern)
        candidates = self._list_with_git(root) if self._use_gitignore else None
        if candidates is None:
            candidates = self._walk(root, rules)
        return self._indexable(candidates, root, rules)

    def project_root_of(self, directory: Path) -> ProjectRoot:
        """The root of the project a directory belongs to: of the git work tree it lies
        in, so a repository has one index whichever of its directories a search names; the
        directory itself outside a work tree, where the repository around it ignores it, or
        where that repository's root is the user's home directory or above it, as a home
        under git for its dotfiles is, which holds projects rather than being one; and
        whether a repository names it. Asked of git whether or not it lists the files."""
        repository_root = self._repository_of(directory)
        if repository_root is None or self._home_directory.is_relative_to(repository_root):
            return ProjectRoot(path=directory, is_repository=False)
        return ProjectRoot(path=repository_root, is_repository=True)

    def _list_with_git(self, root: Path) -> list[str] | None:
        """Ask git which files to index, by their paths from root; None when git cannot or
        should not answer."""
        if self._repository_of(root) is None:
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
        return [os.fsdecode(entry) for entry in entries.split(GIT_PATH_SEPARATOR) if entry]

    def _repository_of(self, root: Path) -> Path | None:
        """The root of the git work tree root lies in; None outside one, when git cannot
        answer, or when that repository ignores root itself."""
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
        return repository_root

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

    def _walk(self, root: Path, rules: IgnoreRules) -> list[str]:
        """The path from root of every file under it, pruning the directories a pattern
        ignores.

        `Path.walk` does not descend into symlinked directories.
        """
        found: list[str] = []
        for directory, subdirectory_names, file_names in root.walk():
            relative_directory = (
                ROOT_DIRECTORY if directory == root else directory.relative_to(root).as_posix()
            )
            subdirectory_names[:] = [
                name
                for name in subdirectory_names
                if not rules.covers_directory(joined_path(relative_directory, name))
            ]
            found.extend(joined_path(relative_directory, name) for name in file_names)
        return found

    def _indexable(self, candidates: list[str], root: Path, rules: IgnoreRules) -> FileStamps:
        """The candidates worth indexing, each file once, sorted by their paths' text.

        The checks that need no file system run first, over every candidate; only what
        passes them is stat'ed. A file reached both directly and through a symlink keeps
        its direct path.
        """
        check = CandidateCheck(root)
        wanted = sorted(path for path in set(candidates) if self._is_wanted(path, rules))
        found = [file for file in map(check.found, wanted) if file is not None]
        # Direct paths first, so a file reached both ways keeps its direct path
        kept_by_target: dict[str, FoundFile] = {}
        for file in sorted(found, key=lambda file: file.through_link):
            kept_by_target.setdefault(file.target, file)
        return {file.path: file.stamp for file in found if kept_by_target[file.target] is file}

    def _is_wanted(self, relative_path: str, rules: IgnoreRules) -> bool:
        """Whether a candidate has a supported extension, a name the index can hold, and no
        pattern ignores it."""
        if extension_of(relative_path) not in self._extensions:
            return False
        if rules.covers_file(relative_path):
            return False
        if not is_utf8(relative_path):
            log.warning("skipped_file_name_not_utf8", file_path=ascii(relative_path))
            return False
        return True
