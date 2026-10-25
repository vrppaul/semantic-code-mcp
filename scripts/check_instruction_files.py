"""Hold the instruction files to their size, to the paths they name and to their places.

`AGENTS.md` is read into every session an agent opens here, and the rules
of an area under `.claude/rules/` into every session that opens a file of
that area. A rule is cheap to add and nothing asks one to leave, so a file
left alone grows until it is read in part and followed less. Each is held
to its bytes, its lines and its longest line, and what no longer fits goes
where it belongs: a rule for one area to that area's file, how a thing is
built to its module's docstring, what a tool can hold to a gate.

Every path the files name is held to being in the repository. A path is a
word of code — of what stands between two backticks on one line — that
names a directory (it has a `/` in it) or ends as a file's name does, and
it is written from the repository's root. One that climbs out of the root
with `..`, or reaches what a link in the repository keeps outside it,
names nothing in it, whatever stands there. One with `*`, `?` or `[` in it
is a pattern, held to matching something: a kind of file, `**/__init__.py`,
is named so. What it matches is looked at one path at a time, and no
further than the first that will do. A word that starts with `/` is an
address the server answers, not a file.

The rules of an area are reached two ways, and both are held. Claude Code
loads a file of them when it opens a file one of its patterns matches, so
each opens with those patterns — `---`, `paths:`, a quoted pattern on each
line after, `---`, written without braces, which are not expanded here —
and each pattern matches a file, not a directory alone: without them every
session would read the rules, and for a pattern no file matches, none
would. One written from the machine's root matches no file of the
repository. Every other reader, another agent or a reviewer of a diff,
finds a file of rules by the line `AGENTS.md` gives it, so `AGENTS.md`
names every one.

`CLAUDE.md` is a symlink to `AGENTS.md`: Claude Code reads `CLAUDE.md`
alone where there is one, so a file of its own there stands in place of
the rules.

All of it is asked of the files as UTF-8 text, each read once. One that
cannot be read so — it is not written as UTF-8, or what is named as rules
is a directory, or a link to nothing — is said by name, and nothing else
is said: its bytes cannot be counted as text, nor a path read in them.

Run in the repository's root, in the project's own environment. The
pre-commit hook runs it on every commit, whatever the commit touches: a
path goes stale when the file it names is moved, not when these change.

    uv run python scripts/check_instruction_files.py

Prints each fault and exits 1, or says nothing and exits 0.
"""

import re
import sys
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, ConfigDict

# The rules every agent reads, and the name Claude Code reads them by
GENERAL_FILE = Path("AGENTS.md")
LINK = Path("CLAUDE.md")
# Where the rules of each area are kept, a file to an area, and what such a file is called
AREA_DIRECTORY = Path(".claude/rules")
AREA_FILES = "*.md"
# How the files are written down, and so how their bytes are counted
ENCODING = "utf-8"
# Lines are counted from one, as an editor counts them
FIRST_LINE = 1

# Code in a line of prose: what stands between two backticks
CODE_SPAN = re.compile(r"`([^`\n]+)`")
# What parts the directories of a path, and starts an address the server answers
SEPARATOR = "/"
# How a file's name ends here: a word of code that ends so names a file though it names no directory
FILE_SUFFIXES = frozenset({".md", ".py", ".ts", ".js", ".css", ".svelte", ".yaml", ".yml", ".toml", ".json", ".sh"})
# What makes a path a pattern for many
PATTERN_MARKS = frozenset("*?[")
# The step of a path that climbs out of the directory it is written from
LEAVES_THE_ROOT = ".."

# How the rules of an area open: the patterns of the paths they are rules for, each quoted on a line of its own
PATTERNS = "patterns"
SCOPE = re.compile(rf'\A---\npaths:\n(?P<{PATTERNS}>(?:  - "[^"\n]+"\n)+)---\n')
QUOTED = re.compile(r'"([^"\n]+)"')
# The line the first pattern stands on, under `---` and `paths:`
FIRST_PATTERN_LINE = 3

# What is said of a file that holds more than it may, and of a line that does
WHERE_IT_GOES = (
    "move a rule for one area to that area's file, how a thing is built to its module's docstring, "
    "what a tool can hold to a gate"
)
OVER_BYTES = "holds {size} bytes, {over} over its {most}: " + WHERE_IT_GOES
OVER_LINES = "holds {size} lines, {over} over its {most}: " + WHERE_IT_GOES
LINE_TOO_LONG = (
    "a line of {size} characters, {over} over the {most} a line may hold: "
    "a rule is a short imperative, so cut it or make it two"
)
# What is said of a path that names nothing, and of rules no reader would be brought to
NAMES_NOTHING = "`{named}` names nothing in the repository: write the path from the repository's root, to what is there"
NOT_NAMED = (
    f"{GENERAL_FILE} does not name this file: give it a line there, among the rules by area, saying when to read it"
)
NO_SCOPE = (
    "does not open with the paths it is rules for, so every session would read it: "
    'open it with `---`, `paths:`, a line `  - "pattern"` for each pattern, `---`'
)
MATCHES_NOTHING = "no file matches `{named}`, so no file opened brings these rules: name what is there"
# What is said of a file whose bytes are not the text it is read as, and of what is named as rules and is no file
NOT_UTF8 = "is not written as UTF-8, so nothing in it can be read or counted: save it as UTF-8"
A_DIRECTORY = (
    f"is a directory, and every `{AREA_FILES}` under {AREA_DIRECTORY} is read as the rules of an area: "
    "name it without `.md`, or remove it"
)
A_LINK_TO_NOTHING = "is a link to nothing, so there are no rules in it to read: link it to what is there, or remove it"
# What is said of the two files every reader starts from
NOT_THE_LINK = (
    f"is not a symlink to {GENERAL_FILE}, which is how Claude Code is given the rules: ln -s {GENERAL_FILE} {LINK}"
)
GENERAL_MISSING = "is not in the repository: it holds the rules every agent reads"

# What the process exits with
ALL_KEPT = 0
FAULT_FOUND = 1


class Limits(BaseModel):
    """How much a file may hold, and how long a line of it may be."""

    model_config = ConfigDict(frozen=True)

    most_bytes: int
    most_lines: int
    longest_line: int


# The general file is read into every session, so it is the shorter. A bullet is one line, so the longest
# line is the longest rule
GENERAL_LIMITS = Limits(most_bytes=6_500, most_lines=100, longest_line=240)
AREA_LIMITS = Limits(most_bytes=5_000, most_lines=100, longest_line=240)


class InstructionFile(BaseModel):
    """A file of instructions as it is read: where it is from the root, what it says, and how much it may hold."""

    model_config = ConfigDict(frozen=True)

    path: Path
    text: str
    limits: Limits


class Measure(BaseModel):
    """How much a file or a line holds, how much it may, and what is said of it when it holds more."""

    model_config = ConfigDict(frozen=True)

    size: int
    most: int
    says: str

    def said_when_over(self) -> list[str]:
        """What is said of holding too much: nothing of what fits."""
        if self.size <= self.most:
            return []
        return [self.says.format(size=self.size, over=self.size - self.most, most=self.most)]


class NamedPath(BaseModel):
    """A path a file names, as it is written, and the line it is written on."""

    model_config = ConfigDict(frozen=True)

    written: str
    line: int


class Fault(BaseModel):
    """One thing wrong with an instruction file: on a line of it, or with the file as a whole."""

    model_config = ConfigDict(frozen=True)

    path: Path
    line: int | None
    says: str

    def __str__(self) -> str:
        place = self.path if self.line is None else f"{self.path}:{self.line}"
        return f"{place}: {self.says}"


def faults_in(root: Path) -> list[Fault]:
    """Every fault in the instruction files of the repository at this root: the general file's, then each area's.

    A file that cannot be read is the one thing said: nothing else can be asked of it, nor of
    the files that are read against it.
    """
    if not (root / GENERAL_FILE).is_file():
        return [Fault(path=GENERAL_FILE, line=None, says=GENERAL_MISSING)]
    area_paths = [found.relative_to(root) for found in sorted((root / AREA_DIRECTORY).rglob(AREA_FILES))]
    read = [_read(root, GENERAL_FILE, GENERAL_LIMITS), *(_read(root, path, AREA_LIMITS) for path in area_paths)]
    unreadable = [fault for fault in read if isinstance(fault, Fault)]
    if unreadable:
        return unreadable
    # Every one of them was read, and the general file first
    general, *areas = [file for file in read if isinstance(file, InstructionFile)]
    named_in_general = frozenset(path.written for path in _paths_named(general))
    return [
        *_faults_of_size(general),
        *_faults_of_paths(general, root),
        *(fault for area in areas for fault in _faults_of_an_area(area, root, named_in_general)),
        *_faults_of_the_link(root),
    ]


def _read(root: Path, path: Path, limits: Limits) -> InstructionFile | Fault:
    """A file of instructions, read once; or what is said of one that cannot be read as the text it is asked as."""
    try:
        text = (root / path).read_text(encoding=ENCODING)
    except UnicodeDecodeError:
        return Fault(path=path, line=None, says=NOT_UTF8)
    except IsADirectoryError:
        return Fault(path=path, line=None, says=A_DIRECTORY)
    except FileNotFoundError:
        return Fault(path=path, line=None, says=A_LINK_TO_NOTHING)
    return InstructionFile(path=path, text=text, limits=limits)


def _faults_of_an_area(area: InstructionFile, root: Path, named_in_general: frozenset[str]) -> list[Fault]:
    """What is wrong with the rules of one area: with the file itself, and with how a reader is brought to it."""
    return [
        *_faults_of_size(area),
        *_faults_of_scope(area, root),
        *_faults_of_paths(area, root),
        *_faults_of_no_name(area, named_in_general),
    ]


def _faults_of_size(file: InstructionFile) -> list[Fault]:
    """What is said of a file that holds more than it may: of its bytes, of its lines, and of each line too long."""
    lines = file.text.splitlines()
    of_the_file = [
        Measure(size=len(file.text.encode(ENCODING)), most=file.limits.most_bytes, says=OVER_BYTES),
        Measure(size=len(lines), most=file.limits.most_lines, says=OVER_LINES),
    ]
    whole = [
        Fault(path=file.path, line=None, says=said) for measure in of_the_file for said in measure.said_when_over()
    ]
    too_long = [
        Fault(path=file.path, line=number, says=said)
        for number, line in enumerate(lines, start=FIRST_LINE)
        for said in Measure(size=len(line), most=file.limits.longest_line, says=LINE_TOO_LONG).said_when_over()
    ]
    return [*whole, *too_long]


def _faults_of_paths(file: InstructionFile, root: Path) -> list[Fault]:
    """What is said of each path a file names that names nothing in the repository."""
    return [
        Fault(path=file.path, line=path.line, says=NAMES_NOTHING.format(named=path.written))
        for path in _paths_named(file)
        if not _is_there(root, path.written)
    ]


def _paths_named(file: InstructionFile) -> list[NamedPath]:
    """Every path a file names in the code of its prose, in the order they are written."""
    return [
        NamedPath(written=word, line=number)
        for number, line in enumerate(file.text.splitlines(), start=FIRST_LINE)
        for code in CODE_SPAN.findall(line)
        for word in code.split()
        if _names_a_path(word)
    ]


def _names_a_path(word: str) -> bool:
    """Whether a word of code is a path of the repository: it names a directory, or ends as a file's name does."""
    if word.startswith(SEPARATOR):
        return False
    return SEPARATOR in word or Path(word).suffix in FILE_SUFFIXES


def _is_there(root: Path, named: str) -> bool:
    """Whether the repository holds what a path names: the file or directory itself, or anything a pattern matches."""
    return any(_found(root, named))


def _found(root: Path, named: str) -> Iterator[Path]:
    """What a path names in the repository, one thing at a time: itself, whatever a pattern matches, or nothing.

    A path written from the machine's root, or one that climbs out of the repository's, names nothing in
    the repository, whatever stands there; nor is anything it reaches through a link that leads outside.
    Whoever asks takes no more than they need: a pattern is not matched against every path first.
    """
    written = Path(named)
    if written.is_absolute() or LEAVES_THE_ROOT in written.parts:
        return iter(())
    reached = root.glob(named) if not PATTERN_MARKS.isdisjoint(named) else iter((root / named,))
    inside = root.resolve()
    return (path for path in reached if path.exists() and path.resolve().is_relative_to(inside))


def _matches_a_file(root: Path, pattern: str) -> bool:
    """Whether a pattern matches a file: a directory matched brings rules to nobody, for it is files that are opened."""
    return any(found.is_file() for found in _found(root, pattern))


def _faults_of_scope(area: InstructionFile, root: Path) -> list[Fault]:
    """What is said of rules that do not open with the paths they are for, or name a pattern no file matches."""
    opening = SCOPE.match(area.text)
    if opening is None:
        return [Fault(path=area.path, line=None, says=NO_SCOPE)]
    return [
        Fault(path=area.path, line=number, says=MATCHES_NOTHING.format(named=pattern))
        for number, pattern in enumerate(QUOTED.findall(opening.group(PATTERNS)), start=FIRST_PATTERN_LINE)
        if not _matches_a_file(root, pattern)
    ]


def _faults_of_no_name(area: InstructionFile, named_in_general: frozenset[str]) -> list[Fault]:
    """What is said of rules the general file does not name: none of its readers but Claude Code would find them."""
    if area.path.as_posix() in named_in_general:
        return []
    return [Fault(path=area.path, line=None, says=NOT_NAMED)]


def _faults_of_the_link(root: Path) -> list[Fault]:
    """What is said of a `CLAUDE.md` that is anything but a link to the general file."""
    link = root / LINK
    if link.is_symlink() and link.readlink() == GENERAL_FILE:
        return []
    return [Fault(path=LINK, line=None, says=NOT_THE_LINK)]


def main() -> int:
    faults = faults_in(Path.cwd())
    for fault in faults:
        print(fault)
    return FAULT_FOUND if faults else ALL_KEPT


if __name__ == "__main__":
    sys.exit(main())
