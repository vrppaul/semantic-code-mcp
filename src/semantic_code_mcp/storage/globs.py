"""A path glob as the regular expression the store's filter matches a chunk's path with.

The store's `regexp_like` reads Rust's regex syntax, which for what is written here is
Python's. A SQL `LIKE` would not do: its `%` crosses every `/` that a `*` stops at, and a
`%` or `_` in a path would match what it stands for in `LIKE`.

A glob reads as in a `.gitignore`: one without a slash names a file or a directory at any
depth, one with a slash a path from the project root; `*` and `?` stay within a
directory, `**` crosses any number of them, and a directory named takes everything under it.
"""

import re
from pathlib import Path

PATH_SEPARATOR = "/"
# The glob's wildcards, longest first, and what each reads as
ANY_DIRECTORIES_GLOB = "**/"
ANY_DIRECTORIES = "(?:.*/)?"
ANY_DEPTH_GLOB = "**"
ANY_DEPTH = ".*"
WITHIN_A_DIRECTORY_GLOB = "*"
WITHIN_A_DIRECTORY = "[^/]*"
ONE_CHARACTER_GLOB = "?"
ONE_CHARACTER = "[^/]"
# A class of characters, `[ab]`, or of every other, `[!ab]`
CLASS_START = "["
CLASS_END = "]"
GLOB_CLASS_NEGATION = "!"
REGEX_CLASS_NEGATION = "^"
# What a regex reads inside a class unless escaped: an escape, a nested class, and the
# start of Rust's set operations (`&&`, `~~`)
CLASS_SPECIAL_CHARACTERS = frozenset("\\[&~")
ESCAPE = "\\"
# A path the glob names, and everything under it when that path is a directory
WHOLE_PATH_START = "^"
AND_BELOW = "(?:/.*)?"
WHOLE_PATH_END = "$"
NOT_FOUND = -1


def glob_pattern(glob: str) -> str:
    """The regular expression that matches the whole of every path the glob names."""
    rooted = glob.startswith(PATH_SEPARATOR) or PATH_SEPARATOR in glob.strip(PATH_SEPARATOR)
    leading = "" if rooted else ANY_DIRECTORIES
    body = glob_body(glob.strip(PATH_SEPARATOR))
    return WHOLE_PATH_START + leading + body + AND_BELOW + WHOLE_PATH_END


def glob_body(glob: str) -> str:
    """The glob's characters as a regular expression, its wildcards read, the rest literal."""
    parts: list[str] = []
    index = 0
    while index < len(glob):
        if glob.startswith(ANY_DIRECTORIES_GLOB, index):
            parts.append(ANY_DIRECTORIES)
            index += len(ANY_DIRECTORIES_GLOB)
        elif glob.startswith(ANY_DEPTH_GLOB, index):
            parts.append(ANY_DEPTH)
            index += len(ANY_DEPTH_GLOB)
        elif glob.startswith(WITHIN_A_DIRECTORY_GLOB, index):
            parts.append(WITHIN_A_DIRECTORY)
            index += len(WITHIN_A_DIRECTORY_GLOB)
        elif glob.startswith(ONE_CHARACTER_GLOB, index):
            parts.append(ONE_CHARACTER)
            index += len(ONE_CHARACTER_GLOB)
        elif glob.startswith(CLASS_START, index) and (end := class_end(glob, index)) != NOT_FOUND:
            parts.append(character_class(glob[index + len(CLASS_START) : end]))
            index = end + len(CLASS_END)
        else:
            parts.append(re.escape(glob[index]))
            index += 1
    return "".join(parts)


def class_end(glob: str, start: int) -> int:
    """Where the class opened at `start` closes; NOT_FOUND when it never does, and the
    bracket is then a character like any other."""
    return glob.find(CLASS_END, start + len(CLASS_START))


def character_class(inside: str) -> str:
    """A glob's class as a regex's: `[!ab]` takes every character but a and b."""
    negated = inside.startswith(GLOB_CLASS_NEGATION)
    members = inside.removeprefix(GLOB_CLASS_NEGATION) if negated else inside
    escaped = "".join(
        ESCAPE + character if character in CLASS_SPECIAL_CHARACTERS else character
        for character in members
    )
    negation = REGEX_CLASS_NEGATION if negated else ""
    return CLASS_START + negation + escaped + CLASS_END


def directory_pattern(directory: Path) -> str:
    """The regular expression that matches the paths a directory holds, its name taken
    as it is written, wildcards and all."""
    return WHOLE_PATH_START + re.escape(directory.as_posix()) + PATH_SEPARATOR
