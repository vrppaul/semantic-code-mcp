"""A path glob as the regular expression the store's filter matches a chunk's path with.

The store's `regexp_like` reads Rust's regex syntax; what is written here, the escapes
`re.escape` writes among it, reads the same in Python's (probed in LanceDB 0.27, and held by
a property test). A SQL `LIKE` would not do: its `%` crosses every `/` that a `*` stops at,
and a `%` or `_` in a path would match what it stands for in `LIKE`.

A glob reads as in a `.gitignore`, from the directory a search was asked from: one without
a slash names a file or a directory at any depth below it, one with a slash a path from
it; `*` and `?` stay within a directory, `**` crosses any number of them, and a directory
named takes everything under it. Every other character is itself, a bracket too: a class
of characters is not read, as Rust's regex reads one apart from a glob (`--`, `[:alpha:]`)
and a class of a glob never matches a `/`, which a regex's does.
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
WILDCARDS = {
    ANY_DIRECTORIES_GLOB: ANY_DIRECTORIES,
    ANY_DEPTH_GLOB: ANY_DEPTH,
    WITHIN_A_DIRECTORY_GLOB: WITHIN_A_DIRECTORY,
    ONE_CHARACTER_GLOB: ONE_CHARACTER,
}
# A path the glob names, and everything under it when that path is a directory
WHOLE_PATH_START = "^"
AND_BELOW = "(?:/.*)?"
WHOLE_PATH_END = "$"


def glob_pattern(glob: str, base: Path | None) -> str:
    """The regular expression that matches the whole of every path the glob names, read
    from `base`, the directory of the project a search was asked from, when it is not the
    project's root."""
    rooted = glob.startswith(PATH_SEPARATOR) or PATH_SEPARATOR in glob.strip(PATH_SEPARATOR)
    below_base = "" if base is None else re.escape(base.as_posix()) + PATH_SEPARATOR
    at_any_depth = "" if rooted else ANY_DIRECTORIES
    body = glob_body(glob.strip(PATH_SEPARATOR))
    return WHOLE_PATH_START + below_base + at_any_depth + body + AND_BELOW + WHOLE_PATH_END


def glob_body(glob: str) -> str:
    """The glob's characters as a regular expression, its wildcards read, the rest literal."""
    parts: list[str] = []
    index = 0
    while index < len(glob):
        wildcard = next(
            (spelling for spelling in WILDCARDS if glob.startswith(spelling, index)), None
        )
        if wildcard is None:
            parts.append(re.escape(glob[index]))
            index += 1
        else:
            parts.append(WILDCARDS[wildcard])
            index += len(wildcard)
    return "".join(parts)
