"""A path glob as the regular expression the store's filter matches a chunk's path with.

The store's `regexp_like` reads Rust's regex syntax; what is written here, the escapes
`re.escape` writes among it, reads the same in Python's (probed in LanceDB 0.27, and held by
a property test). A SQL `LIKE` would not do: its `%` crosses every `/` that a `*` stops at,
and a `%` or `_` in a path would match what it stands for in `LIKE`.

It takes a glob as `semantic_code_mcp.domain.globs` reads it: a name, or a path with a
leading `/`. Every character but the wildcards is itself; a class of characters is not
read, as Rust's regex reads one apart from a glob (`--`, `[:alpha:]`) and a class of a
glob never matches a `/`, which a regex's does.
"""

import re
from pathlib import Path

from semantic_code_mcp.domain.globs import PATH_SEPARATOR

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
    """The regular expression that matches the whole of every path a read glob names,
    from `base`, the directory of the project a search was asked from, when it is not the
    project's root."""
    is_path = glob.startswith(PATH_SEPARATOR)
    below_base = "" if base is None else re.escape(base.as_posix()) + PATH_SEPARATOR
    at_any_depth = "" if is_path else ANY_DIRECTORIES
    body = glob_body(glob.removeprefix(PATH_SEPARATOR))
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
