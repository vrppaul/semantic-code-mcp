"""What a search's path glob may hold, and the glob as a search reads it.

A glob reads as in a `.gitignore`, from the directory a search was asked from: one with a
slash before its last character names a path from there, one without a name at any depth
below it; `*` and `?` stay within a directory, `**` crosses any number of them, and a
directory named takes everything under it. Reading gives each glob one spelling, a path
from that directory with a leading `/`, and refuses what a search would otherwise answer
with nothing and no word why. The store turns the glob it is given into its own filter.
"""

PATH_SEPARATOR = "/"
CURRENT_DIRECTORY = "."
PARENT_DIRECTORY = ".."
# A component a path spells the directory it is in with, or that two slashes leave
NAMELESS_COMPONENTS = frozenset({"", CURRENT_DIRECTORY})
# What a glob of a .gitignore or a shell may start with or hold that a search does not
# read, and what an agent can do instead
NEGATION = "!"
NEGATION_INSTEAD = "A leading ! is not read; name what to leave out in exclude_glob"
REFUSED_CHARACTERS = {
    "[": "A class of characters is not read; ? stands for one character",
    "{": "Braces are not read; search once for each name, or narrow by languages",
}
PARENT_DIRECTORY_INSTEAD = "A glob reads from the directory searched and cannot step out of it"


class UnreadableGlobError(ValueError):
    """A glob a search would read otherwise than its writer meant, refused with the reason."""


def read_glob(glob: str | None) -> str | None:
    """The glob as a search reads it: blanks around it, `.` components and doubled slashes
    dropped, a path from the directory searched spelled with a leading `/`; None for one
    that names nothing past that directory, as the empty string an agent fills an optional
    argument with. Raises UnreadableGlobError for a glob that steps out of the directory,
    or holds what a search does not read."""
    if glob is None:
        return None
    stripped = glob.strip()
    refusal = refusal_of(stripped)
    if refusal is not None:
        message = f"{refusal}: {glob}"
        raise UnreadableGlobError(message)
    components = stripped.split(PATH_SEPARATOR)
    named = [component for component in components if component not in NAMELESS_COMPONENTS]
    if not named:
        return None
    joined = PATH_SEPARATOR.join(named)
    is_path = PATH_SEPARATOR in stripped.rstrip(PATH_SEPARATOR)
    return PATH_SEPARATOR + joined if is_path else joined


def refusal_of(glob: str) -> str | None:
    """Why a search cannot read a glob as its writer meant; None when it can."""
    if glob.startswith(NEGATION):
        return NEGATION_INSTEAD
    if PARENT_DIRECTORY in glob.split(PATH_SEPARATOR):
        return PARENT_DIRECTORY_INSTEAD
    return next(
        (reason for character, reason in REFUSED_CHARACTERS.items() if character in glob), None
    )
