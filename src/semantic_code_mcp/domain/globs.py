"""What a search's path glob may hold, and the glob as a search reads it.

A glob reads as in a `.gitignore`, from the directory a search was asked from: one with a
slash before a part it names is a path from there, one without (a trailing slash aside) a
name at any depth below it; `*` and `?` stay within a directory, `**` crosses any number
of them, and a directory named takes everything under it. Reading gives each glob one
spelling, which reads the same again: a name, or a path with a leading `/`, its parts
without blanks around them. It refuses what a search would otherwise answer with nothing
and no word why. A `SearchScope` holds its globs read, and the store turns them into its
own filter.

These are not the grammar of the ignore patterns a user sets, which discovery reads as
`fnmatch` does.
"""

PATH_SEPARATOR = "/"
CURRENT_DIRECTORY = "."
PARENT_DIRECTORY = ".."
# A part a path spells the directory it is in with, or that two slashes leave
NAMELESS_PARTS = frozenset({"", CURRENT_DIRECTORY})
# The longest glob a search reads, as written and once read: thousands of wildcards
# overflow the store's regex. Its refusal does not repeat the glob back.
LONGEST_GLOB = 1000
TOO_LONG = f"A glob of more than {LONGEST_GLOB} characters is not read"
# What a glob of a .gitignore or a shell may start with or hold that a search does not
# read, and what an agent can do instead
NEGATION = "!"
NEGATION_INSTEAD = "A leading ! is not read; name what to leave out in exclude_glob"
REFUSED_CHARACTERS = {
    "[": "A class of characters is not read; ? stands for one character",
    "{": (
        "Braces are not read; search once for each name or narrow by languages, and let ?"
        " stand for a brace a path holds"
    ),
    "\\": (
        "Escapes are not read, and / alone separates directories; ? stands for the character"
        " an escape would keep"
    ),
}
PARENT_DIRECTORY_INSTEAD = (
    "A glob reads from the directory searched and cannot step out of it; name a directory"
    " above it in project_path instead"
)


class UnreadableGlobError(ValueError):
    """A glob a search would read otherwise than its writer meant, refused with the reason."""


def read_glob(glob: str | None) -> str | None:
    """The glob as a search reads it, in its one spelling; None for one that names nothing
    past the directory searched, as the empty string an agent fills an optional argument
    with. Raises UnreadableGlobError for a glob that steps out of the directory, or holds
    what a search does not read."""
    if glob is None:
        return None
    if len(glob) > LONGEST_GLOB:
        raise UnreadableGlobError(TOO_LONG)
    refusal = refusal_of(glob)
    if refusal is not None:
        message = f"{refusal}: {glob}"
        raise UnreadableGlobError(message)
    read = spelling_of(glob)
    if read is not None and len(read) > LONGEST_GLOB:
        raise UnreadableGlobError(TOO_LONG)
    return read


def spelling_of(glob: str) -> str | None:
    """The one spelling of a glob a search can read, or None for one naming no part."""
    parts = parts_of(glob)
    named = [part for part in parts if part not in NAMELESS_PARTS]
    if not named:
        return None
    # A slash makes a path when a named part follows it, as it does a leading one here
    is_path = any(part not in NAMELESS_PARTS for part in parts[1:])
    joined = PATH_SEPARATOR.join(named)
    return PATH_SEPARATOR + joined if is_path else joined


def refusal_of(glob: str) -> str | None:
    """Why a search cannot read a glob as its writer meant, its length aside; None when it
    can."""
    if glob.strip().startswith(NEGATION):
        return NEGATION_INSTEAD
    if PARENT_DIRECTORY in parts_of(glob):
        return PARENT_DIRECTORY_INSTEAD
    return next(
        (reason for character, reason in REFUSED_CHARACTERS.items() if character in glob), None
    )


def parts_of(glob: str) -> list[str]:
    """The parts of a glob between its slashes, without blanks around them."""
    return [part.strip() for part in glob.split(PATH_SEPARATOR)]
