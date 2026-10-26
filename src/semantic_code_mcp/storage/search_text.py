"""The words a chunk is found by in the full-text index.

The index's tokenizer splits text at every character that is not a letter or a digit,
so `get_user_name` is found by `user`, but it keeps `getUserName` and `HTTPServerError`
whole, and a query for "user name" or "server error" missed them. So the text the index
holds of a chunk is its content, its name and its parent, which a method's own text may
not hold, each followed by the words of every identifier in it whose case changes inside.
"""

import re

from semantic_code_mcp.domain.models import Chunk

# An identifier whose case changes inside it: a lower-case letter or a digit before a
# capital, or two capitals before a lower-case letter, as in `getUserName` and `HTTPServer`
CAMEL_CASE_IDENTIFIER = re.compile(r"\w*(?:[a-z0-9][A-Z]|[A-Z]{2}[a-z])\w*")
# A word of such an identifier: an acronym, a word with or without its capital, a number
CAMEL_CASE_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
WORD_SEPARATOR = " "
LINE_SEPARATOR = "\n"


def search_text(chunk: Chunk) -> str:
    """What the full-text index holds of a chunk: its content, its name and its parent,
    each followed by a line of words for every identifier in it written in camel case."""
    parts = [chunk.content, chunk.name]
    if chunk.parent is not None:
        parts.append(chunk.parent)
    lines = [line for part in parts for line in [part, *camel_case_lines(part)]]
    return LINE_SEPARATOR.join(lines)


def camel_case_lines(text: str) -> list[str]:
    """A line of words for each identifier in the text whose case changes inside it."""
    return [
        WORD_SEPARATOR.join(identifier_words(identifier))
        for identifier in CAMEL_CASE_IDENTIFIER.findall(text)
    ]


def identifier_words(identifier: str) -> list[str]:
    """The words an identifier is made of, in their case: `HTTPServer` is HTTP and Server."""
    return CAMEL_CASE_WORD.findall(identifier)
