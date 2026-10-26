"""The words a chunk is found by in the full-text index.

The index's tokenizer splits text at every character that is not a letter or a digit,
so `get_user_name` is found by `user`, but it keeps `getUserName` and `HTTPServerError`
whole, and a method's text does not name its class. So the text the index holds of a
chunk is its content, then its name and its parent, each followed by a line of the words
of every identifier in it whose case changes inside. The content's own identifiers stay
whole: splitting them moved no answer up on the harness, and cost a pass over every text.
"""

import re

from semantic_code_mcp.domain.models import Chunk

# A run of letters, digits and underscores: an identifier, or a plain word
IDENTIFIER = re.compile(r"\w+")
# What separates the words of a snake-case identifier
SNAKE_CASE_SEPARATOR = "_"
WORD_SEPARATOR = " "
LINE_SEPARATOR = "\n"
# A run of capitals ending in this lower-case letter is a plural acronym, as in `URLs`
PLURAL_ENDING = "s"


def search_text(chunk: Chunk) -> str:
    """What the full-text index holds of a chunk: its content, then its name and its
    parent, each followed by a line of words for every identifier in it whose case
    changes inside."""
    parts = [chunk.name]
    if chunk.parent is not None:
        parts.append(chunk.parent)
    lines = [line for part in parts for line in [part, *case_change_lines(part)]]
    return LINE_SEPARATOR.join([chunk.content, *lines])


def case_change_lines(text: str) -> list[str]:
    """A line of words for each identifier in the text whose case changes inside it."""
    return [
        WORD_SEPARATOR.join(words)
        for identifier in IDENTIFIER.findall(text)
        if len(words := identifier_words(identifier)) > len(snake_case_words(identifier))
    ]


def snake_case_words(identifier: str) -> list[str]:
    """The words of an identifier split at its underscores alone."""
    return [word for word in identifier.split(SNAKE_CASE_SEPARATOR) if word]


def identifier_words(identifier: str) -> list[str]:
    """The words an identifier is made of, in their case: `HTTPServer` is HTTP and Server,
    `URLs` is one word, `naïveBayes` naïve and Bayes, `utf8` utf and 8. One pass over its
    characters, so a long identifier costs no more than its length."""
    words: list[str] = []
    word_start = 0
    for index, character in enumerate(identifier):
        if not character.isalnum():
            words.append(identifier[word_start:index])
            word_start = index + 1
        elif starts_a_word(identifier, index):
            words.append(identifier[word_start:index])
            word_start = index
    words.append(identifier[word_start:])
    return [word for word in words if word]


def starts_a_word(identifier: str, index: int) -> bool:
    """Whether the character at `index` begins a word: a capital after a lower-case
    letter, a digit after a letter or a letter after a digit, or the capital that ends a
    run of them and begins a word of its own (`HTTPServer`), but not a plural's (`URLs`)."""
    if index == 0:
        return False
    character = identifier[index]
    previous = identifier[index - 1]
    if not previous.isalnum():
        return False
    if character.isdigit() != previous.isdigit():
        return True
    if character.isupper() and previous.islower():
        return True
    following = identifier[index + 1 : index + 2]
    return (
        character.isupper()
        and previous.isupper()
        and following.islower()
        and not is_plural_ending(identifier, index + 1)
    )


def is_plural_ending(identifier: str, index: int) -> bool:
    """Whether the letter at `index` is the `s` of an acronym's plural: an `s` that no
    lower-case letter follows."""
    following = identifier[index + 1 : index + 2]
    return identifier[index] == PLURAL_ENDING and not following.islower()
