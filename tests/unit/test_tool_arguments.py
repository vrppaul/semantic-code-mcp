"""Tests for how the tools read the arguments an agent gives them."""

import re

import pytest
from hypothesis import given, strategies
from mcp.server.mcpserver.exceptions import ToolError

from semantic_code_mcp.domain.globs import (
    LONGEST_GLOB,
    NEGATION_INSTEAD,
    PARENT_DIRECTORY,
    PARENT_DIRECTORY_INSTEAD,
    REFUSED_CHARACTERS,
    TOO_LONG,
    parts_of,
    read_glob,
)
from semantic_code_mcp.server.tools import given_glob


@pytest.mark.parametrize(
    ("glob", "read"),
    [
        pytest.param(None, None, id="none given"),
        pytest.param("", None, id="an empty string, as agents fill optional strings"),
        pytest.param("  ", None, id="blanks"),
        pytest.param("/", None, id="a slash alone"),
        pytest.param(".", None, id="the directory searched"),
        pytest.param("./", None, id="the directory searched, with a slash"),
        pytest.param("./.", None, id="the directory searched, twice"),
        pytest.param("*.py", "*.py", id="a name"),
        pytest.param("  *.py  ", "*.py", id="a name, blanks around it"),
        pytest.param("src/*.py", "/src/*.py", id="a path from the directory searched"),
        pytest.param("./src", "/src", id="a leading ./, the directory searched"),
        pytest.param("a/./b.py", "/a/b.py", id="a . inside a path"),
        pytest.param("x//y.py", "/x/y.py", id="two slashes"),
        pytest.param("src/.", "src", id="a . closing it, as a trailing slash"),
        pytest.param("docs/", "docs", id="a trailing slash, a directory at any depth"),
        pytest.param("src /", "src", id="blanks before a slash"),
        pytest.param(" a / b ", "/a/b", id="blanks around each part"),
        pytest.param("/a/b", "/a/b", id="a glob read already"),
    ],
)
def test_a_glob_reads_from_the_directory_searched(glob: str | None, read: str | None):
    assert given_glob(glob) == read


@pytest.mark.parametrize(
    ("glob", "reason"),
    [
        pytest.param("..", PARENT_DIRECTORY_INSTEAD, id="the directory above"),
        pytest.param("../lib/*.py", PARENT_DIRECTORY_INSTEAD, id="a path from the one above"),
        pytest.param("src/ .. /lib", PARENT_DIRECTORY_INSTEAD, id="a path that climbs back"),
        pytest.param("[ab].py", REFUSED_CHARACTERS["["], id="a class of characters"),
        pytest.param("src/[!t]*.py", REFUSED_CHARACTERS["["], id="a class left out"),
        pytest.param("*.{py,rs}", REFUSED_CHARACTERS["{"], id="braces of names"),
        pytest.param("{{slug}}/**", REFUSED_CHARACTERS["{"], id="braces taken literally"),
        pytest.param("\\*.py", REFUSED_CHARACTERS["\\"], id="an escape of a .gitignore"),
        pytest.param(" !tests", NEGATION_INSTEAD, id="a leading ! of a .gitignore"),
    ],
)
def test_a_glob_the_search_cannot_read_is_refused_with_the_reason(glob: str, reason: str):
    with pytest.raises(ToolError, match=re.escape(f"{reason}: {glob}")):
        given_glob(glob)


@pytest.mark.parametrize(
    "glob",
    [
        pytest.param("?" * (LONGEST_GLOB + 1), id="as written"),
        pytest.param("a/" + "?" * (LONGEST_GLOB - len("a/")), id="once read, with its leading /"),
        pytest.param("[" + "?" * LONGEST_GLOB, id="refused for another reason too"),
    ],
)
def test_a_glob_too_long_for_the_store_s_regex_is_refused_without_its_echo(glob: str):
    with pytest.raises(ToolError, match=f"^{re.escape(TOO_LONG)}$"):
        given_glob(glob)


# What a glob an agent writes is made of, but the characters a search refuses outright,
# and no step out of the directory searched
WRITTEN_GLOBS = strategies.text(alphabet="ab.*?/ \t", max_size=40).filter(
    lambda glob: PARENT_DIRECTORY not in parts_of(glob)
)


@given(glob=WRITTEN_GLOBS)
def test_a_glob_read_reads_the_same_again(glob: str):
    assert read_glob(read_glob(glob)) == read_glob(glob)
