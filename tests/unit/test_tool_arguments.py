"""Tests for how the tools read the arguments an agent gives them."""

import re

import pytest
from mcp.server.mcpserver.exceptions import ToolError

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
        pytest.param("src/.", "/src", id="a . closing a path"),
        pytest.param("docs/", "docs", id="a trailing slash"),
    ],
)
def test_a_glob_reads_from_the_directory_searched(glob: str | None, read: str | None):
    assert given_glob(glob) == read


@pytest.mark.parametrize(
    "glob",
    [
        pytest.param("..", id="the directory above"),
        pytest.param("../lib/*.py", id="a path from the directory above"),
        pytest.param("src/../lib", id="a path that climbs back"),
        pytest.param("[ab].py", id="a class of characters"),
        pytest.param("src/[!t]*.py", id="a class left out"),
        pytest.param("*.{py,rs}", id="braces of names"),
        pytest.param("{src,lib}/**", id="braces of directories"),
        pytest.param("!tests", id="a leading ! of a .gitignore"),
    ],
)
def test_a_glob_the_search_cannot_read_is_refused_by_name(glob: str):
    with pytest.raises(ToolError, match=re.escape(glob)):
        given_glob(glob)
