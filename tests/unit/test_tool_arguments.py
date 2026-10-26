"""Tests for how the tools read the arguments an agent gives them."""

import pytest

from semantic_code_mcp.server.tools import given_glob


@pytest.mark.parametrize(
    ("glob", "read"),
    [
        pytest.param(None, None, id="none given"),
        pytest.param("", None, id="an empty string, as agents fill optional strings"),
        pytest.param("  ", None, id="blanks"),
        pytest.param("/", None, id="a slash alone"),
        pytest.param("src/*.py", "src/*.py", id="a glob"),
    ],
)
def test_a_glob_that_names_no_path_narrows_nothing(glob: str | None, read: str | None):
    assert given_glob(glob) == read
