"""Tests for what a search's scope becomes in the store's filter."""

import re
from pathlib import Path

import pytest

from semantic_code_mcp.domain.models import ChunkKind, Language, SearchScope
from semantic_code_mcp.storage.globs import glob_pattern
from semantic_code_mcp.storage.lancedb import scope_filter


@pytest.mark.parametrize(
    ("glob", "path", "matches"),
    [
        pytest.param("*.py", "top.py", True, id="a name pattern, at the root"),
        pytest.param("*.py", "src/deep/module.py", True, id="a name pattern, at any depth"),
        pytest.param("*.py", "src/module.rs", False, id="a name pattern, another name"),
        pytest.param("src/*.py", "src/module.py", True, id="a star, within its directory"),
        pytest.param("src/*.py", "src/deep/module.py", False, id="a star, not across a slash"),
        pytest.param("src/**/*.py", "src/module.py", True, id="two stars, no directory"),
        pytest.param("src/**/*.py", "src/a/b/module.py", True, id="two stars, any depth"),
        pytest.param("src/**/*.py", "lib/src/module.py", False, id="a path, from the root"),
        pytest.param("docs", "docs/guide/install.md", True, id="a directory, what it holds"),
        pytest.param("/docs", "site/docs/index.md", False, id="a leading slash, the root's"),
        pytest.param("docs/", "docs/index.md", True, id="a trailing slash, a directory"),
        pytest.param("file?.py", "file1.py", True, id="a question mark, one character"),
        pytest.param("file?.py", "file10.py", False, id="a question mark, not two"),
        pytest.param("a?b", "a/b", False, id="a question mark, never a slash"),
        pytest.param("[ab].rs", "[ab].rs", True, id="a bracket, itself"),
        pytest.param("[ab].rs", "b.rs", False, id="a bracket, no class"),
        pytest.param("lib(1)/*.py", "lib(1)/module.py", True, id="what a regex would read"),
        pytest.param("100%_done.md", "100%_done.md", True, id="what SQL's LIKE would read"),
    ],
)
def test_a_glob_takes_the_paths_it_names(glob: str, path: str, matches: bool):
    assert bool(re.fullmatch(glob_pattern(glob, base=None), path)) is matches


@pytest.mark.parametrize(
    ("glob", "path", "matches"),
    [
        pytest.param("/config.py", "package/config.py", True, id="rooted, in the directory"),
        pytest.param("/config.py", "config.py", False, id="rooted, not the repository's"),
        pytest.param("src/*.py", "package/src/module.py", True, id="a path from the directory"),
        pytest.param("*.py", "package/deep/module.py", True, id="a name, at any depth"),
        pytest.param("package", "package/config.py", False, id="a name, not the directory's"),
        pytest.param("package", "package/package/x.py", True, id="a name, below the directory"),
    ],
)
def test_a_glob_reads_from_the_directory_a_search_was_asked_from(
    glob: str, path: str, matches: bool
):
    """An agent that names a directory of a repository names paths from that directory."""
    assert bool(re.fullmatch(glob_pattern(glob, base=Path("package")), path)) is matches


def test_a_scope_that_takes_every_chunk_needs_no_filter():
    scope = SearchScope(kinds=[], languages=[], path_glob=None, exclude_glob=None, directory=None)

    assert scope_filter(scope) is None


def test_a_scope_becomes_one_condition_for_each_thing_it_names():
    scope = SearchScope(
        kinds=[ChunkKind.klass, ChunkKind.method],
        languages=[Language.rust],
        path_glob="src/*.rs",
        exclude_glob="it's.rs",
        directory=Path("crates/core.d"),
    )

    assert scope_filter(scope) == (
        "kind IN ('class', 'method') AND language IN ('rust')"
        r" AND regexp_like(path, '^crates/core\.d/src/[^/]*\.rs(?:/.*)?$')"
        r" AND NOT regexp_like(path, '^crates/core\.d/(?:.*/)?it''s\.rs(?:/.*)?$')"
        " AND starts_with(path, 'crates/core.d/')"
    )
