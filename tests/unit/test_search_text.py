"""Tests for the words a chunk is found by in the full-text index."""

from pathlib import Path

import pytest

from semantic_code_mcp.domain.models import Chunk, ChunkKind, Language
from semantic_code_mcp.storage.search_text import identifier_words, search_text


@pytest.mark.parametrize(
    ("identifier", "words"),
    [
        pytest.param("getUserName", ["get", "User", "Name"], id="camel case"),
        pytest.param("HTTPServerError", ["HTTP", "Server", "Error"], id="an acronym first"),
        pytest.param("XMLHttpRequest", ["XML", "Http", "Request"], id="acronyms of two cases"),
        pytest.param("utf8Decode", ["utf", "8", "Decode"], id="a number inside"),
        pytest.param("fetchURLsNow", ["fetch", "URLs", "Now"], id="an acronym's plural"),
        pytest.param("naïveBayes", ["naïve", "Bayes"], id="a letter outside ASCII"),
        pytest.param("ÉcoleNormale", ["École", "Normale"], id="a capital outside ASCII"),
        pytest.param("read_file", ["read", "file"], id="snake case"),
    ],
)
def test_an_identifier_comes_apart_into_its_words(identifier: str, words: list[str]):
    assert identifier_words(identifier) == words


def test_a_chunk_is_found_by_its_content_its_name_and_its_parent_with_their_words():
    """The content stays as it is: splitting its identifiers moved no answer up."""
    chunk = Chunk(
        chunk_id="stack-push",
        path=Path("src/stack.rs"),
        language=Language.rust,
        kind=ChunkKind.method,
        name="pushItem",
        parent="BoundedStack",
        line_start=1,
        line_end=1,
        content="fn pushItem(&mut self) { self.itemCount += 1; let snake_case; }",
    )

    assert search_text(chunk).split("\n") == [
        chunk.content,
        "pushItem",
        "push Item",
        "BoundedStack",
        "Bounded Stack",
    ]


def test_a_snake_case_name_adds_no_words_as_the_tokenizer_splits_it_already():
    chunk = Chunk(
        chunk_id="read-file",
        path=Path("io.py"),
        language=Language.python,
        kind=ChunkKind.function,
        name="read_file",
        parent=None,
        line_start=1,
        line_end=1,
        content="def read_file(path): ...",
    )

    assert search_text(chunk).split("\n") == [chunk.content, "read_file"]
