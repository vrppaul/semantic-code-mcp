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
    ],
)
def test_an_identifier_in_camel_case_comes_apart_into_its_words(identifier: str, words: list[str]):
    assert identifier_words(identifier) == words


def test_a_chunk_is_found_by_its_content_its_camel_case_words_its_name_and_its_parent():
    chunk = Chunk(
        chunk_id="stack-push",
        path=Path("src/stack.rs"),
        language=Language.rust,
        kind=ChunkKind.method,
        name="push",
        parent="BoundedStack",
        line_start=1,
        line_end=1,
        content="fn push(&mut self) { self.itemCount += 1; let host: HTTPServer; let snake_case; }",
    )

    assert search_text(chunk).split("\n") == [
        chunk.content,
        "item Count",
        "HTTP Server",
        "push",
        "BoundedStack",
        "Bounded Stack",
    ]
