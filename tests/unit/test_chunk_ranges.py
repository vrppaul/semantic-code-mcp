"""Tests that every chunker's chunks cover exactly their lines of the file: an agent reads a
result's range to quote it, so the range must hold the chunk's content and nothing else.

Every chunker builds its chunks through `ChunkCollector`, which takes the content from the
range, so this holds the collector's arithmetic and that no file, however odd, breaks it.
Where a chunk starts and ends (a decorator, a doc comment, a heading) the example tests of
each chunker hold."""

from pathlib import Path

import pytest
from hypothesis import given, strategies

from semantic_code_mcp.chunkers.base import BaseTreeSitterChunker
from semantic_code_mcp.chunkers.markdown import MarkdownChunker
from semantic_code_mcp.chunkers.python import PythonChunker
from semantic_code_mcp.chunkers.rust import RustChunker
from semantic_code_mcp.domain.models import Language

LINE_BREAK = "\n"
# What a file's pieces are joined by, a Windows line break among them
LINE_BREAKS = ("\n", "\r\n", "\n\n\n")
# Pieces of each language a file is put together from, among them the shapes chunkers have
# got wrong: a comment between a decorator and its definition, nested and generic owners,
# modules of tests, empty headings, code fences, broken syntax, blank lines
PIECES = {
    Language.python: (
        "def function():\n    return 1",
        "@decorator\n# a comment between\ndef decorated():\n    pass",
        'class Owner:\n    """An owner."""\n\n    def method(self):\n        pass',
        "class Outer:\n    class Inner:\n        def deep(self):\n            pass",
        "if True:\n    def conditional():\n        pass",
        "async def waiting():\n    await other()",
        '"""A module docstring."""',
        "# a comment",
        "value = 1",
        "    stray_indent = True",
        "def broken(:",
        "   ",
    ),
    Language.rust: (
        "fn function() {}",
        "pub struct Stack<T> {\n    items: Vec<T>,\n}",
        "impl<T: Clone> Stack<T> {\n    fn push(&mut self, item: T) {}\n}",
        "impl fmt::Display for crate::Path {\n    fn fmt(&self) {}\n}",
        "mod inner {\n    fn nested() {}\n}",
        "#[cfg(test)]\nmod tests {\n    #[test]\n    fn checks() {}\n}",
        "trait Shape {\n    fn area(&self) -> f64;\n}",
        "enum Choice { One, Two }",
        "/// A doc comment",
        "// a comment",
        "}",
        "   ",
    ),
    Language.markdown: (
        "# Title",
        "## Section",
        "### Deeper",
        "#",
        "Setext title\n=====",
        "Setext section\n-----",
        "Some text.",
        "- a list item",
        "```python\n# not a heading\n```",
        "```",
        "---\ntitle: front\n---",
        "> a quote",
        "   ",
    ),
}
# Arbitrary text mixed in, a few characters at a time
TEXT_MAX_SIZE = 5
PIECES_PER_FILE = 12
CHUNKERS = [
    pytest.param(PythonChunker(), Path("module.py"), id="python"),
    pytest.param(RustChunker(), Path("module.rs"), id="rust"),
    pytest.param(MarkdownChunker(), Path("document.md"), id="markdown"),
]


def source_of(language: Language) -> strategies.SearchStrategy[str]:
    """A file of the language: its pieces, line breaks and arbitrary text in any order."""
    part = strategies.one_of(
        strategies.sampled_from((*PIECES[language], *LINE_BREAKS)),
        strategies.text(max_size=TEXT_MAX_SIZE),
    )
    return strategies.lists(part, max_size=PIECES_PER_FILE).map("".join)


def lines_of(source: str) -> list[str]:
    """A file's lines as an editor numbers them: a line break ending the file starts none."""
    return source.removesuffix(LINE_BREAK).split(LINE_BREAK)


@pytest.mark.parametrize(("chunker", "path"), CHUNKERS)
@given(data=strategies.data())
def test_a_chunk_s_line_range_holds_exactly_its_content(
    chunker: BaseTreeSitterChunker, path: Path, data: strategies.DataObject
):
    source = data.draw(source_of(chunker.language))
    lines = lines_of(source)

    chunks = chunker.chunk_string(source, path)

    assert [chunk.content for chunk in chunks] == [
        LINE_BREAK.join(lines[chunk.line_start - 1 : chunk.line_end]) for chunk in chunks
    ]
