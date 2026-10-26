"""Tests for Markdown tree-sitter chunker."""

from pathlib import Path

from semantic_code_mcp.chunkers.markdown import MarkdownChunker
from semantic_code_mcp.domain.models import ChunkKind
from tests.helpers import ChunkOwner, owners_of


class TestMarkdownChunker:
    """Tests for MarkdownChunker."""

    def test_single_heading_with_content(self):
        """Single heading with a paragraph produces one section chunk."""
        markdown_text = "# Installation\n\nRun `pip install foo` to get started.\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(markdown_text, Path("README.md"))

        assert len(chunks) == 1
        assert chunks[0].name == "Installation"
        assert chunks[0].kind == ChunkKind.section
        assert "# Installation" in chunks[0].content
        assert "pip install foo" in chunks[0].content

    def test_multiple_top_level_headings(self):
        """Two H1 headings produce two separate chunks."""
        markdown_text = "# First\n\nContent one.\n\n# Second\n\nContent two.\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(markdown_text, Path("doc.md"))

        assert len(chunks) == 2
        assert chunks[0].name == "First"
        assert chunks[0].kind == ChunkKind.section
        assert "Content one" in chunks[0].content
        assert chunks[1].name == "Second"
        assert chunks[1].kind == ChunkKind.section
        assert "Content two" in chunks[1].content

    def test_nested_headings_produce_separate_chunks(self):
        """H2 under H1 produces separate chunks (flattened)."""
        markdown_text = (
            "# Top\n\nTop content.\n\n## Sub A\n\nSub A content.\n\n## Sub B\n\nSub B content.\n"
        )
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(markdown_text, Path("doc.md"))

        assert len(chunks) == 3
        assert chunks[0].name == "Top"
        assert chunks[0].kind == ChunkKind.section
        assert "Top content" in chunks[0].content
        # Top chunk should NOT contain sub-section content
        assert "Sub A content" not in chunks[0].content

        assert chunks[1].name == "Sub A"
        assert "Sub A content" in chunks[1].content

        assert chunks[2].name == "Sub B"
        assert "Sub B content" in chunks[2].content

    def test_deeply_nested_headings(self):
        """H1 > H2 > H3 all produce separate chunks."""
        markdown_text = "# H1\n\nH1 text.\n\n## H2\n\nH2 text.\n\n### H3\n\nH3 text.\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(markdown_text, Path("doc.md"))

        assert len(chunks) == 3
        names = [chunk.name for chunk in chunks]
        assert names == ["H1", "H2", "H3"]
        assert all(chunk.kind == ChunkKind.section for chunk in chunks)

    def test_preamble_before_first_heading(self):
        """Content before first heading becomes a module chunk."""
        markdown_text = "Some preamble text.\n\n# First Heading\n\nHeading content.\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(markdown_text, Path("README.md"))

        assert len(chunks) == 2
        assert chunks[0].name == "README"
        assert chunks[0].kind == ChunkKind.module
        assert "preamble text" in chunks[0].content

        assert chunks[1].name == "First Heading"
        assert chunks[1].kind == ChunkKind.section

    def test_no_headings_produces_module_chunk(self):
        """Document with no headings produces one module chunk."""
        markdown_text = "Just some text.\n\nAnother paragraph.\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(markdown_text, Path("notes.md"))

        assert len(chunks) == 1
        assert chunks[0].name == "notes"
        assert chunks[0].kind == ChunkKind.module
        assert "Just some text" in chunks[0].content

    def test_empty_file(self):
        """Empty file returns no chunks."""
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string("", Path("empty.md"))

        assert chunks == []

    def test_heading_without_content(self):
        """Heading with no content after it still produces a chunk."""
        markdown_text = "# Heading Only\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(markdown_text, Path("doc.md"))

        assert len(chunks) == 1
        assert chunks[0].name == "Heading Only"
        assert chunks[0].kind == ChunkKind.section

    def test_heading_with_code_block(self):
        """Fenced code block stays within its section chunk."""
        markdown_text = "# Examples\n\n```python\ndef hello():\n    print('hi')\n```\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(markdown_text, Path("doc.md"))

        assert len(chunks) == 1
        assert chunks[0].name == "Examples"
        assert "def hello():" in chunks[0].content
        assert "```python" in chunks[0].content

    def test_line_numbers_single_section(self):
        """Line numbers are 1-indexed and correct for a single section."""
        markdown_text = "# Title\n\nParagraph.\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(markdown_text, Path("doc.md"))

        assert (chunks[0].line_start, chunks[0].line_end) == (1, 3)

    def test_line_numbers_multiple_sections(self):
        """Line numbers are correct for multiple sections."""
        markdown_text = "# First\n\nContent one.\n\n# Second\n\nContent two.\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(markdown_text, Path("doc.md"))

        assert (chunks[0].line_start, chunks[0].line_end) == (1, 3)
        assert (chunks[1].line_start, chunks[1].line_end) == (5, 7)

    def test_line_numbers_with_preamble(self):
        """Preamble chunk has correct line numbers."""
        markdown_text = "Preamble.\n\n# Heading\n\nContent.\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(markdown_text, Path("doc.md"))

        assert chunks[0].line_start == 1  # preamble
        assert chunks[1].line_start == 3  # heading

    def test_chunk_file_from_disk(self, tmp_path: Path):
        """chunk_file reads from disk and returns chunks."""
        file_path = tmp_path / "test.md"
        file_path.write_text("# Hello\n\nWorld.\n")

        chunker = MarkdownChunker()
        chunks = chunker.chunk_file(file_path.parent, Path(file_path.name))

        assert len(chunks) == 1
        assert chunks[0].name == "Hello"
        assert chunks[0].path == Path("test.md")

    def test_extensions_attribute(self):
        """MarkdownChunker declares .markdown_text extension."""
        assert MarkdownChunker.extensions == (".md",)

    def test_h2_heading_level(self):
        """H2 headings work correctly as sections."""
        markdown_text = "## Setup\n\nSetup instructions.\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(markdown_text, Path("doc.md"))

        assert len(chunks) == 1
        assert chunks[0].name == "Setup"
        assert chunks[0].kind == ChunkKind.section

    def test_whitespace_only_file(self):
        """Whitespace-only file returns no chunks."""
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string("   \n\n  \n", Path("blank.md"))

        assert chunks == []

    def test_sibling_h2_sections(self):
        """Multiple H2 sections at same level produce separate chunks."""
        markdown_text = "## A\n\nA content.\n\n## B\n\nB content.\n\n## C\n\nC content.\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(markdown_text, Path("doc.md"))

        assert len(chunks) == 3
        assert [chunk.name for chunk in chunks] == ["A", "B", "C"]

    def test_content_boundaries_exclude_subsections(self):
        """A parent section's content doesn't bleed into child sections."""
        markdown_text = "# Parent\n\nParent text.\n\n## Child\n\nChild text.\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(markdown_text, Path("doc.md"))

        parent = chunks[0]
        child = chunks[1]
        assert "Parent text" in parent.content
        assert "Child text" not in parent.content
        assert "Child text" in child.content
        assert "Parent text" not in child.content

    def test_list_within_section(self):
        """Lists are included in their section's content."""
        markdown_text = "# Items\n\n- one\n- two\n- three\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(markdown_text, Path("doc.md"))

        assert len(chunks) == 1
        assert "- one" in chunks[0].content
        assert "- three" in chunks[0].content

    def test_setext_heading(self):
        """Setext-style headings (Title\\n=====) are recognized as sections."""
        markdown_text = "Title\n=====\n\nContent under setext heading.\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(markdown_text, Path("doc.md"))

        assert len(chunks) == 1
        assert chunks[0].name == "Title"
        assert chunks[0].kind == ChunkKind.section
        assert "Content under setext" in chunks[0].content

    def test_setext_h2_heading(self):
        """Setext H2 (Title\\n-----) is recognized as a section."""
        markdown_text = "Subtitle\n--------\n\nSub content.\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(markdown_text, Path("doc.md"))

        assert len(chunks) == 1
        assert chunks[0].name == "Subtitle"
        assert chunks[0].kind == ChunkKind.section

    def test_content_no_trailing_newline(self):
        """Content without trailing newline is fully captured."""
        markdown_text = "# Title\n\nLast line"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(markdown_text, Path("doc.md"))

        assert len(chunks) == 1
        assert "Last line" in chunks[0].content

    def test_heading_immediately_before_subheading(self):
        """Heading immediately followed by sub-heading (no blank line)."""
        markdown_text = "# Top\n## Sub\n\nSub content.\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(markdown_text, Path("doc.md"))

        # Top has no own content, Sub has content
        sub_chunks = [chunk for chunk in chunks if chunk.name == "Sub"]
        assert len(sub_chunks) == 1
        assert "Sub content" in sub_chunks[0].content

    def test_frontmatter_excluded(self):
        """YAML frontmatter is not included in chunks."""
        markdown_text = "---\ntitle: Hello\n---\n\n# Content\n\nBody text.\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(markdown_text, Path("doc.md"))

        # Should have the heading section, frontmatter should not appear
        names = [chunk.name for chunk in chunks]
        assert "Content" in names
        # No chunk should contain the frontmatter YAML
        for chunk in chunks:
            assert "title: Hello" not in chunk.content

    def test_exact_content_single_section(self):
        """Verify exact content capture (no off-by-one)."""
        markdown_text = "# Title\n\nLine one.\nLine two.\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(markdown_text, Path("doc.md"))

        assert chunks[0].content == "# Title\n\nLine one.\nLine two."


def test_an_empty_heading_adds_nothing_to_the_names_of_the_sections_below_it():
    markdown_text = """# Guide

Intro.

##

Text.

### Deep

More.
"""

    chunks = MarkdownChunker().chunk_string(markdown_text, Path("guide.md"))

    assert owners_of(chunks) == [
        ChunkOwner(name="Guide", parent=None),
        ChunkOwner(name="guide", parent="Guide"),
        ChunkOwner(name="Deep", parent="Guide"),
    ]


def test_a_section_names_the_sections_it_is_under():
    markdown_text = """# Guide

Intro.

## Install

Steps.

### Linux

More.

## Use

Text.
"""

    chunks = MarkdownChunker().chunk_string(markdown_text, Path("guide.md"))

    assert owners_of(chunks) == [
        ChunkOwner(name="Guide", parent=None),
        ChunkOwner(name="Install", parent="Guide"),
        ChunkOwner(name="Linux", parent="Guide > Install"),
        ChunkOwner(name="Use", parent="Guide"),
    ]
