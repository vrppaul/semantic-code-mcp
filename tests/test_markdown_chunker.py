"""Tests for Markdown tree-sitter chunker."""

from pathlib import Path

from semantic_code_mcp.chunkers.markdown import MarkdownChunker
from semantic_code_mcp.models import ChunkType


class TestMarkdownChunker:
    """Tests for MarkdownChunker."""

    def test_single_heading_with_content(self):
        """Single heading with a paragraph produces one section chunk."""
        md = "# Installation\n\nRun `pip install foo` to get started.\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(md, file_path="README.md")

        assert len(chunks) == 1
        assert chunks[0].name == "Installation"
        assert chunks[0].chunk_type == ChunkType.section
        assert "# Installation" in chunks[0].content
        assert "pip install foo" in chunks[0].content

    def test_multiple_top_level_headings(self):
        """Two H1 headings produce two separate chunks."""
        md = "# First\n\nContent one.\n\n# Second\n\nContent two.\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(md, file_path="doc.md")

        assert len(chunks) == 2
        assert chunks[0].name == "First"
        assert chunks[0].chunk_type == ChunkType.section
        assert "Content one" in chunks[0].content
        assert chunks[1].name == "Second"
        assert chunks[1].chunk_type == ChunkType.section
        assert "Content two" in chunks[1].content

    def test_nested_headings_produce_separate_chunks(self):
        """H2 under H1 produces separate chunks (flattened)."""
        md = "# Top\n\nTop content.\n\n## Sub A\n\nSub A content.\n\n## Sub B\n\nSub B content.\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(md, file_path="doc.md")

        assert len(chunks) == 3
        assert chunks[0].name == "Top"
        assert chunks[0].chunk_type == ChunkType.section
        assert "Top content" in chunks[0].content
        # Top chunk should NOT contain sub-section content
        assert "Sub A content" not in chunks[0].content

        assert chunks[1].name == "Sub A"
        assert "Sub A content" in chunks[1].content

        assert chunks[2].name == "Sub B"
        assert "Sub B content" in chunks[2].content

    def test_deeply_nested_headings(self):
        """H1 > H2 > H3 all produce separate chunks."""
        md = "# H1\n\nH1 text.\n\n## H2\n\nH2 text.\n\n### H3\n\nH3 text.\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(md, file_path="doc.md")

        assert len(chunks) == 3
        names = [c.name for c in chunks]
        assert names == ["H1", "H2", "H3"]
        assert all(c.chunk_type == ChunkType.section for c in chunks)

    def test_preamble_before_first_heading(self):
        """Content before first heading becomes a module chunk."""
        md = "Some preamble text.\n\n# First Heading\n\nHeading content.\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(md, file_path="README.md")

        assert len(chunks) == 2
        assert chunks[0].name == "README"
        assert chunks[0].chunk_type == ChunkType.module
        assert "preamble text" in chunks[0].content

        assert chunks[1].name == "First Heading"
        assert chunks[1].chunk_type == ChunkType.section

    def test_no_headings_produces_module_chunk(self):
        """Document with no headings produces one module chunk."""
        md = "Just some text.\n\nAnother paragraph.\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(md, file_path="notes.md")

        assert len(chunks) == 1
        assert chunks[0].name == "notes"
        assert chunks[0].chunk_type == ChunkType.module
        assert "Just some text" in chunks[0].content

    def test_empty_file(self):
        """Empty file returns no chunks."""
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string("", file_path="empty.md")

        assert chunks == []

    def test_heading_without_content(self):
        """Heading with no content after it still produces a chunk."""
        md = "# Heading Only\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(md, file_path="doc.md")

        assert len(chunks) == 1
        assert chunks[0].name == "Heading Only"
        assert chunks[0].chunk_type == ChunkType.section

    def test_heading_with_code_block(self):
        """Fenced code block stays within its section chunk."""
        md = "# Examples\n\n```python\ndef hello():\n    print('hi')\n```\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(md, file_path="doc.md")

        assert len(chunks) == 1
        assert chunks[0].name == "Examples"
        assert "def hello():" in chunks[0].content
        assert "```python" in chunks[0].content

    def test_line_numbers_single_section(self):
        """Line numbers are 1-indexed and correct for a single section."""
        md = "# Title\n\nParagraph.\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(md, file_path="doc.md")

        assert chunks[0].line_start == 1
        # "# Title\n\nParagraph.\n" → lines 1-4 (trailing newline = empty line 4)
        assert chunks[0].line_end == 4

    def test_line_numbers_multiple_sections(self):
        """Line numbers are correct for multiple sections."""
        md = "# First\n\nContent one.\n\n# Second\n\nContent two.\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(md, file_path="doc.md")

        assert chunks[0].line_start == 1
        assert chunks[1].line_start == 5
        # "# Second\n\nContent two.\n" → lines 5-8
        assert chunks[1].line_end == 8

    def test_line_numbers_with_preamble(self):
        """Preamble chunk has correct line numbers."""
        md = "Preamble.\n\n# Heading\n\nContent.\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(md, file_path="doc.md")

        assert chunks[0].line_start == 1  # preamble
        assert chunks[1].line_start == 3  # heading

    def test_chunk_file_from_disk(self, tmp_path: Path):
        """chunk_file reads from disk and returns chunks."""
        file_path = tmp_path / "test.md"
        file_path.write_text("# Hello\n\nWorld.\n")

        chunker = MarkdownChunker()
        chunks = chunker.chunk_file(str(file_path))

        assert len(chunks) == 1
        assert chunks[0].name == "Hello"
        assert chunks[0].file_path == str(file_path)

    def test_extensions_attribute(self):
        """MarkdownChunker declares .md extension."""
        assert MarkdownChunker.extensions == (".md",)

    def test_h2_heading_level(self):
        """H2 headings work correctly as sections."""
        md = "## Setup\n\nSetup instructions.\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(md, file_path="doc.md")

        assert len(chunks) == 1
        assert chunks[0].name == "Setup"
        assert chunks[0].chunk_type == ChunkType.section

    def test_whitespace_only_file(self):
        """Whitespace-only file returns no chunks."""
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string("   \n\n  \n", file_path="blank.md")

        assert chunks == []

    def test_sibling_h2_sections(self):
        """Multiple H2 sections at same level produce separate chunks."""
        md = "## A\n\nA content.\n\n## B\n\nB content.\n\n## C\n\nC content.\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(md, file_path="doc.md")

        assert len(chunks) == 3
        assert [c.name for c in chunks] == ["A", "B", "C"]

    def test_content_boundaries_exclude_subsections(self):
        """A parent section's content doesn't bleed into child sections."""
        md = "# Parent\n\nParent text.\n\n## Child\n\nChild text.\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(md, file_path="doc.md")

        parent = chunks[0]
        child = chunks[1]
        assert "Parent text" in parent.content
        assert "Child text" not in parent.content
        assert "Child text" in child.content
        assert "Parent text" not in child.content

    def test_list_within_section(self):
        """Lists are included in their section's content."""
        md = "# Items\n\n- one\n- two\n- three\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(md, file_path="doc.md")

        assert len(chunks) == 1
        assert "- one" in chunks[0].content
        assert "- three" in chunks[0].content

    def test_setext_heading(self):
        """Setext-style headings (Title\\n=====) are recognized as sections."""
        md = "Title\n=====\n\nContent under setext heading.\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(md, file_path="doc.md")

        assert len(chunks) == 1
        assert chunks[0].name == "Title"
        assert chunks[0].chunk_type == ChunkType.section
        assert "Content under setext" in chunks[0].content

    def test_setext_h2_heading(self):
        """Setext H2 (Title\\n-----) is recognized as a section."""
        md = "Subtitle\n--------\n\nSub content.\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(md, file_path="doc.md")

        assert len(chunks) == 1
        assert chunks[0].name == "Subtitle"
        assert chunks[0].chunk_type == ChunkType.section

    def test_content_no_trailing_newline(self):
        """Content without trailing newline is fully captured."""
        md = "# Title\n\nLast line"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(md, file_path="doc.md")

        assert len(chunks) == 1
        assert "Last line" in chunks[0].content

    def test_heading_immediately_before_subheading(self):
        """Heading immediately followed by sub-heading (no blank line)."""
        md = "# Top\n## Sub\n\nSub content.\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(md, file_path="doc.md")

        # Top has no own content, Sub has content
        sub_chunks = [c for c in chunks if c.name == "Sub"]
        assert len(sub_chunks) == 1
        assert "Sub content" in sub_chunks[0].content

    def test_frontmatter_excluded(self):
        """YAML frontmatter is not included in chunks."""
        md = "---\ntitle: Hello\n---\n\n# Content\n\nBody text.\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(md, file_path="doc.md")

        # Should have the heading section, frontmatter should not appear
        names = [c.name for c in chunks]
        assert "Content" in names
        # No chunk should contain the frontmatter YAML
        for c in chunks:
            assert "title: Hello" not in c.content

    def test_exact_content_single_section(self):
        """Verify exact content capture (no off-by-one)."""
        md = "# Title\n\nLine one.\nLine two.\n"
        chunker = MarkdownChunker()
        chunks = chunker.chunk_string(md, file_path="doc.md")

        assert chunks[0].content == "# Title\n\nLine one.\nLine two."
