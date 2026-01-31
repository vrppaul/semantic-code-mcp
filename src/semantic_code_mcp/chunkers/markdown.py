"""Markdown tree-sitter chunker — heading-based section extraction."""

from enum import StrEnum, auto
from pathlib import Path

import structlog
import tree_sitter_markdown as tsmarkdown
from tree_sitter import Language, Node

from semantic_code_mcp.chunkers.base import BaseTreeSitterChunker
from semantic_code_mcp.models import Chunk, ChunkType

log = structlog.get_logger()


class NodeType(StrEnum):
    """Markdown tree-sitter node types."""

    document = auto()
    section = auto()
    atx_heading = auto()
    setext_heading = auto()
    paragraph = auto()
    inline = auto()


class MarkdownChunker(BaseTreeSitterChunker):
    """Chunker for Markdown files using tree-sitter-markdown.

    Walks nested `section` nodes in the AST. Each section with a heading
    becomes a `ChunkType.section` chunk; sections without headings (preamble)
    become `ChunkType.module` chunks. Nested sections are flattened into
    separate chunks.

    See docs/decisions/005-markdown-chunking.md.
    """

    language = Language(tsmarkdown.language())
    extensions = (".md",)

    def _extract_chunks(self, root: Node, file_path: str, lines: list[str]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for child in root.children:
            if child.type == NodeType.section:
                self._extract_section(child, file_path, lines, chunks)
        return chunks

    def _extract_section(
        self,
        section: Node,
        file_path: str,
        lines: list[str],
        chunks: list[Chunk],
    ) -> None:
        """Extract a chunk from a section node, then recurse into sub-sections."""
        heading = self._find_heading(section)
        child_sections = [c for c in section.children if c.type == NodeType.section]

        # Determine content range (0-indexed rows).
        # For parent sections: from section start to just before first child section.
        # For leaf sections: from section start to section end.
        start_row = section.start_point[0]
        if child_sections:
            end_row = child_sections[0].start_point[0] - 1
        else:
            end_row = section.end_point[0]

        # Skip sections that would produce empty content
        # (end_row can be < start_row when a heading is immediately followed
        # by a sub-section with no content in between)
        if end_row >= start_row:
            # Convert to 1-indexed lines, consistent with base chunker convention.
            # Slice with [start-1 : end] to include the end line.
            start_line = start_row + 1
            end_line = end_row + 1
            content = "\n".join(lines[start_line - 1 : end_line])

            # Strip trailing empty lines from content
            content = content.rstrip("\n")

            # Only emit if there's actual content (not just whitespace)
            if content.strip():
                if heading:
                    name = self._heading_text(heading)
                    chunk_type = ChunkType.section
                else:
                    name = Path(file_path).stem
                    chunk_type = ChunkType.module

                chunks.append(
                    Chunk(
                        file_path=file_path,
                        line_start=start_line,
                        line_end=end_line,
                        content=content,
                        chunk_type=chunk_type,
                        name=name,
                    )
                )

        # Recurse into child sections
        for child in child_sections:
            self._extract_section(child, file_path, lines, chunks)

    def _find_heading(self, section: Node) -> Node | None:
        """Find an atx_heading or setext_heading child of a section node."""
        for child in section.children:
            if child.type in (NodeType.atx_heading, NodeType.setext_heading):
                return child
        return None

    def _heading_text(self, heading: Node) -> str:
        """Extract the heading text from an atx_heading or setext_heading node.

        atx_heading: text is in a direct `inline` child.
        setext_heading: text is in `paragraph` → `inline`.
        """
        if heading.type == NodeType.atx_heading:
            for child in heading.children:
                if child.type == NodeType.inline and child.text:
                    return child.text.decode("utf-8")
        elif heading.type == NodeType.setext_heading:
            for child in heading.children:
                if child.type == NodeType.paragraph:
                    for grandchild in child.children:
                        if grandchild.type == NodeType.inline and grandchild.text:
                            return grandchild.text.decode("utf-8")
        return ""
