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
    # Sections already cover every line; a file of front matter alone stays out.
    whole_file_fallback = False

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
        """Extract a chunk from a section's own content, then recurse into sub-sections.

        A section's own content is everything before its first sub-section: the
        heading and the blocks under it.
        """
        own_content = [child for child in section.children if child.type != NodeType.section]
        if own_content:
            heading = self._find_heading(section)
            if heading is not None:
                name = self._heading_text(heading)
                chunk_type = ChunkType.section
            else:
                name = Path(file_path).stem
                chunk_type = ChunkType.module
            chunk = self._make_chunk(
                own_content[0], own_content[-1], file_path, lines, chunk_type, name
            )
            if chunk.content.strip():
                chunks.append(chunk)

        for child in section.children:
            if child.type == NodeType.section:
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
