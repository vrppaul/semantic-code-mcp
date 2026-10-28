"""Markdown tree-sitter chunker — heading-based section extraction."""

from enum import StrEnum, auto
from typing import override

import tree_sitter
import tree_sitter_markdown as tsmarkdown
from tree_sitter import Node

from semantic_code_mcp.chunkers.base import SOURCE_ENCODING, BaseTreeSitterChunker, ChunkCollector
from semantic_code_mcp.domain.models import ChunkKind, Language


class NodeType(StrEnum):
    """Markdown tree-sitter node types."""

    section = auto()
    atx_heading = auto()
    setext_heading = auto()
    paragraph = auto()
    inline = auto()


class MarkdownChunker(BaseTreeSitterChunker):
    """Chunker for Markdown files using tree-sitter-markdown.

    Walks nested `section` nodes in the AST. Each section with a heading
    becomes a `ChunkKind.section` chunk; sections without headings (preamble)
    become `ChunkKind.module` chunks. Nested sections are flattened into
    separate chunks, each naming the headings above it as its parent,
    `Guide > Install` for a section under `## Install` under `# Guide`.

    See docs/decisions/005-markdown-chunking.md.
    """

    language = Language.markdown
    # tree-sitter-markdown 0.5.4's stub names CapsuleType without importing it, which reads
    # as any type, so the checker takes the deprecated overload for an int
    grammar = tree_sitter.Language(tsmarkdown.language())  # ty: ignore[deprecated]
    extensions = (".md",)
    # Sections already cover every line; a file of front matter alone stays out.
    whole_file_fallback = False

    @override
    def _extract_chunks(self, root: Node, collector: ChunkCollector) -> None:
        for child in root.children:
            if child.type == NodeType.section:
                self._extract_section(child, collector, parent=None)

    def _extract_section(
        self, section: Node, collector: ChunkCollector, *, parent: str | None
    ) -> None:
        """Extract a chunk from a section's own content, then recurse into sub-sections.

        A section's own content is everything before its first sub-section: the
        heading and the blocks under it.
        """
        heading = self._find_heading(section)
        heading_text = None if heading is None else self._heading_text(heading)
        own_content = [child for child in section.children if child.type != NodeType.section]
        if own_content:
            if heading_text is not None:
                collector.add(
                    own_content[0], own_content[-1], ChunkKind.section, heading_text, parent=parent
                )
            else:
                collector.add(
                    own_content[0],
                    own_content[-1],
                    ChunkKind.module,
                    collector.path.stem,
                    parent=parent,
                )

        for child in section.children:
            if child.type == NodeType.section:
                self._extract_section(child, collector, parent=self.qualified(parent, heading_text))

    def _find_heading(self, section: Node) -> Node | None:
        """Find an atx_heading or setext_heading child of a section node."""
        for child in section.children:
            if child.type in (NodeType.atx_heading, NodeType.setext_heading):
                return child
        return None

    def _heading_text(self, heading: Node) -> str | None:
        """Extract the heading text from an atx_heading or setext_heading node; None for a
        heading without text, as a bare `##`.

        atx_heading: text is in a direct `inline` child.
        setext_heading: text is in `paragraph` → `inline`.
        """
        if heading.type == NodeType.atx_heading:
            for child in heading.children:
                if child.type == NodeType.inline and child.text:
                    return child.text.decode(SOURCE_ENCODING)
        elif heading.type == NodeType.setext_heading:
            for child in heading.children:
                if child.type == NodeType.paragraph:
                    for grandchild in child.children:
                        if grandchild.type == NodeType.inline and grandchild.text:
                            return grandchild.text.decode(SOURCE_ENCODING)
        return None
