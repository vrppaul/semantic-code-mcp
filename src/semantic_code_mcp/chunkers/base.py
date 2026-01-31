"""Base class for tree-sitter based code chunkers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import structlog
from tree_sitter import Language, Node, Parser

from semantic_code_mcp.models import Chunk, ChunkType

log = structlog.get_logger()


class BaseTreeSitterChunker(ABC):
    """Base class for tree-sitter based code chunkers.

    Provides shared logic for parsing, file reading, and Chunk construction.
    Subclasses must set `language` and `extensions` class variables,
    then implement `_extract_chunks()` for language-specific AST walking.
    """

    language: ClassVar[Language]
    extensions: ClassVar[tuple[str, ...]]

    def chunk_file(self, file_path: str) -> list[Chunk]:
        """Extract chunks from a source file.

        Args:
            file_path: Path to the source file.

        Returns:
            List of Chunk objects.
        """
        path = Path(file_path)
        try:
            content = path.read_text()
        except OSError as e:
            log.warning("failed_to_read_file", file_path=file_path, error=str(e))
            return []

        return self.chunk_string(content, file_path)

    def chunk_string(self, code: str, file_path: str) -> list[Chunk]:
        """Extract chunks from a code string.

        Thread-safe: creates a fresh Parser per call since tree-sitter
        parsers mutate internal state during parse().

        Args:
            code: Source code.
            file_path: Path to use in chunk metadata.

        Returns:
            List of Chunk objects.
        """
        if not code.strip():
            return []

        try:
            parser = Parser(self.language)
            tree = parser.parse(code.encode())
        except (ValueError, UnicodeDecodeError) as e:
            log.warning("parse_failed", file_path=file_path, error=str(e))
            return []

        lines = code.split("\n")
        chunks = self._extract_chunks(tree.root_node, file_path, lines)

        log.debug("chunked_file", file_path=file_path, chunks_count=len(chunks))
        return chunks

    @abstractmethod
    def _extract_chunks(self, root: Node, file_path: str, lines: list[str]) -> list[Chunk]:
        """Extract language-specific chunks from the AST root.

        Args:
            root: Tree-sitter root node.
            file_path: Source file path.
            lines: Source code split into lines.

        Returns:
            List of Chunk objects.
        """
        ...

    def _make_chunk(
        self,
        node: Node,
        file_path: str,
        lines: list[str],
        chunk_type: ChunkType,
        name: str,
    ) -> Chunk:
        """Create a Chunk from an AST node.

        Args:
            node: AST node defining the line range.
            file_path: Source file path.
            lines: Source code lines.
            chunk_type: Type of chunk.
            name: Name of the chunk (function/class name).

        Returns:
            Chunk object.
        """
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        content = "\n".join(lines[start_line - 1 : end_line])

        return Chunk(
            file_path=file_path,
            line_start=start_line,
            line_end=end_line,
            content=content,
            chunk_type=chunk_type,
            name=name,
        )
