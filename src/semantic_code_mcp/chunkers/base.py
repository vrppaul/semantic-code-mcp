"""Base class for tree-sitter based code chunkers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import structlog
from tree_sitter import Language, Node, Parser

from semantic_code_mcp.models import Chunk, ChunkType

log = structlog.get_logger()

SOURCE_ENCODING = "utf-8"
# Undecodable bytes become U+FFFD instead of dropping the whole file.
DECODE_ERROR_HANDLING = "replace"
# A NUL among the first bytes marks a binary or UTF-16 file, as git decides it.
BINARY_SNIFF_BYTES = 8000
NUL_BYTE = b"\0"
# Larger files are generated or data, not source anyone searches by meaning.
MAX_SOURCE_BYTES = 1_000_000
# A file without definitions is indexed in windows of this many lines.
FALLBACK_WINDOW_LINES = 100


def node_last_row(node: Node) -> int:
    """The last row a node covers (0-indexed).

    Tree-sitter ends a node that swallows its trailing newline at column 0 of the
    next row; that row holds none of the node's text.
    """
    end_row, end_column = node.end_point
    if end_column == 0 and end_row > node.start_point[0]:
        return end_row - 1
    return end_row


def field_text(node: Node, field_name: str) -> str | None:
    """Text of a node's named field, such as the `name` of a function."""
    field = node.child_by_field_name(field_name)
    if field is None or field.text is None:
        return None
    return field.text.decode()


class BaseTreeSitterChunker(ABC):
    """Base class for tree-sitter based code chunkers.

    Provides shared logic for parsing, file reading, and Chunk construction.
    Subclasses must set `language` and `extensions` class variables,
    then implement `_extract_chunks()` for language-specific AST walking.

    A file in which a subclass finds no definition — nothing but a module docstring,
    or nothing at all — is indexed whole instead, in windows of lines, so a script or
    a settings module stays searchable. Subclasses that already chunk every part of a
    file set `whole_file_fallback = False`. Binary files and files over
    `MAX_SOURCE_BYTES` are skipped.
    """

    language: ClassVar[Language]
    extensions: ClassVar[tuple[str, ...]]
    whole_file_fallback: ClassVar[bool] = True

    def chunk_file(self, file_path: str) -> list[Chunk]:
        """Extract chunks from a source file.

        Args:
            file_path: Path to the source file.

        Returns:
            List of Chunk objects; empty when the file cannot be read, is binary or too large.
        """
        try:
            source = Path(file_path).read_bytes()
        except OSError as error:
            log.warning("failed_to_read_file", file_path=file_path, error=str(error))
            return []

        if len(source) > MAX_SOURCE_BYTES:
            log.warning("skipped_large_file", file_path=file_path, size_bytes=len(source))
            return []
        if NUL_BYTE in source[:BINARY_SNIFF_BYTES]:
            log.warning("skipped_binary_file", file_path=file_path)
            return []

        try:
            code = source.decode(SOURCE_ENCODING)
        except UnicodeDecodeError as error:
            log.warning("file_not_utf8", file_path=file_path, error=str(error))
            code = source.decode(SOURCE_ENCODING, errors=DECODE_ERROR_HANDLING)

        return self.chunk_string(code, file_path)

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
        except ValueError as error:
            log.warning("parse_failed", file_path=file_path, error=str(error))
            return []

        lines = code.split("\n")
        chunks = self._extract_chunks(tree.root_node, file_path, lines)
        has_definitions = any(chunk.chunk_type != ChunkType.module for chunk in chunks)
        if self.whole_file_fallback and not has_definitions:
            chunks = self._whole_file_windows(tree.root_node, file_path, lines)

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

    def _whole_file_windows(self, root: Node, file_path: str, lines: list[str]) -> list[Chunk]:
        """The whole file as `module` chunks of at most FALLBACK_WINDOW_LINES lines each."""
        name = Path(file_path).stem
        last_row = node_last_row(root)
        windows = [
            self._make_row_chunk(
                first_row,
                min(first_row + FALLBACK_WINDOW_LINES, last_row + 1) - 1,
                file_path,
                lines,
                ChunkType.module,
                name,
            )
            for first_row in range(root.start_point[0], last_row + 1, FALLBACK_WINDOW_LINES)
        ]
        return [window for window in windows if window.content.strip()]

    def _make_chunk(
        self,
        first_node: Node,
        last_node: Node,
        file_path: str,
        lines: list[str],
        chunk_type: ChunkType,
        name: str,
    ) -> Chunk:
        """Create a Chunk spanning from the first node's row to the last node's row.

        Args:
            first_node: Node whose start row opens the chunk (a decorator, a doc comment).
            last_node: Node whose last row closes the chunk.
            file_path: Source file path.
            lines: Source code lines.
            chunk_type: Type of chunk.
            name: Name of the chunk (function/class name).

        Returns:
            Chunk object.
        """
        return self._make_row_chunk(
            first_node.start_point[0], node_last_row(last_node), file_path, lines, chunk_type, name
        )

    def _make_row_chunk(
        self,
        first_row: int,
        last_row: int,
        file_path: str,
        lines: list[str],
        chunk_type: ChunkType,
        name: str,
    ) -> Chunk:
        """Create a Chunk of the rows from first_row to last_row (0-indexed, inclusive).

        Trailing blank lines are left out, so the line range selects exactly the content.
        """
        while last_row > first_row and not lines[last_row].strip():
            last_row -= 1

        return Chunk(
            file_path=file_path,
            line_start=first_row + 1,
            line_end=last_row + 1,
            content="\n".join(lines[first_row : last_row + 1]),
            chunk_type=chunk_type,
            name=name,
        )
