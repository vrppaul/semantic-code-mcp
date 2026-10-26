"""Base class for tree-sitter based code chunkers."""

import hashlib
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import ClassVar

import structlog
import tree_sitter
from tree_sitter import Node, Parser

from semantic_code_mcp.domain.models import Chunk, ChunkKind, Language

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
# A chunk's id is this many hex digits of a digest of what identifies it, those parts
# joined by NUL: the path and the occurrence never hold one and the text comes last, so
# no two keys read alike
CHUNK_ID_LENGTH = 32
CHUNK_ID_SEPARATOR = "\0"


def chunk_identifier(path: Path, occurrence: int, embedded_text: str) -> str:
    """The id of a chunk: a digest of its file's path, the text the model embeds for it,
    and how many chunks of that text come before it in the file.

    Lines inserted or edited elsewhere in the file leave it as it is; an edit to its own
    text, or a move to another file, gives it a new one. So the vector stored under an id
    is always the vector of that text.
    """
    key = CHUNK_ID_SEPARATOR.join([path.as_posix(), str(occurrence), embedded_text])
    return hashlib.sha256(key.encode()).hexdigest()[:CHUNK_ID_LENGTH]


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


class ChunkCollector:
    """The chunks of one file, built from tree-sitter nodes as a chunker finds them.

    A chunk spans whole lines, from the first line of its first node to the last line
    of its last node; trailing blank lines are left out, so the line range selects
    exactly the content, and a chunk of blank lines alone is not kept.
    """

    def __init__(self, path: Path, language: Language, lines: list[str]) -> None:
        self.path = path
        self._language = language
        self._lines = lines
        self._chunks: list[Chunk] = []
        # How many chunks of each text the file has given so far
        self._occurrences: Counter[str] = Counter()

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    @property
    def has_definitions(self) -> bool:
        """Whether any chunk is more than module-level text."""
        return any(chunk.kind != ChunkKind.module for chunk in self._chunks)

    def add(
        self,
        first_node: Node,
        last_node: Node,
        kind: ChunkKind,
        name: str,
        *,
        parent: str | None,
    ) -> None:
        """Add a chunk spanning from the first node's row to the last node's row."""
        self.add_rows(
            first_node.start_point[0], node_last_row(last_node), kind, name, parent=parent
        )

    def add_rows(
        self, first_row: int, last_row: int, kind: ChunkKind, name: str, *, parent: str | None
    ) -> None:
        """Add a chunk of the rows from first_row to last_row (0-indexed, inclusive)."""
        while last_row > first_row and not self._lines[last_row].strip():
            last_row -= 1
        content = "\n".join(self._lines[first_row : last_row + 1])
        if not content.strip():
            return
        occurrence = self._occurrences[content]
        self._occurrences[content] += 1
        self._chunks.append(
            Chunk(
                # The model embeds a chunk's content as it is
                chunk_id=chunk_identifier(self.path, occurrence, content),
                path=self.path,
                language=self._language,
                kind=kind,
                name=name,
                parent=parent,
                line_start=first_row + 1,
                line_end=last_row + 1,
                content=content,
            )
        )

    def replace_with_windows(self, root: Node) -> None:
        """Drop what was found and cover the whole file with `module` chunks instead,
        each at most FALLBACK_WINDOW_LINES lines."""
        self._chunks = []
        self._occurrences.clear()
        last_row = node_last_row(root)
        for first_row in range(root.start_point[0], last_row + 1, FALLBACK_WINDOW_LINES):
            window_last_row = min(first_row + FALLBACK_WINDOW_LINES, last_row + 1) - 1
            self.add_rows(first_row, window_last_row, ChunkKind.module, self.path.stem, parent=None)


class BaseTreeSitterChunker(ABC):
    """Base class for tree-sitter based code chunkers.

    Provides shared logic for parsing, file reading, and Chunk construction.
    Subclasses set `language`, its tree-sitter `grammar`, the `extensions` it reads and
    the `parent_separator` of its qualified names, then implement `_extract_chunks()` for
    language-specific AST walking.

    A file in which a subclass finds no definition — nothing but a module docstring,
    or nothing at all — is indexed whole instead, in windows of lines, so a script or
    a settings module stays searchable. Subclasses that already chunk every part of a
    file set `whole_file_fallback = False`. Binary files and files over
    `MAX_SOURCE_BYTES` are skipped.
    """

    language: ClassVar[Language]
    grammar: ClassVar[tree_sitter.Language]
    extensions: ClassVar[tuple[str, ...]]
    # Joins an owner's name to what is defined in it, as the language writes it
    parent_separator: ClassVar[str]
    whole_file_fallback: ClassVar[bool] = True

    def chunk_file(self, project_root: Path, path: Path) -> list[Chunk]:
        """The chunks of the file at `path` from the project root; empty when it cannot be
        read, is binary or is too large."""
        file_path = str(path)
        full_path = project_root / path
        try:
            size_bytes = full_path.stat().st_size
            if size_bytes > MAX_SOURCE_BYTES:
                log.warning("skipped_large_file", file_path=file_path, size_bytes=size_bytes)
                return []
            source = full_path.read_bytes()
        except OSError as error:
            log.warning("failed_to_read_file", file_path=file_path, error=str(error))
            return []

        if NUL_BYTE in source[:BINARY_SNIFF_BYTES]:
            log.warning("skipped_binary_file", file_path=file_path)
            return []

        try:
            code = source.decode(SOURCE_ENCODING)
        except UnicodeDecodeError as error:
            log.warning("file_not_utf8", file_path=file_path, error=str(error))
            code = source.decode(SOURCE_ENCODING, errors=DECODE_ERROR_HANDLING)

        return self.chunk_string(code, path)

    def chunk_string(self, code: str, path: Path) -> list[Chunk]:
        """The chunks of source code, each naming `path` as its file.

        Thread-safe: creates a fresh Parser per call since tree-sitter
        parsers mutate internal state during parse().
        """
        if not code.strip():
            return []

        try:
            parser = Parser(self.grammar)
            tree = parser.parse(code.encode())
        except ValueError as error:
            log.warning("parse_failed", file_path=str(path), error=str(error))
            return []

        collector = ChunkCollector(path, self.language, code.split("\n"))
        self._extract_chunks(tree.root_node, collector)
        if self.whole_file_fallback and not collector.has_definitions:
            collector.replace_with_windows(tree.root_node)

        log.debug("chunked_file", file_path=str(path), chunks_count=len(collector.chunks))
        return collector.chunks

    def qualified(self, parent: str | None, name: str | None) -> str | None:
        """The qualified name of `name` defined in `parent`; an owner without a name adds
        nothing to its parent's."""
        if name is None:
            return parent
        if parent is None:
            return name
        return f"{parent}{self.parent_separator}{name}"

    @abstractmethod
    def _extract_chunks(self, root: Node, collector: ChunkCollector) -> None:
        """Find the language's chunks under the AST root and add them to the collector."""
