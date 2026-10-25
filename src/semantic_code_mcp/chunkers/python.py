"""Tree-sitter based AST chunking for Python code."""

from enum import StrEnum, auto
from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Node

from semantic_code_mcp.chunkers.base import BaseTreeSitterChunker, field_text
from semantic_code_mcp.models import Chunk, ChunkType


class NodeType(StrEnum):
    """Tree-sitter Python AST node types."""

    function_definition = auto()
    class_definition = auto()
    decorated_definition = auto()
    comment = auto()
    newline = auto()
    expression_statement = auto()
    string = auto()


class PythonChunker(BaseTreeSitterChunker):
    """Extracts semantic chunks from Python code using tree-sitter AST.

    Module docstrings, functions, classes and methods become chunks. A decorated
    definition's chunk starts at its first decorator.
    """

    language = Language(tspython.language())
    extensions = (".py",)

    def _extract_chunks(self, root: Node, file_path: str, lines: list[str]) -> list[Chunk]:
        """Extract Python-specific chunks from the AST."""
        chunks: list[Chunk] = []

        # Extract module docstring if present (PEP 257: first statement)
        module_chunk = self._extract_module_docstring(root, file_path, lines)
        if module_chunk:
            chunks.append(module_chunk)

        self._extract_definitions(root, file_path, lines, chunks, in_class=False)
        return chunks

    def _extract_definitions(
        self,
        node: Node,
        file_path: str,
        lines: list[str],
        chunks: list[Chunk],
        in_class: bool,
    ) -> None:
        """Extract the function and class definitions directly inside a node."""
        for child in node.children:
            match child.type:
                case NodeType.function_definition | NodeType.class_definition:
                    self._extract_definition(child, child, file_path, lines, chunks, in_class)
                case NodeType.decorated_definition:
                    # The `definition` field skips decorators and any comments between them.
                    definition = child.child_by_field_name("definition")
                    if definition is not None:
                        self._extract_definition(
                            child, definition, file_path, lines, chunks, in_class
                        )

    def _extract_definition(
        self,
        span_node: Node,
        definition: Node,
        file_path: str,
        lines: list[str],
        chunks: list[Chunk],
        in_class: bool,
    ) -> None:
        """Extract one definition; a class also yields chunks for its methods.

        Args:
            span_node: Node whose lines the chunk covers (the decorated wrapper, if any).
            definition: The function_definition or class_definition itself.
            file_path: Source file path.
            lines: Source code lines.
            chunks: Chunks collected so far.
            in_class: Whether the definition sits in a class body.
        """
        name = field_text(definition, "name")
        if name is None:
            return

        if definition.type == NodeType.class_definition:
            chunks.append(
                self._make_chunk(span_node, span_node, file_path, lines, ChunkType.klass, name)
            )
            body = definition.child_by_field_name("body")
            if body is not None:
                self._extract_definitions(body, file_path, lines, chunks, in_class=True)
            return

        chunk_type = ChunkType.method if in_class else ChunkType.function
        chunks.append(self._make_chunk(span_node, span_node, file_path, lines, chunk_type, name))

    def _extract_module_docstring(
        self,
        root: Node,
        file_path: str,
        lines: list[str],
    ) -> Chunk | None:
        """Extract module-level docstring per PEP 257.

        The module docstring must be the first statement in the file.
        Comments and blank lines before it are allowed, but any other
        statement (import, assignment, etc.) means there is no module docstring.
        """
        for child in root.children:
            if child.type in (NodeType.comment, NodeType.newline):
                continue

            # First real statement must be an expression_statement with a string
            if child.type == NodeType.expression_statement and any(
                part.type == NodeType.string for part in child.children
            ):
                name = Path(file_path).stem
                return self._make_chunk(child, child, file_path, lines, ChunkType.module, name)

            # Any other node type means no module docstring
            return None

        return None
