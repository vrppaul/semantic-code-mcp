"""Tree-sitter based AST chunking for Python code."""

from enum import StrEnum, auto
from pathlib import Path
from typing import override

import tree_sitter_python as tspython
from tree_sitter import Language, Node

from semantic_code_mcp.chunkers.base import BaseTreeSitterChunker, ChunkCollector, field_text
from semantic_code_mcp.domain.models import ChunkType


class NodeType(StrEnum):
    """Tree-sitter Python AST node types."""

    function_definition = auto()
    class_definition = auto()
    decorated_definition = auto()
    comment = auto()
    newline = auto()
    expression_statement = auto()
    string = auto()


class FieldName(StrEnum):
    """Tree-sitter Python field names."""

    name = auto()
    body = auto()
    definition = auto()


class PythonChunker(BaseTreeSitterChunker):
    """Extracts semantic chunks from Python code using tree-sitter AST.

    Module docstrings, functions, classes and methods become chunks. A decorated
    definition's chunk starts at its first decorator.
    """

    language = Language(tspython.language())
    extensions = (".py",)

    @override
    def _extract_chunks(self, root: Node, collector: ChunkCollector) -> None:
        """Extract the module docstring, then the definitions under the root."""
        self._extract_module_docstring(root, collector)
        self._extract_definitions(root, collector, in_class=False)

    def _extract_definitions(
        self, node: Node, collector: ChunkCollector, *, in_class: bool
    ) -> None:
        """Extract the function and class definitions directly inside a node."""
        for child in node.children:
            match child.type:
                case NodeType.function_definition | NodeType.class_definition:
                    self._extract_definition(child, child, collector, in_class=in_class)
                case NodeType.decorated_definition:
                    # The `definition` field skips decorators and any comments between them.
                    definition = child.child_by_field_name(FieldName.definition)
                    if definition is not None:
                        self._extract_definition(child, definition, collector, in_class=in_class)

    def _extract_definition(
        self, span_node: Node, definition: Node, collector: ChunkCollector, *, in_class: bool
    ) -> None:
        """Extract one definition; a class also yields chunks for its methods.

        Args:
            span_node: Node whose lines the chunk covers (the decorated wrapper, if any).
            definition: The function_definition or class_definition itself.
            collector: The chunks of the file so far.
            in_class: Whether the definition sits in a class body.
        """
        name = field_text(definition, FieldName.name)
        if name is None:
            return

        if definition.type == NodeType.class_definition:
            collector.add(span_node, span_node, ChunkType.klass, name)
            body = definition.child_by_field_name(FieldName.body)
            if body is not None:
                self._extract_definitions(body, collector, in_class=True)
            return

        chunk_type = ChunkType.method if in_class else ChunkType.function
        collector.add(span_node, span_node, chunk_type, name)

    def _extract_module_docstring(self, root: Node, collector: ChunkCollector) -> None:
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
                collector.add(child, child, ChunkType.module, Path(collector.file_path).stem)

            # Only the first real statement can be the module docstring
            return
