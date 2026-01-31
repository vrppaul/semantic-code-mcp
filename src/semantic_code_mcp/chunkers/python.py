"""Tree-sitter based AST chunking for Python code."""

from enum import StrEnum, auto
from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Node

from semantic_code_mcp.chunkers.base import BaseTreeSitterChunker
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
    """Extracts semantic chunks from Python code using tree-sitter AST."""

    language = Language(tspython.language())
    extensions = (".py",)

    def _extract_chunks(self, root: Node, file_path: str, lines: list[str]) -> list[Chunk]:
        """Extract Python-specific chunks from the AST."""
        chunks: list[Chunk] = []

        # Extract module docstring if present (PEP 257: first statement)
        module_chunk = self._extract_module_docstring(root, file_path, lines)
        if module_chunk:
            chunks.append(module_chunk)

        # Walk the tree and extract functions and classes at module level
        self._extract_from_node(root, file_path, lines, chunks, in_class=False)

        return chunks

    def _extract_from_node(
        self,
        node: Node,
        file_path: str,
        lines: list[str],
        chunks: list[Chunk],
        in_class: bool,
    ) -> None:
        """Recursively extract chunks from AST nodes."""
        for child in node.children:
            if child.type == NodeType.function_definition:
                chunk = self._extract_function(child, file_path, lines, in_class)
                if chunk:
                    chunks.append(chunk)

            elif child.type == NodeType.decorated_definition:
                self._extract_decorated(child, file_path, lines, chunks, in_class)

            elif child.type == NodeType.class_definition:
                self._extract_class_with_methods(
                    child,
                    child,
                    file_path,
                    lines,
                    chunks,
                    decorated=False,
                )

    def _extract_decorated(
        self,
        node: Node,
        file_path: str,
        lines: list[str],
        chunks: list[Chunk],
        in_class: bool,
    ) -> None:
        """Handle a decorated_definition node."""
        decorated_child = self._get_decorated_definition(node)
        if not decorated_child:
            return

        if decorated_child.type == NodeType.function_definition:
            chunk = self._extract_function(node, file_path, lines, in_class, decorated=True)
            if chunk:
                chunks.append(chunk)
        elif decorated_child.type == NodeType.class_definition:
            self._extract_class_with_methods(
                node,
                decorated_child,
                file_path,
                lines,
                chunks,
                decorated=True,
            )

    def _extract_class_with_methods(
        self,
        node: Node,
        class_node: Node,
        file_path: str,
        lines: list[str],
        chunks: list[Chunk],
        decorated: bool,
    ) -> None:
        """Extract a class chunk and recurse into its body for methods."""
        chunk = self._extract_class(node, file_path, lines, decorated=decorated)
        if chunk:
            chunks.append(chunk)
        body = class_node.child_by_field_name("body")
        if body:
            self._extract_from_node(body, file_path, lines, chunks, in_class=True)

    def _get_decorated_definition(self, node: Node) -> Node | None:
        """Get the actual definition from a decorated_definition node."""
        for child in node.children:
            if child.type in NodeType:
                if child.type == NodeType.decorated_definition:
                    return self._get_decorated_definition(child)
                return child
        return None

    def _extract_function(
        self,
        node: Node,
        file_path: str,
        lines: list[str],
        in_class: bool,
        decorated: bool = False,
    ) -> Chunk | None:
        """Extract a function/method chunk."""
        if decorated:
            func_node = self._get_decorated_definition(node)
        else:
            func_node = node

        if not func_node:
            return None

        name_node = func_node.child_by_field_name("name")
        if not name_node:
            return None
        if name_node.text is None:
            return None
        name = name_node.text.decode()

        chunk_type = ChunkType.method if in_class else ChunkType.function
        return self._make_chunk(node, file_path, lines, chunk_type, name)

    def _extract_class(
        self,
        node: Node,
        file_path: str,
        lines: list[str],
        decorated: bool = False,
    ) -> Chunk | None:
        """Extract a class chunk."""
        if decorated:
            class_node = self._get_decorated_definition(node)
        else:
            class_node = node

        if not class_node:
            return None

        name_node = class_node.child_by_field_name("name")
        if not name_node:
            return None
        if name_node.text is None:
            return None
        name = name_node.text.decode()

        return self._make_chunk(node, file_path, lines, ChunkType.klass, name)

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
            if child.type == NodeType.expression_statement:
                for sub in child.children:
                    if sub.type == NodeType.string:
                        name = Path(file_path).stem
                        return self._make_chunk(child, file_path, lines, ChunkType.module, name)

            # Any other node type means no module docstring
            return None

        return None
