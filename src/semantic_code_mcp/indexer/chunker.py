"""Tree-sitter based AST chunking for Python code."""

from enum import StrEnum, auto
from pathlib import Path

import structlog
import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

from semantic_code_mcp.models import Chunk, ChunkType

log = structlog.get_logger()

# Initialize Python language
PY_LANGUAGE = Language(tspython.language())


class NodeType(StrEnum):
    """Tree-sitter Python AST node types."""

    function_definition = auto()
    class_definition = auto()
    decorated_definition = auto()


class PythonChunker:
    """Extracts semantic chunks from Python code using tree-sitter AST."""

    def __init__(self) -> None:
        """Initialize the Python chunker."""
        self.parser = Parser(PY_LANGUAGE)

    def chunk_file(self, file_path: str) -> list[Chunk]:
        """Extract chunks from a Python file.

        Args:
            file_path: Path to the Python file.

        Returns:
            List of Chunk objects for functions, classes, and methods.
        """
        path = Path(file_path)
        try:
            content = path.read_text()
        except OSError as e:
            log.warning("failed_to_read_file", file_path=file_path, error=str(e))
            return []

        return self.chunk_string(content, file_path)

    def chunk_string(self, code: str, file_path: str) -> list[Chunk]:
        """Extract chunks from Python code string.

        Args:
            code: Python source code.
            file_path: Path to use in chunk metadata.

        Returns:
            List of Chunk objects.
        """
        if not code.strip():
            return []

        try:
            tree = self.parser.parse(code.encode())
        except (ValueError, UnicodeDecodeError) as e:
            log.warning("parse_failed", file_path=file_path, error=str(e))
            return []

        chunks: list[Chunk] = []
        lines = code.split("\n")

        # Walk the tree and extract functions and classes at module level
        self._extract_from_node(tree.root_node, file_path, lines, chunks, in_class=False)

        log.debug("chunked_file", file_path=file_path, chunks_count=len(chunks))
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
                # Don't recurse into function body for nested functions

            elif child.type == NodeType.decorated_definition:
                # Handle decorated functions and classes
                decorated_child = self._get_decorated_definition(child)
                if decorated_child:
                    if decorated_child.type == NodeType.function_definition:
                        chunk = self._extract_function(
                            child, file_path, lines, in_class, decorated=True
                        )
                        if chunk:
                            chunks.append(chunk)
                    elif decorated_child.type == NodeType.class_definition:
                        chunk = self._extract_class(child, file_path, lines, decorated=True)
                        if chunk:
                            chunks.append(chunk)
                        # Extract methods from within the class body
                        body = decorated_child.child_by_field_name("body")
                        if body:
                            self._extract_from_node(body, file_path, lines, chunks, in_class=True)

            elif child.type == NodeType.class_definition:
                chunk = self._extract_class(child, file_path, lines, decorated=False)
                if chunk:
                    chunks.append(chunk)
                # Extract methods from within the class body
                body = child.child_by_field_name("body")
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
        """Extract a function/method chunk.

        Args:
            node: AST node (function_definition or decorated_definition).
            file_path: Source file path.
            lines: Source code lines.
            in_class: Whether this is a method.
            decorated: Whether the node is a decorated_definition.

        Returns:
            Chunk or None if extraction fails.
        """
        if decorated:
            func_node = self._get_decorated_definition(node)
        else:
            func_node = node

        if not func_node:
            return None

        # Get function name
        name_node = func_node.child_by_field_name("name")
        if not name_node:
            return None
        name = name_node.text.decode()

        # Get line range (1-indexed)
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        # Extract content
        content = "\n".join(lines[start_line - 1 : end_line])

        chunk_type = ChunkType.METHOD if in_class else ChunkType.FUNCTION

        return Chunk(
            file_path=file_path,
            line_start=start_line,
            line_end=end_line,
            content=content,
            chunk_type=chunk_type,
            name=name,
        )

    def _extract_class(
        self,
        node: Node,
        file_path: str,
        lines: list[str],
        decorated: bool = False,
    ) -> Chunk | None:
        """Extract a class chunk.

        Args:
            node: AST node (class_definition or decorated_definition).
            file_path: Source file path.
            lines: Source code lines.
            decorated: Whether the node is a decorated_definition.

        Returns:
            Chunk or None if extraction fails.
        """
        if decorated:
            class_node = self._get_decorated_definition(node)
        else:
            class_node = node

        if not class_node:
            return None

        # Get class name
        name_node = class_node.child_by_field_name("name")
        if not name_node:
            return None
        name = name_node.text.decode()

        # Get line range (1-indexed)
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        # Extract content
        content = "\n".join(lines[start_line - 1 : end_line])

        return Chunk(
            file_path=file_path,
            line_start=start_line,
            line_end=end_line,
            content=content,
            chunk_type=ChunkType.CLASS,
            name=name,
        )
