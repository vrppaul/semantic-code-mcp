"""Tree-sitter based AST chunking for Rust code."""

from enum import StrEnum, auto
from pathlib import Path

import tree_sitter_rust as tsrust
from tree_sitter import Language, Node

from semantic_code_mcp.chunkers.base import BaseTreeSitterChunker, field_text
from semantic_code_mcp.models import Chunk, ChunkType


class NodeType(StrEnum):
    """Tree-sitter Rust AST node types."""

    line_comment = auto()
    inner_doc_comment_marker = auto()
    outer_doc_comment_marker = auto()
    function_item = auto()
    struct_item = auto()
    enum_item = auto()
    trait_item = auto()
    impl_item = auto()
    mod_item = auto()
    attribute_item = auto()
    generic_type = auto()
    reference_type = auto()
    pointer_type = auto()
    scoped_type_identifier = auto()


# Type nodes that wrap another type, and the field holding the wrapped one:
# `Stack<T>` → `Stack`, `&Stack` → `Stack`, `fmt::Display` → `Display`.
TYPE_WRAPPER_FIELDS: dict[str, str] = {
    NodeType.generic_type: "type",
    NodeType.reference_type: "type",
    NodeType.pointer_type: "type",
    NodeType.scoped_type_identifier: "name",
}


class RustChunker(BaseTreeSitterChunker):
    """Extracts semantic chunks from Rust code using tree-sitter AST.

    Functions, structs, enums, traits and impl blocks become chunks, with the
    attributes and `///` doc comments above them; functions inside impl and trait
    bodies become methods; items inside inline `mod` blocks are extracted too.
    """

    language = Language(tsrust.language())
    extensions = (".rs",)

    def _extract_chunks(self, root: Node, file_path: str, lines: list[str]) -> list[Chunk]:
        """Extract Rust-specific chunks from the AST."""
        chunks: list[Chunk] = []

        module_chunk = self._extract_module_doc(root, file_path, lines)
        if module_chunk:
            chunks.append(module_chunk)

        self._walk_items(root, file_path, lines, chunks, ChunkType.function)
        return chunks

    def _extract_module_doc(self, root: Node, file_path: str, lines: list[str]) -> Chunk | None:
        """Extract the `//!` inner doc comments at the top of a module."""
        doc_comments: list[Node] = []
        for child in root.children:
            if child.type != NodeType.line_comment:
                break
            if self._is_inner_doc_comment(child):
                doc_comments.append(child)

        if not doc_comments:
            return None
        name = Path(file_path).stem
        return self._make_chunk(
            doc_comments[0], doc_comments[-1], file_path, lines, ChunkType.module, name
        )

    def _walk_items(
        self,
        container: Node,
        file_path: str,
        lines: list[str],
        chunks: list[Chunk],
        function_type: ChunkType,
    ) -> None:
        """Extract the items directly inside a container node.

        Args:
            container: A source file, an inline module body, or an impl/trait body.
            file_path: Source file path.
            lines: Source code lines.
            chunks: Chunks collected so far.
            function_type: What a function here is — a function or a method.
        """
        items = list(container.children)
        for index, item in enumerate(items):
            start_node = items[self._find_item_start(items, index)]

            match item.type:
                case NodeType.function_item:
                    self._append_named_item(
                        start_node, item, file_path, lines, chunks, function_type
                    )

                case NodeType.struct_item | NodeType.enum_item:
                    self._append_named_item(
                        start_node, item, file_path, lines, chunks, ChunkType.klass
                    )

                case NodeType.trait_item:
                    self._append_named_item(
                        start_node, item, file_path, lines, chunks, ChunkType.klass
                    )
                    self._walk_body(item, file_path, lines, chunks, ChunkType.method)

                case NodeType.impl_item:
                    impl_name = self._impl_name(item)
                    if impl_name is not None:
                        chunks.append(
                            self._make_chunk(
                                start_node, item, file_path, lines, ChunkType.klass, impl_name
                            )
                        )
                        self._walk_body(item, file_path, lines, chunks, ChunkType.method)

                case NodeType.mod_item:
                    self._walk_body(item, file_path, lines, chunks, function_type)

    def _walk_body(
        self,
        item: Node,
        file_path: str,
        lines: list[str],
        chunks: list[Chunk],
        function_type: ChunkType,
    ) -> None:
        """Walk the `body` of an impl, trait or inline module, if it has one."""
        body = item.child_by_field_name("body")
        if body is not None:
            self._walk_items(body, file_path, lines, chunks, function_type)

    def _append_named_item(
        self,
        start_node: Node,
        item: Node,
        file_path: str,
        lines: list[str],
        chunks: list[Chunk],
        chunk_type: ChunkType,
    ) -> None:
        """Append a chunk for an item named by its `name` field."""
        name = field_text(item, "name")
        if name is not None:
            chunks.append(self._make_chunk(start_node, item, file_path, lines, chunk_type, name))

    def _find_item_start(self, siblings: list[Node], item_index: int) -> int:
        """Index of the first attribute or `///` doc comment directly above an item."""
        start_index = item_index
        for previous_index in range(item_index - 1, -1, -1):
            previous = siblings[previous_index]
            if previous.type == NodeType.attribute_item or self._is_outer_doc_comment(previous):
                start_index = previous_index
            else:
                break
        return start_index

    def _is_inner_doc_comment(self, node: Node) -> bool:
        """Check if a comment is a `//!` inner doc comment."""
        return any(child.type == NodeType.inner_doc_comment_marker for child in node.children)

    def _is_outer_doc_comment(self, node: Node) -> bool:
        """Check if a comment is a `///` outer doc comment."""
        return any(child.type == NodeType.outer_doc_comment_marker for child in node.children)

    def _impl_name(self, impl_node: Node) -> str | None:
        """Name an impl block: `Stack` for `impl<T> Stack<T>`, `Display for Stack` for
        `impl fmt::Display for Stack<i32>`."""
        type_node = impl_node.child_by_field_name("type")
        if type_node is None:
            return None
        type_name = self._base_type_name(type_node)

        trait_node = impl_node.child_by_field_name("trait")
        if trait_node is None:
            return type_name
        return f"{self._base_type_name(trait_node)} for {type_name}"

    def _base_type_name(self, type_node: Node) -> str:
        """Name of a type without its generic arguments, reference or module path."""
        current = type_node
        while (wrapped_field := TYPE_WRAPPER_FIELDS.get(current.type)) is not None:
            wrapped = current.child_by_field_name(wrapped_field)
            if wrapped is None:
                break
            current = wrapped
        return current.text.decode() if current.text is not None else ""
