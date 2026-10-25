"""Tree-sitter based AST chunking for Rust code."""

from enum import StrEnum, auto
from pathlib import Path
from typing import override

import structlog
import tree_sitter_rust as tsrust
from tree_sitter import Language, Node

from semantic_code_mcp.chunkers.base import BaseTreeSitterChunker, ChunkCollector, field_text
from semantic_code_mcp.domain.models import ChunkType

log = structlog.get_logger()


class NodeType(StrEnum):
    """Tree-sitter Rust AST node types."""

    line_comment = auto()
    inner_doc_comment_marker = auto()
    outer_doc_comment_marker = auto()
    inner_attribute_item = auto()
    attribute_item = auto()
    function_item = auto()
    function_signature_item = auto()
    struct_item = auto()
    enum_item = auto()
    trait_item = auto()
    impl_item = auto()
    mod_item = auto()
    generic_type = auto()
    reference_type = auto()
    pointer_type = auto()
    scoped_type_identifier = auto()
    # The `!` of a negative impl, `impl !Send for Handle`
    negation = "!"


class FieldName(StrEnum):
    """Tree-sitter Rust field names."""

    name = auto()
    body = auto()
    type = auto()
    trait = auto()
    type_arguments = auto()


# Type nodes that wrap another type, and the field holding the wrapped one:
# `Stack<T>` → `Stack`, `&Stack` → `Stack`, `fmt::Display` → `Display`.
TYPE_WRAPPER_FIELDS: dict[str, FieldName] = {
    NodeType.generic_type: FieldName.type,
    NodeType.reference_type: FieldName.type,
    NodeType.pointer_type: FieldName.type,
    NodeType.scoped_type_identifier: FieldName.name,
}

# Items that become chunks or hold items that do; anything else (`use`, `const`, a
# comment) is passed over without looking for its attributes.
CHUNKED_ITEM_TYPES = frozenset(
    {
        NodeType.function_item,
        NodeType.function_signature_item,
        NodeType.struct_item,
        NodeType.enum_item,
        NodeType.trait_item,
        NodeType.impl_item,
        NodeType.mod_item,
    }
)


class RustChunker(BaseTreeSitterChunker):
    """Extracts semantic chunks from Rust code using tree-sitter AST.

    Functions, structs, enums, traits and impl blocks become chunks, with the
    attributes and `///` or `/** */` doc comments above them; functions and required
    method signatures inside impl and trait bodies become methods; items inside inline
    `mod` blocks are extracted too.
    """

    language = Language(tsrust.language())
    extensions = (".rs",)

    @override
    def _extract_chunks(self, root: Node, collector: ChunkCollector) -> None:
        """Extract the module doc, then the items under the root."""
        self._extract_module_doc(root, collector)
        self._walk_items(root, collector, ChunkType.function)

    def _extract_module_doc(self, root: Node, collector: ChunkCollector) -> None:
        """Extract the `//!` inner doc comments at the top of a module.

        Inner attributes (`#![...]`) and plain comments may stand among them.
        """
        doc_comments: list[Node] = []
        for child in root.children:
            if child.type not in (NodeType.line_comment, NodeType.inner_attribute_item):
                break
            if self._is_inner_doc_comment(child):
                doc_comments.append(child)

        if doc_comments:
            name = Path(collector.file_path).stem
            collector.add(doc_comments[0], doc_comments[-1], ChunkType.module, name)

    def _walk_items(
        self, container: Node, collector: ChunkCollector, function_type: ChunkType
    ) -> None:
        """Extract the items directly inside a container node.

        Args:
            container: A source file, an inline module body, or an impl/trait body.
            collector: The chunks of the file so far.
            function_type: What a function here is — a function or a method.
        """
        siblings = list(container.children)
        for index, item in enumerate(siblings):
            if item.type not in CHUNKED_ITEM_TYPES:
                continue
            start_node = self._item_start(siblings, index)

            match item.type:
                case NodeType.function_item | NodeType.function_signature_item:
                    self._append_named_item(start_node, item, collector, function_type)

                case NodeType.struct_item | NodeType.enum_item:
                    self._append_named_item(start_node, item, collector, ChunkType.klass)

                case NodeType.trait_item:
                    self._append_named_item(start_node, item, collector, ChunkType.klass)
                    self._walk_body(item, collector, ChunkType.method)

                case NodeType.impl_item:
                    self._append_impl(start_node, item, collector)

                case NodeType.mod_item:
                    self._walk_body(item, collector, function_type)

    def _append_impl(self, start_node: Node, impl_node: Node, collector: ChunkCollector) -> None:
        """Append a chunk for an impl block, then one for each of its methods."""
        impl_name = self._impl_name(impl_node)
        if impl_name is None:
            log.debug("unnamed_impl_skipped", file_path=collector.file_path)
            return
        collector.add(start_node, impl_node, ChunkType.klass, impl_name)
        self._walk_body(impl_node, collector, ChunkType.method)

    def _walk_body(self, item: Node, collector: ChunkCollector, function_type: ChunkType) -> None:
        """Walk the `body` of an impl, trait or inline module, if it has one."""
        body = item.child_by_field_name(FieldName.body)
        if body is not None:
            self._walk_items(body, collector, function_type)

    def _append_named_item(
        self, start_node: Node, item: Node, collector: ChunkCollector, chunk_type: ChunkType
    ) -> None:
        """Append a chunk for an item named by its `name` field."""
        name = field_text(item, FieldName.name)
        if name is not None:
            collector.add(start_node, item, chunk_type, name)

    def _item_start(self, siblings: list[Node], item_index: int) -> Node:
        """The first attribute or outer doc comment directly above an item, or the item."""
        start = siblings[item_index]
        for previous in reversed(siblings[:item_index]):
            is_attached = previous.type == NodeType.attribute_item or self._is_outer_doc_comment(
                previous
            )
            if not is_attached:
                break
            start = previous
        return start

    def _is_inner_doc_comment(self, node: Node) -> bool:
        """Check if a comment is a `//!` inner doc comment."""
        return any(child.type == NodeType.inner_doc_comment_marker for child in node.children)

    def _is_outer_doc_comment(self, node: Node) -> bool:
        """Check if a comment is a `///` or `/** */` outer doc comment."""
        return any(child.type == NodeType.outer_doc_comment_marker for child in node.children)

    def _impl_name(self, impl_node: Node) -> str | None:
        """Name an impl block: `Stack` for `impl<T> Stack<T>`, `Display for Stack` for
        `impl fmt::Display for Stack<i32>`, `!Send for Handle` for a negative impl,
        `From<u8> for Level` for a generic trait."""
        type_node = impl_node.child_by_field_name(FieldName.type)
        if type_node is None:
            return None
        type_name = self._base_type_name(type_node)

        trait_node = impl_node.child_by_field_name(FieldName.trait)
        if trait_node is None or type_name is None:
            return type_name
        trait_name = self._trait_name(trait_node)
        if trait_name is None:
            return None
        is_negative = any(child.type == NodeType.negation for child in impl_node.children)
        negation = NodeType.negation if is_negative else ""
        return f"{negation}{trait_name} for {type_name}"

    def _trait_name(self, trait_node: Node) -> str | None:
        """A trait's name with its type arguments, which tell `From<u8>` from `From<u16>`."""
        base_name = self._base_type_name(trait_node)
        if trait_node.type != NodeType.generic_type or base_name is None:
            return base_name
        arguments = field_text(trait_node, FieldName.type_arguments)
        return base_name if arguments is None else f"{base_name}{arguments}"

    def _base_type_name(self, type_node: Node) -> str | None:
        """Name of a type without its generic arguments, reference or module path."""
        current = type_node
        while (wrapped_field := TYPE_WRAPPER_FIELDS.get(current.type)) is not None:
            wrapped = current.child_by_field_name(wrapped_field)
            if wrapped is None:
                break
            current = wrapped
        if current.text is None:
            return None
        return current.text.decode()
