"""Tree-sitter based AST chunking for Rust code."""

from pathlib import Path

import structlog
import tree_sitter_rust as tsrust
from tree_sitter import Language, Node

from semantic_code_mcp.chunkers.base import BaseTreeSitterChunker
from semantic_code_mcp.models import Chunk, ChunkType

log = structlog.get_logger()


class RustChunker(BaseTreeSitterChunker):
    """Extracts semantic chunks from Rust code using tree-sitter AST."""

    language = Language(tsrust.language())
    extensions = (".rs",)

    def _extract_chunks(self, root: Node, file_path: str, lines: list[str]) -> list[Chunk]:
        """Extract Rust-specific chunks from the AST."""
        chunks: list[Chunk] = []

        # Extract //! module doc comments (consecutive inner doc comments at top)
        module_chunk = self._extract_module_doc(root, file_path, lines)
        if module_chunk:
            chunks.append(module_chunk)

        # Walk top-level items, tracking preceding attributes/doc comments
        self._walk_items(root, file_path, lines, chunks)

        return chunks

    def _extract_module_doc(self, root: Node, file_path: str, lines: list[str]) -> Chunk | None:
        """Extract //! inner doc comments at module level."""
        doc_lines_start: int | None = None
        doc_lines_end: int | None = None

        for child in root.children:
            if child.type == "line_comment" and self._is_inner_doc_comment(child):
                if doc_lines_start is None:
                    doc_lines_start = child.start_point[0]
                doc_lines_end = child.end_point[0]
            elif child.type == "line_comment":
                # Regular comment, skip
                continue
            else:
                break

        if doc_lines_start is not None and doc_lines_end is not None:
            start = doc_lines_start + 1
            end = doc_lines_end + 1
            content = "\n".join(lines[start - 1 : end])
            name = Path(file_path).stem
            return Chunk(
                file_path=file_path,
                line_start=start,
                line_end=end,
                content=content,
                chunk_type=ChunkType.MODULE,
                name=name,
            )

        return None

    def _is_inner_doc_comment(self, node: Node) -> bool:
        """Check if a line_comment is a //! inner doc comment."""
        return any(child.type == "inner_doc_comment_marker" for child in node.children)

    def _walk_items(
        self,
        parent: Node,
        file_path: str,
        lines: list[str],
        chunks: list[Chunk],
    ) -> None:
        """Walk top-level items and extract chunks."""
        children = list(parent.children)
        i = 0
        while i < len(children):
            child = children[i]

            # Find the start of attributes/doc comments preceding an item
            item_start_idx = self._find_item_start(children, i)

            match child.type:
                case "function_item":
                    start_node = children[item_start_idx] if item_start_idx < i else child
                    chunk = self._extract_function(child, start_node, file_path, lines)
                    if chunk:
                        chunks.append(chunk)

                case "struct_item":
                    start_node = children[item_start_idx] if item_start_idx < i else child
                    chunk = self._extract_named_item(
                        child, start_node, file_path, lines, ChunkType.CLASS
                    )
                    if chunk:
                        chunks.append(chunk)

                case "enum_item":
                    start_node = children[item_start_idx] if item_start_idx < i else child
                    chunk = self._extract_named_item(
                        child, start_node, file_path, lines, ChunkType.CLASS
                    )
                    if chunk:
                        chunks.append(chunk)

                case "trait_item":
                    start_node = children[item_start_idx] if item_start_idx < i else child
                    chunk = self._extract_named_item(
                        child, start_node, file_path, lines, ChunkType.CLASS
                    )
                    if chunk:
                        chunks.append(chunk)
                    # Extract default method implementations inside the trait
                    self._extract_trait_methods(child, file_path, lines, chunks)

                case "impl_item":
                    start_node = children[item_start_idx] if item_start_idx < i else child
                    self._extract_impl(child, start_node, file_path, lines, chunks)

            i += 1

    def _find_item_start(self, children: list[Node], current_idx: int) -> int:
        """Find the index where attributes/doc comments start before this item.

        Walks backward from current_idx to find preceding attribute_item
        and outer doc comment nodes.
        """
        start = current_idx
        j = current_idx - 1
        while j >= 0:
            prev = children[j]
            is_attr = prev.type == "attribute_item"
            is_doc = prev.type == "line_comment" and self._is_outer_doc_comment(prev)
            if is_attr or is_doc:
                start = j
            else:
                break
            j -= 1
        return start

    def _is_outer_doc_comment(self, node: Node) -> bool:
        """Check if a line_comment is a /// outer doc comment."""
        return any(child.type == "outer_doc_comment_marker" for child in node.children)

    def _extract_function(
        self,
        func_node: Node,
        start_node: Node,
        file_path: str,
        lines: list[str],
    ) -> Chunk | None:
        """Extract a function as FUNCTION chunk."""
        name = self._get_name(func_node)
        if not name:
            return None

        # Build a synthetic span from start_node to func_node end
        start_line = start_node.start_point[0] + 1
        end_line = func_node.end_point[0] + 1
        content = "\n".join(lines[start_line - 1 : end_line])

        return Chunk(
            file_path=file_path,
            line_start=start_line,
            line_end=end_line,
            content=content,
            chunk_type=ChunkType.FUNCTION,
            name=name,
        )

    def _extract_named_item(
        self,
        item_node: Node,
        start_node: Node,
        file_path: str,
        lines: list[str],
        chunk_type: ChunkType,
    ) -> Chunk | None:
        """Extract a named item (struct, enum, trait) as a chunk."""
        name = self._get_type_name(item_node)
        if not name:
            return None

        start_line = start_node.start_point[0] + 1
        end_line = item_node.end_point[0] + 1
        content = "\n".join(lines[start_line - 1 : end_line])

        return Chunk(
            file_path=file_path,
            line_start=start_line,
            line_end=end_line,
            content=content,
            chunk_type=chunk_type,
            name=name,
        )

    def _extract_impl(
        self,
        impl_node: Node,
        start_node: Node,
        file_path: str,
        lines: list[str],
        chunks: list[Chunk],
    ) -> None:
        """Extract impl block as CLASS chunk, and its methods as METHOD chunks."""
        impl_name = self._get_impl_name(impl_node)
        if not impl_name:
            return

        # The impl block itself as a CLASS chunk
        start_line = start_node.start_point[0] + 1
        end_line = impl_node.end_point[0] + 1
        content = "\n".join(lines[start_line - 1 : end_line])

        chunks.append(
            Chunk(
                file_path=file_path,
                line_start=start_line,
                line_end=end_line,
                content=content,
                chunk_type=ChunkType.CLASS,
                name=impl_name,
            )
        )

        # Extract methods inside the impl
        decl_list = impl_node.child_by_field_name("body")
        if not decl_list:
            return

        self._extract_methods_from_body(decl_list, file_path, lines, chunks)

    def _extract_trait_methods(
        self,
        trait_node: Node,
        file_path: str,
        lines: list[str],
        chunks: list[Chunk],
    ) -> None:
        """Extract default method implementations from a trait body."""
        body = trait_node.child_by_field_name("body")
        if not body:
            return

        self._extract_methods_from_body(body, file_path, lines, chunks)

    def _extract_methods_from_body(
        self,
        body: Node,
        file_path: str,
        lines: list[str],
        chunks: list[Chunk],
    ) -> None:
        """Extract function_item nodes from a declaration_list as METHOD chunks."""
        body_children = list(body.children)
        for idx, child in enumerate(body_children):
            if child.type != "function_item":
                continue

            name = self._get_name(child)
            if not name:
                continue

            # Find preceding attributes/doc comments
            item_start_idx = self._find_item_start(body_children, idx)
            start_node = body_children[item_start_idx] if item_start_idx < idx else child

            start_line = start_node.start_point[0] + 1
            end_line = child.end_point[0] + 1
            content = "\n".join(lines[start_line - 1 : end_line])

            chunks.append(
                Chunk(
                    file_path=file_path,
                    line_start=start_line,
                    line_end=end_line,
                    content=content,
                    chunk_type=ChunkType.METHOD,
                    name=name,
                )
            )

    def _get_name(self, node: Node) -> str | None:
        """Get the name from a function_item node."""
        name_node = node.child_by_field_name("name")
        if not name_node or name_node.text is None:
            return None
        return name_node.text.decode()

    def _get_type_name(self, node: Node) -> str | None:
        """Get the type_identifier name from a struct/enum/trait node."""
        for child in node.children:
            if child.type == "type_identifier" and child.text is not None:
                return child.text.decode()
        return None

    def _get_impl_name(self, impl_node: Node) -> str | None:
        """Get the name for an impl block.

        For `impl Type`, returns "Type".
        For `impl Trait for Type`, returns "Trait for Type".
        """
        type_ids = [
            c for c in impl_node.children if c.type == "type_identifier" and c.text is not None
        ]
        has_for = any(c.type == "for" for c in impl_node.children)

        if has_for and len(type_ids) >= 2:
            trait_name = type_ids[0].text.decode()  # type: ignore[union-attr]
            type_name = type_ids[1].text.decode()  # type: ignore[union-attr]
            return f"{trait_name} for {type_name}"

        if type_ids:
            return type_ids[0].text.decode()  # type: ignore[union-attr]

        return None
