"""Composite chunker — dispatches to language-specific chunkers by extension."""

from pathlib import Path

import structlog

from semantic_code_mcp.chunkers.base import BaseTreeSitterChunker
from semantic_code_mcp.models import Chunk

log = structlog.get_logger()


class CompositeChunker:
    """Chunker composed of language-specific chunkers, dispatching by file extension.

    Implements ChunkerProtocol.
    """

    def __init__(self, chunkers: list[BaseTreeSitterChunker]) -> None:
        """Initialize with a list of language-specific chunkers.

        Args:
            chunkers: List of chunkers, each declaring supported extensions.

        Raises:
            ValueError: If two chunkers claim the same extension.
        """
        self._extension_map: dict[str, BaseTreeSitterChunker] = {}
        for chunker in chunkers:
            for ext in chunker.extensions:
                if ext in self._extension_map:
                    existing = type(self._extension_map[ext]).__name__
                    new = type(chunker).__name__
                    msg = f"Extension {ext!r} already registered by {existing}, cannot add {new}"
                    raise ValueError(msg)
                self._extension_map[ext] = chunker

    def chunk_file(self, file_path: str) -> list[Chunk]:
        """Extract chunks from a source file, dispatching by extension.

        Args:
            file_path: Path to the source file.

        Returns:
            List of Chunk objects, or empty list for unsupported extensions.
        """
        suffix = Path(file_path).suffix
        chunker = self._extension_map.get(suffix)
        if chunker is None:
            log.debug("unsupported_extension", file_path=file_path, extension=suffix)
            return []

        return chunker.chunk_file(file_path)

    @property
    def supported_extensions(self) -> list[str]:
        """All file extensions handled by this composite."""
        return sorted(self._extension_map.keys())
