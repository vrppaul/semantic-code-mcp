"""Composite chunker — dispatches to language-specific chunkers by extension."""

from pathlib import Path
from typing import override

import structlog

from semantic_code_mcp.chunkers.base import BaseTreeSitterChunker
from semantic_code_mcp.domain.interfaces import Chunker
from semantic_code_mcp.domain.models import Chunk

log = structlog.get_logger()

# The version of the chunks the chunkers make: bumped by any change to what a file
# gives, so an index built by an older version is built again
CHUNKER_VERSION = 1


class CompositeChunker(Chunker):
    """Chunker composed of language-specific chunkers, dispatching by file extension."""

    def __init__(self, chunkers: list[BaseTreeSitterChunker]) -> None:
        """Initialize with a list of language-specific chunkers.

        Args:
            chunkers: List of chunkers, each declaring supported extensions.

        Raises:
            ValueError: If two chunkers claim the same extension.
        """
        self._chunker_by_extension: dict[str, BaseTreeSitterChunker] = {}
        for chunker in chunkers:
            for extension in chunker.extensions:
                registered = self._chunker_by_extension.get(extension)
                if registered is not None:
                    message = (
                        f"Extension {extension!r} already registered by "
                        f"{type(registered).__name__}, cannot add {type(chunker).__name__}"
                    )
                    raise ValueError(message)
                self._chunker_by_extension[extension] = chunker

    @override
    def chunk_file(self, file_path: str) -> list[Chunk]:
        """Extract chunks from a source file, dispatching by extension.

        Args:
            file_path: Path to the source file.

        Returns:
            List of Chunk objects, or empty list for unsupported extensions.
        """
        suffix = Path(file_path).suffix
        chunker = self._chunker_by_extension.get(suffix)
        if chunker is None:
            log.debug("unsupported_extension", file_path=file_path, extension=suffix)
            return []

        return chunker.chunk_file(file_path)

    @property
    @override
    def version(self) -> int:
        return CHUNKER_VERSION

    @property
    @override
    def supported_extensions(self) -> list[str]:
        """All file extensions handled by this composite."""
        return sorted(self._chunker_by_extension)
