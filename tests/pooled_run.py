"""A first index in a pool of processes whose chunker writes to standard output, run in a process
of its own, so a test reads what reached that process's standard output and error:

    python -m tests.pooled_run <project> <cache directory>

It logs as the server does, to standard error.
"""

import asyncio
import sys
from pathlib import Path

from semantic_code_mcp.app import build_chunker
from semantic_code_mcp.config import Settings
from semantic_code_mcp.logging import configure_logging
from tests.fakes import FakeIndexStore, StdoutWritingChunker
from tests.helpers import indexing_service
from tests.indexing import first_index
from tests.stores import NO_CANDIDATES


def main() -> None:
    project, cache_directory = (Path(argument) for argument in sys.argv[1:])
    settings = Settings(cache_dir=cache_directory)
    configure_logging(debug=settings.debug)
    indexing = indexing_service(
        FakeIndexStore(NO_CANDIDATES), settings, StdoutWritingChunker(build_chunker())
    )
    asyncio.run(first_index(indexing, project))


if __name__ == "__main__":
    main()
