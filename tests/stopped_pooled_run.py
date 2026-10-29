"""A first index in a pool of processes stopped in its first write, in a process of its own for a
test to kill: it says `writing` on stdout once the write has begun, and never goes on.

    python -m tests.stopped_pooled_run <project> <cache directory>

It logs as the server does, to standard error.
"""

import asyncio
import sys
from pathlib import Path

from semantic_code_mcp.app import build_chunker
from semantic_code_mcp.config import Settings
from semantic_code_mcp.logging import configure_logging
from tests.fakes import FakeIndexStore, StoreStoppedInItsWrite
from tests.helpers import indexing_service
from tests.indexing import first_index
from tests.stores import NO_CANDIDATES


def main() -> None:
    project, cache_directory = (Path(argument) for argument in sys.argv[1:])
    settings = Settings(cache_dir=cache_directory)
    configure_logging(debug=settings.debug)
    store = StoreStoppedInItsWrite(FakeIndexStore(NO_CANDIDATES))
    asyncio.run(first_index(indexing_service(store, settings, build_chunker()), project))


if __name__ == "__main__":
    main()
