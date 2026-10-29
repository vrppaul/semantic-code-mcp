"""A first index on a system whose forkserver cannot start, in a process of its own: the resource
tracker started first, as a server's earlier pool would have, then the interpreter Python
starts its processes with pointed at none. Prints the chunks the run wrote.

    python -m tests.unstartable_pool_run <project> <cache directory>

It logs as the server does, to standard error.
"""

import asyncio
import multiprocessing
import sys
from pathlib import Path

from semantic_code_mcp.app import build_chunker
from semantic_code_mcp.config import Settings
from semantic_code_mcp.logging import configure_logging
from tests.fakes import FakeIndexStore
from tests.helpers import indexing_service
from tests.indexing import NO_PROGRESS
from tests.stores import NO_CANDIDATES

# An interpreter no process can be started with
MISSING_INTERPRETER = "/nonexistent/python"


def main() -> None:
    project, cache_directory = (Path(argument) for argument in sys.argv[1:])
    settings = Settings(cache_dir=cache_directory)
    configure_logging(debug=settings.debug)
    # A lock registers with the resource tracker, which starts it
    multiprocessing.Lock()
    multiprocessing.set_executable(MISSING_INTERPRETER)
    indexing = indexing_service(FakeIndexStore(NO_CANDIDATES), settings, build_chunker())
    report = asyncio.run(indexing.index(project, force=False, progress=NO_PROGRESS))
    sys.stdout.write(f"{report.chunks_indexed}\n")


if __name__ == "__main__":
    main()
