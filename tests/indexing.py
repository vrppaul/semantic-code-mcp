"""What the indexing tests share: projects of files of small functions or of much source, an
edit of a file, a first index of a project, and the identity the chunkers build an index of."""

import os
from pathlib import Path

from semantic_code_mcp.chunkers.composite import CHUNKER_VERSION
from semantic_code_mcp.domain.models import IndexIdentity
from semantic_code_mcp.services.chunking import (
    BATCH_FILES,
    BATCHES_IN_FLIGHT_PER_WORKER,
    POOL_SOURCE_BYTES,
)
from semantic_code_mcp.services.indexing import WRITE_SLICE_SIZE, IndexingService
from semantic_code_mcp.services.progress import ProgressReporter
from tests.fakes import FakeIndexStore
from tests.helpers import POOL_WORKERS

NO_PROGRESS = ProgressReporter(None)
# An mtime well after any file the test just wrote
LATER_MTIME_SECONDS = 4_000_000_000
# A function added to a module of a project
ADDED_FUNCTION = "\n\ndef added():\n    return 1\n"
# Files whose functions fill one write with the first two and a second with the third
FILES_OF_FUNCTIONS = 3
FUNCTIONS_PER_FILE = WRITE_SLICE_SIZE * 3 // 4
# The names of those files, and the first of them
MODULE_NAME_FORMAT = "module_{file_number}.py"
FIRST_MODULE = MODULE_NAME_FORMAT.format(file_number=0)
# What the chunkers in these tests build an index from
IDENTITY = IndexIdentity(chunker_version=CHUNKER_VERSION)
# Batches of files more than a pool of the tests' workers hands out at once, so a run hands
# them out as others come back
POOLED_BATCHES = POOL_WORKERS * BATCHES_IN_FLIGHT_PER_WORKER + 2
POOLED_FILES = BATCH_FILES * POOLED_BATCHES
# Modules of functions holding long strings, enough source in all for a run to pay a pool's
# start, and enough functions in each that a run writes its first slice before its pool
# chunks its last batch
LARGE_MODULE_BYTES = POOL_SOURCE_BYTES // POOLED_FILES + 1
FUNCTIONS_PER_LARGE_MODULE = 8
PADDED_FUNCTION_FORMAT = 'def function_{file_number}_{number}():\n    return "{padding}"\n\n\n'
PADDING_CHARACTER = "x"


def touch_later(path: Path) -> None:
    """An edit: the file's mtime moves on."""
    os.utime(path, (LATER_MTIME_SECONDS, LATER_MTIME_SECONDS))


def functions_in_files(root: Path, files: int, functions: int) -> Path:
    """A project of several files, each holding a number of small functions, one chunk each."""
    project = root / "functions"
    project.mkdir()
    for file_number in range(files):
        (project / MODULE_NAME_FORMAT.format(file_number=file_number)).write_text(
            "".join(
                f"def function_{file_number}_{number}():\n    return {number}\n\n\n"
                for number in range(functions)
            )
        )
    return project


def large_modules(root: Path, files: int) -> Path:
    """A project of modules of more than LARGE_MODULE_BYTES each, a chunk for each of their
    FUNCTIONS_PER_LARGE_MODULE functions."""
    project = root / "large"
    project.mkdir()
    padding = PADDING_CHARACTER * (LARGE_MODULE_BYTES // FUNCTIONS_PER_LARGE_MODULE)
    for file_number in range(files):
        (project / MODULE_NAME_FORMAT.format(file_number=file_number)).write_text(
            "".join(
                PADDED_FUNCTION_FORMAT.format(
                    file_number=file_number, number=number, padding=padding
                )
                for number in range(FUNCTIONS_PER_LARGE_MODULE)
            )
        )
    return project


def note_writes(store: FakeIndexStore, writes_seen: list[int]) -> None:
    """Note how many writes have reached the store."""
    writes_seen.append(len(store.updates))


async def first_index(indexing: IndexingService, project: Path) -> None:
    """A first index of the project, in several writes."""
    await indexing.index(project, force=False, progress=NO_PROGRESS)
