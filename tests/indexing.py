"""What the indexing tests share: projects of files of small functions, an edit of a file, a
first index of a project, and the identity the chunkers build an index of."""

import os
from pathlib import Path

from semantic_code_mcp.chunkers.composite import CHUNKER_VERSION
from semantic_code_mcp.domain.models import IndexIdentity
from semantic_code_mcp.services.indexing import WRITE_SLICE_SIZE, IndexingService
from semantic_code_mcp.services.progress import ProgressReporter

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


async def first_index(indexing: IndexingService, project: Path) -> None:
    """A first index of the project, in several writes."""
    await indexing.index(project, force=False, progress=NO_PROGRESS)
