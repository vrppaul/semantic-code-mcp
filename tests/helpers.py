"""Helpers shared by tests."""

import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from semantic_code_mcp.domain.models import Chunk, ChunkType


def run_git(repository: Path, *arguments: str) -> None:
    """Run a git command in a repository with a fixed test identity."""
    subprocess.run(  # nosec B603, B607
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
    )


def make_repository(root: Path) -> Path:
    """Create an empty git repository, creating the directory if needed."""
    root.mkdir(parents=True, exist_ok=True)
    run_git(root, "init", "--quiet")
    return root


def commit_all(repository: Path) -> None:
    """Stage and commit everything in a repository."""
    run_git(repository, "add", "--all")
    run_git(repository, "commit", "--quiet", "--message", "snapshot")


class ChunkSummary(BaseModel):
    """What a chunker test compares of a chunk: its type and its name."""

    model_config = ConfigDict(frozen=True)

    chunk_type: ChunkType
    name: str


class LineRange(BaseModel):
    """The lines a chunk covers, counted from one."""

    model_config = ConfigDict(frozen=True)

    line_start: int
    line_end: int


def summarize(chunks: list[Chunk]) -> list[ChunkSummary]:
    """Each chunk's type and name, in order."""
    return [ChunkSummary(chunk_type=chunk.chunk_type, name=chunk.name) for chunk in chunks]


def line_ranges(chunks: list[Chunk]) -> list[LineRange]:
    """Each chunk's line range, in order."""
    return [LineRange(line_start=chunk.line_start, line_end=chunk.line_end) for chunk in chunks]
