"""Helpers shared by tests."""

import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from semantic_code_mcp.app import open_index_store
from semantic_code_mcp.config import Settings
from semantic_code_mcp.discovery import SourceDiscovery
from semantic_code_mcp.domain.interfaces import Chunker, Embedder, IndexStore
from semantic_code_mcp.domain.models import Chunk, ChunkType
from semantic_code_mcp.services.indexing import IndexingService
from semantic_code_mcp.services.projects import ProjectRegistry


def run_git(repository: Path, *arguments: str) -> None:
    """Run a git command in a repository with a fixed test identity."""
    subprocess.run(  # nosec B603, B607
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
    )


def git_output(repository: Path, *arguments: str) -> str:
    """What a git command prints in a repository."""
    return subprocess.run(  # nosec B603, B607
        ["git", *arguments], cwd=repository, check=True, capture_output=True, text=True
    ).stdout


def make_repository(root: Path) -> Path:
    """Create an empty git repository, creating the directory if needed."""
    root.mkdir(parents=True, exist_ok=True)
    run_git(root, "init", "--quiet")
    return root


def commit_all(repository: Path) -> None:
    """Stage and commit everything in a repository."""
    run_git(repository, "add", "--all")
    run_git(repository, "commit", "--quiet", "--message", "snapshot")


def discovery_for(settings: Settings, chunker: Chunker) -> SourceDiscovery:
    """The real discovery, finding the files a chunker reads."""
    return SourceDiscovery(
        extensions=chunker.supported_extensions,
        ignore_patterns=settings.ignore_patterns,
        use_gitignore=settings.use_gitignore,
        git_timeout_seconds=settings.git_timeout_seconds,
    )


def indexing_service(
    store: IndexStore, embedder: Embedder, settings: Settings, chunker: Chunker
) -> IndexingService:
    """The indexing of a project over the given parts, with the real discovery."""
    return IndexingService(
        discovery=discovery_for(settings, chunker),
        chunker=chunker,
        embedder=embedder,
        store=store,
    )


def registry_of(settings: Settings, embedder: Embedder, chunker: Chunker) -> ProjectRegistry:
    """The registry of a server whose model and chunker are the given ones, its indexes in
    the real store."""
    return ProjectRegistry(
        settings=settings,
        embedder=embedder,
        chunker=chunker,
        discovery=discovery_for(settings, chunker),
        open_store=open_index_store,
    )


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
