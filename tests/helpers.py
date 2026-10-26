"""Helpers shared by tests."""

import subprocess
from pathlib import Path

import lancedb
from lance.fragment import LanceFragment
from pydantic import BaseModel, ConfigDict

from semantic_code_mcp.app import open_index_store
from semantic_code_mcp.config import Settings
from semantic_code_mcp.discovery import SourceDiscovery
from semantic_code_mcp.domain.interfaces import Chunker, Embedder, IndexStore
from semantic_code_mcp.domain.models import Chunk, ChunkKind
from semantic_code_mcp.services.indexing import IndexingService
from semantic_code_mcp.services.projects import ProjectRegistry
from semantic_code_mcp.storage.lancedb import CHUNKS_TABLE, LanceDBIndexStore

# Where LanceDB keeps each version of every index of a table, a directory each: while the
# chunks table has no index but the full-text one, the newest is the full-text index's
TABLE_INDEX_VERSIONS = "_indices"


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


def chunks_outside_the_full_text_index(store: LanceDBIndexStore) -> int:
    """How many chunks of a store its full-text index does not hold yet."""
    chunks_table = lancedb.connect(str(store.directory)).open_table(CHUNKS_TABLE)
    full_text_index = chunks_table.list_indices()[0]
    return chunks_table.index_stats(full_text_index.name).num_unindexed_rows


def chunks_fragments(store: LanceDBIndexStore) -> list[LanceFragment]:
    """The fragments the latest version of a store's chunks table reads, in order."""
    chunks_table = lancedb.connect(str(store.directory)).open_table(CHUNKS_TABLE)
    return chunks_table.to_lance().get_fragments()


def newest_full_text_index_files(store: LanceDBIndexStore) -> int:
    """How many files the newest version of a store's full-text index holds: each part an
    upkeep folds in adds some, and a rebuild starts again from one part."""
    versions = store.directory / f"{CHUNKS_TABLE}.lance" / TABLE_INDEX_VERSIONS
    newest = max(versions.iterdir(), key=lambda version: version.stat().st_mtime_ns)
    return len(list(newest.rglob("*")))


def discovery_for(settings: Settings, chunker: Chunker) -> SourceDiscovery:
    """The real discovery, finding the files a chunker reads."""
    return SourceDiscovery(
        extensions=chunker.supported_extensions,
        ignore_patterns=settings.ignore_patterns,
        use_gitignore=settings.use_gitignore,
        git_timeout_seconds=settings.git_timeout_seconds,
        home_directory=Path.home(),
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

    kind: ChunkKind
    name: str


class LineRange(BaseModel):
    """The lines a chunk covers, counted from one."""

    model_config = ConfigDict(frozen=True)

    line_start: int
    line_end: int


class ChunkOwner(BaseModel):
    """What a chunker test compares of where a chunk is defined: its name and its parent."""

    model_config = ConfigDict(frozen=True)

    name: str
    parent: str | None


def owners_of(chunks: list[Chunk]) -> list[ChunkOwner]:
    """Each chunk's name and the qualified name of what it is defined in, in order."""
    return [ChunkOwner(name=chunk.name, parent=chunk.parent) for chunk in chunks]


def summarize(chunks: list[Chunk]) -> list[ChunkSummary]:
    """Each chunk's type and name, in order."""
    return [ChunkSummary(kind=chunk.kind, name=chunk.name) for chunk in chunks]


def function_named(name: str) -> ChunkSummary:
    return ChunkSummary(kind=ChunkKind.function, name=name)


def method_named(name: str) -> ChunkSummary:
    return ChunkSummary(kind=ChunkKind.method, name=name)


def class_named(name: str) -> ChunkSummary:
    """A class's summary, as Rust's structs, enums, traits and impls are classes too."""
    return ChunkSummary(kind=ChunkKind.klass, name=name)


def module_named(name: str) -> ChunkSummary:
    return ChunkSummary(kind=ChunkKind.module, name=name)


def section_named(name: str) -> ChunkSummary:
    return ChunkSummary(kind=ChunkKind.section, name=name)


def line_ranges(chunks: list[Chunk]) -> list[LineRange]:
    """Each chunk's line range, in order."""
    return [LineRange(line_start=chunk.line_start, line_end=chunk.line_end) for chunk in chunks]
