"""Helpers shared by tests."""

import subprocess
import sys
from functools import partial
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from semantic_code_mcp.app import files_opener, store_opener
from semantic_code_mcp.config import Settings
from semantic_code_mcp.discovery import SourceDiscovery
from semantic_code_mcp.domain.interfaces import Chunker, Embedder, IndexStore
from semantic_code_mcp.domain.models import Chunk, ChunkKind
from semantic_code_mcp.pools import open_chunking_pool
from semantic_code_mcp.services.chunking import Chunking, PoolOpener
from semantic_code_mcp.services.indexing import IndexingService
from semantic_code_mcp.services.projects import ProjectRegistry, vector_identity
from semantic_code_mcp.services.vectors import VectorService
from tests.fakes import FAKE_MODEL

# Claude Code faked for the agents harness, which no test may run for real
FAKE_CLAUDE = Path(__file__).parent / "data" / "fake_claude.py"
EXECUTABLE_MODE = 0o755
# The workers of a pool in these tests, given whatever the machine's cores, so a run of source
# enough for a pool makes one on any machine the tests run on; and none
POOL_WORKERS = 2
NO_POOL = 0


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
        home_directory=Path.home(),
    )


def indexing_service(
    store: IndexStore,
    settings: Settings,
    chunker: Chunker,
    *,
    workers: int = POOL_WORKERS,
    open_pool: PoolOpener | None = None,
) -> IndexingService:
    """The indexing of a project's words over the given parts, its files read as the settings
    ask by the real discovery, a run of source enough chunked in a pool of the workers given,
    the real one unless another opener is, its status counting the fake model's vectors."""
    return IndexingService(
        files=files_opener(settings, discovery_for(settings, chunker))(),
        chunking=Chunking(
            chunker,
            workers=workers,
            open_pool=open_pool or partial(open_chunking_pool, debug_logging=settings.debug),
        ),
        store=store,
        model=FAKE_MODEL,
    )


def vector_service(store: IndexStore, embedder: Embedder) -> VectorService:
    """What gives the chunks of a store the vectors of an embedder's model."""
    return VectorService(embedder=embedder, store=store, model=vector_identity(embedder))


def registry_of(settings: Settings, embedder: Embedder, chunker: Chunker) -> ProjectRegistry:
    """The registry of a server whose model and chunker are the given ones, its indexes in
    the store the settings pick, each project's files read as they ask."""
    discovery = discovery_for(settings, chunker)
    return ProjectRegistry(
        settings=settings,
        embedder=embedder,
        chunker=chunker,
        discovery=discovery,
        open_store=store_opener(settings),
        open_files=files_opener(settings, discovery),
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


def install_fake_claude(directory: Path) -> Path:
    """The fake Claude Code as `claude` in a directory of its own, run by this interpreter;
    the directory, to go first on PATH."""
    directory.mkdir(parents=True)
    claude = directory / "claude"
    claude.write_text(f"#!{sys.executable}\n{FAKE_CLAUDE.read_text()}")
    claude.chmod(EXECUTABLE_MODE)
    return directory
