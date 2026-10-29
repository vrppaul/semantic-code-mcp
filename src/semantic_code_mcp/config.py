"""Settings, read once from `SEMANTIC_CODE_MCP_*` environment variables at start-up."""

import hashlib
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from semantic_code_mcp.domain.models import (
    HIGHEST_SIMILARITY,
    LOWEST_SIMILARITY,
    SearchMode,
    StoreKind,
)

# Never source. A name matches a path component at any depth; `name/**` matches only
# at the project root, so a source package called `build` or `dist` deeper down is indexed.
DEFAULT_IGNORE_PATTERNS = (
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".cache",
    "site-packages",
    ".semantic-code",
    "target/**",
    "dist/**",
    "build/**",
)
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "semantic-code-mcp"
# A project-local index lives in this directory of the project itself
LOCAL_INDEX_DIRECTORY = ".semantic-code"
# A global index directory is named by this many hex digits of a hash of the project path
INDEX_HASH_LENGTH = 16


class Settings(BaseSettings):
    """Application settings, from environment variables with the SEMANTIC_CODE_MCP_ prefix.

    No `.env` file is read: the server runs in the user's project, and its `.env` is
    not ours to read.
    """

    model_config = SettingsConfigDict(env_prefix="SEMANTIC_CODE_MCP_", extra="ignore")

    embedding_model: str = "all-MiniLM-L6-v2"

    cache_dir: Path = DEFAULT_CACHE_DIR
    local_index: bool = False
    # What keeps the indexes (ADR 009, D8); each store keeps indexes of its own
    index_store: StoreKind = StoreKind.sqlite
    # How every search ranks: by words alone, loading no model, unless a user asks for
    # meaning too (ADR 009, D2)
    search_mode: SearchMode = SearchMode.words

    debug: bool = False
    # Profile every request with pyinstrument, a dev dependency
    profile: bool = False

    # File discovery (see SourceDiscovery for the pattern rules)
    ignore_patterns: list[str] = Field(default_factory=lambda: list(DEFAULT_IGNORE_PATTERNS))
    # Ask git which files .gitignore excludes. False walks the file system and applies
    # ignore_patterns only — for repos whose .gitignore hides source on purpose.
    use_gitignore: bool = True
    # Git normally answers in milliseconds; past this, discovery walks the file system
    git_timeout_seconds: float = 10.0
    # Follow a project's files with a watcher, so a search reads what changed rather than
    # every file (ADR 010, O1); off, or where no watcher's barrier holds, each search scans
    watch_files: bool = True

    # How close in meaning to the query, a cosine similarity, a hit must be to stay: one no
    # keyword matched, and, never held to more, one whose words match the query's. Set on
    # the harness for the default model; another model scores on a scale of its own (ADR
    # 006, stage 6, holds the evidence)
    min_similarity: float = Field(default=0.35, ge=LOWEST_SIMILARITY, le=HIGHEST_SIMILARITY)
    min_similarity_when_words_match: float = Field(
        default=0.25, ge=LOWEST_SIMILARITY, le=HIGHEST_SIMILARITY
    )

    @model_validator(mode="after")
    def word_matches_held_to_no_more(self) -> Settings:
        if self.min_similarity_when_words_match > self.min_similarity:
            message = (
                "min_similarity_when_words_match must not exceed min_similarity: a hit"
                " whose words match the query's is never held to more than one whose do not"
            )
            raise ValueError(message)
        return self

    @property
    def profiles_dir(self) -> Path:
        """Where profiles of requests are written, outside any user's project."""
        return self.cache_dir / "profiles"


def get_index_path(settings: Settings, project_path: Path) -> Path:
    """The directory a project's index lives in.

    Args:
        settings: Application settings.
        project_path: Path to the project root.

    Returns:
        `.semantic-code/` in the project with `local_index`, else a directory of the
        cache named by a hash of the project's path; either way every spelling of one
        project gives the same directory.
    """
    project_root = project_path.resolve()
    if settings.local_index:
        return project_root / LOCAL_INDEX_DIRECTORY

    path_hash = hashlib.sha256(str(project_root).encode()).hexdigest()
    return settings.cache_dir / path_hash[:INDEX_HASH_LENGTH]
