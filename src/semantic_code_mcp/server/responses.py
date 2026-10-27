"""What the tools answer: the structured content an agent reads."""

from pydantic import BaseModel, ConfigDict, Field

from semantic_code_mcp.domain.models import ChunkKind

# What an agent reads of a hit's similarity in the tool's output schema
SIMILARITY_DESCRIPTION = (
    "How close the chunk's meaning is to the query's alone, a cosine similarity from -1 to 1."
    " Results come best first by meaning and words together, so a hit its words found can"
    " show a low similarity."
)
# What an agent reads of the root an index tool answers for
PROJECT_ROOT_DESCRIPTION = (
    "The root of the project whose index this is: a directory inside a git repository is"
    " indexed, and reported, with its whole repository."
)


class SearchResultView(BaseModel):
    """One hit as the agent sees it; `content` is cut to a line limit, `truncated` says so."""

    model_config = ConfigDict(frozen=True)

    # Absolute, for the agent to read the file at
    file_path: str
    line_start: int
    line_end: int
    name: str
    # Its values are in the tool's output schema, where a filter's will be
    kind: ChunkKind
    # The qualified name of what the chunk is defined in: a class, an impl, a heading
    parent: str | None
    content: str
    similarity: float = Field(description=SIMILARITY_DESCRIPTION)
    truncated: bool


class SearchTimingsView(BaseModel):
    """Where the time of a search went, in milliseconds."""

    model_config = ConfigDict(frozen=True)

    embedding_ms: float
    search_ms: float
    total_ms: float
    indexing_ms: float | None


class SearchStatsView(BaseModel):
    """What a search kept and dropped."""

    model_config = ConfigDict(frozen=True)

    results_count: int
    filtered_out: int
    unique_files: int
    tokens_estimate: int


class IndexingSummaryView(BaseModel):
    """What the search indexed before it searched."""

    model_config = ConfigDict(frozen=True)

    files_indexed: int
    chunks_indexed: int


class SearchDebugView(BaseModel):
    """Timings and counts of a search, sent only when asked for."""

    model_config = ConfigDict(frozen=True)

    timings: SearchTimingsView
    stats: SearchStatsView
    indexing: IndexingSummaryView | None


class SearchCodeResponse(BaseModel):
    """The answer of `search_code`."""

    model_config = ConfigDict(frozen=True)

    results: list[SearchResultView]
    debug: SearchDebugView | None


class IndexCodebaseResponse(BaseModel):
    """The answer of `index_codebase`."""

    model_config = ConfigDict(frozen=True)

    project_root: str = Field(description=PROJECT_ROOT_DESCRIPTION)
    files_indexed: int
    chunks_indexed: int
    files_deleted: int
    duration_seconds: float


class IndexManifestView(BaseModel):
    """What built an index."""

    model_config = ConfigDict(frozen=True)

    model_name: str
    # None where nothing pins the model's weights, as for a local path
    model_revision: str | None
    # How the embedder read the model: another reading keeps an index of its own
    embedder_version: int
    chunker_version: int
    schema_version: int


class IndexStatusResponse(BaseModel):
    """The answer of `index_status`."""

    model_config = ConfigDict(frozen=True)

    project_root: str = Field(description=PROJECT_ROOT_DESCRIPTION)
    is_indexed: bool
    last_updated: str | None
    files_count: int
    chunks_count: int
    stale_files_count: int
    manifest: IndexManifestView | None
