"""What the tools answer: the structured content an agent reads."""

from pydantic import BaseModel, ConfigDict


class SearchResultView(BaseModel):
    """One hit as the agent sees it; `content` is cut to a line limit, `truncated` says so."""

    model_config = ConfigDict(frozen=True)

    file_path: str
    line_start: int
    line_end: int
    name: str
    chunk_type: str
    content: str
    score: float
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
    rebuild_reason: str | None


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
    """The answer of `index_codebase`; `rebuild_reason` says why it indexed everything."""

    model_config = ConfigDict(frozen=True)

    files_indexed: int
    chunks_indexed: int
    files_deleted: int
    duration_seconds: float
    rebuild_reason: str | None


class IndexManifestView(BaseModel):
    """What built an index."""

    model_config = ConfigDict(frozen=True)

    model_name: str
    chunker_version: int
    schema_version: int


class IndexStatusResponse(BaseModel):
    """The answer of `index_status`."""

    model_config = ConfigDict(frozen=True)

    is_indexed: bool
    last_updated: str | None
    files_count: int
    chunks_count: int
    stale_files_count: int
    manifest: IndexManifestView | None
    # Set when the next search or index run builds the index again whole
    rebuild_reason: str | None
