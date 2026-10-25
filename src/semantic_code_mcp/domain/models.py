"""Domain models: chunks, search results, index state and the reports of each operation."""

from datetime import datetime
from enum import StrEnum, auto

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

FIRST_LINE = 1
LOWEST_SCORE = 0.0
HIGHEST_SCORE = 1.0


class ChunkType(StrEnum):
    """Type of code chunk."""

    module = auto()
    function = auto()
    klass = "class"
    method = auto()
    section = auto()


class Chunk(BaseModel):
    """A chunk of code extracted from a file."""

    model_config = ConfigDict(frozen=True)

    file_path: str
    line_start: int
    line_end: int
    content: str
    chunk_type: ChunkType
    name: str

    @field_validator("line_start")
    @classmethod
    def line_start_must_be_positive(cls, line_start: int) -> int:
        if line_start < FIRST_LINE:
            message = f"line_start must be >= {FIRST_LINE}"
            raise ValueError(message)
        return line_start

    @model_validator(mode="after")
    def line_end_not_before_line_start(self) -> Chunk:
        if self.line_end < self.line_start:
            message = "line_end must be >= line_start"
            raise ValueError(message)
        return self


class ChunkLocation(BaseModel):
    """Where a chunk starts: the key under which two searches' hits of one chunk meet."""

    model_config = ConfigDict(frozen=True)

    file_path: str
    line_start: int


class SearchResult(BaseModel):
    """A chunk with a similarity score from search."""

    model_config = ConfigDict(frozen=True)

    file_path: str
    line_start: int
    line_end: int
    content: str
    chunk_type: ChunkType
    name: str
    score: float

    @field_validator("score")
    @classmethod
    def score_must_be_in_range(cls, score: float) -> float:
        if not LOWEST_SCORE <= score <= HIGHEST_SCORE:
            message = f"score must be between {LOWEST_SCORE} and {HIGHEST_SCORE}"
            raise ValueError(message)
        return score

    @property
    def location(self) -> ChunkLocation:
        return ChunkLocation(file_path=self.file_path, line_start=self.line_start)

    def with_score(self, score: float) -> SearchResult:
        """The same hit with another score."""
        return SearchResult(
            file_path=self.file_path,
            line_start=self.line_start,
            line_end=self.line_end,
            content=self.content,
            chunk_type=self.chunk_type,
            name=self.name,
            score=score,
        )


class IndexStatus(BaseModel):
    """Status of the index for a codebase."""

    model_config = ConfigDict(frozen=True)

    is_indexed: bool
    last_updated: datetime | None
    files_count: int
    chunks_count: int
    stale_files: list[str]


class StoreStatistics(BaseModel):
    """What a vector store holds: the files it has chunks of, and how many chunks."""

    model_config = ConfigDict(frozen=True)

    indexed_files: list[str]
    chunks_count: int


class FileChanges(BaseModel):
    """Result of comparing current files with cached mtimes."""

    model_config = ConfigDict(frozen=True)

    new: list[str]
    modified: list[str]
    deleted: list[str]

    @property
    def stale_files(self) -> list[str]:
        """Files that need re-indexing (new + modified)."""
        return self.new + self.modified


class ScanPlan(BaseModel):
    """What an index run has to do: files to (re)index and files to drop."""

    model_config = ConfigDict(frozen=True)

    files_to_index: list[str]
    files_to_delete: list[str]

    @property
    def has_work(self) -> bool:
        return bool(self.files_to_index or self.files_to_delete)


class IndexResult(BaseModel):
    """Result of an indexing operation."""

    model_config = ConfigDict(frozen=True)

    files_indexed: int
    chunks_indexed: int
    files_deleted: int
    duration_seconds: float


class SearchTimings(BaseModel):
    """Where the time of one search went, in milliseconds."""

    model_config = ConfigDict(frozen=True)

    embedding_ms: float
    search_ms: float
    total_ms: float


class SearchReport(BaseModel):
    """What one search found, how much it dropped, and what it cost."""

    model_config = ConfigDict(frozen=True)

    results: list[SearchResult]
    filtered_count: int
    timings: SearchTimings
    # Set when the search had to index stale files first
    index_result: IndexResult | None
