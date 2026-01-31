"""Domain models for chunks, search results, and index status."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, field_validator, model_validator


class ChunkType(str, Enum):
    """Type of code chunk."""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"


class Chunk(BaseModel):
    """A chunk of code extracted from a file."""

    file_path: str
    line_start: int
    line_end: int
    content: str
    chunk_type: ChunkType
    name: str

    @field_validator("line_start")
    @classmethod
    def line_start_must_be_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("line_start must be >= 1")
        return v

    @model_validator(mode="after")
    def line_end_gte_line_start(self) -> Chunk:
        if self.line_end < self.line_start:
            raise ValueError("line_end must be >= line_start")
        return self


class ChunkWithEmbedding(BaseModel):
    """A chunk paired with its embedding vector for storage."""

    chunk: Chunk
    embedding: list[float]


class SearchResult(Chunk):
    """A chunk with a similarity score from search."""

    score: float

    @field_validator("score")
    @classmethod
    def score_must_be_in_range(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("score must be between 0 and 1")
        return v


class IndexStatus(BaseModel):
    """Status of the index for a codebase."""

    is_indexed: bool
    last_updated: datetime | None
    files_count: int
    chunks_count: int
    stale_files: list[str]


class FileChanges(BaseModel):
    """Result of comparing current files with cached mtimes."""

    new: list[str]
    modified: list[str]
    deleted: list[str]

    @property
    def has_changes(self) -> bool:
        return bool(self.new or self.modified or self.deleted)

    @property
    def stale_files(self) -> list[str]:
        """Files that need re-indexing (new + modified)."""
        return self.new + self.modified


class ScanPlan(BaseModel):
    """Plan for what needs indexing, produced by detect_changes()."""

    files_to_index: list[str]
    files_to_delete: list[str]
    all_files: list[str]

    @property
    def has_work(self) -> bool:
        return bool(self.files_to_index or self.files_to_delete)


class IndexResult(BaseModel):
    """Result of an indexing operation."""

    files_indexed: int
    chunks_indexed: int
    files_deleted: int
    duration_seconds: float = 0.0
