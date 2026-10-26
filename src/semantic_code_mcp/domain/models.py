"""Domain models: chunks, search results, index state and the reports of each operation."""

from datetime import datetime
from enum import StrEnum, auto
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

FIRST_LINE = 1
# The range of a cosine similarity: opposite meanings to the same
LOWEST_SIMILARITY = -1.0
HIGHEST_SIMILARITY = 1.0


class Language(StrEnum):
    """A language the chunkers read."""

    python = auto()
    rust = auto()
    markdown = auto()


class ChunkKind(StrEnum):
    """What a chunk is. Rust's structs, enums, traits and impls are classes."""

    module = auto()
    function = auto()
    klass = "class"
    method = auto()
    section = auto()


def relative_path(path: Path) -> Path:
    """A path the index keeps: from the project root, so a project moved keeps its index."""
    if path.is_absolute():
        message = f"a path from the project root was expected, not {path}"
        raise ValueError(message)
    return path


class Chunk(BaseModel):
    """A part of a file the index embeds and finds: a definition, a module's docstring or
    a section of a document."""

    model_config = ConfigDict(frozen=True)

    # Stays while the chunk's file and text stay, whatever moves around it
    chunk_id: str
    # The file's path from the project root
    path: Path
    language: Language
    kind: ChunkKind
    name: str
    # The qualified name of what the chunk is defined in, `Stack` for `Stack.push`;
    # None at the top of its file
    parent: str | None
    line_start: int
    line_end: int
    content: str

    @field_validator("path")
    @classmethod
    def path_must_be_relative(cls, path: Path) -> Path:
        return relative_path(path)

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


def similarity_in_range(similarity: float) -> float:
    """A cosine similarity, which lies in [-1, 1]."""
    if not LOWEST_SIMILARITY <= similarity <= HIGHEST_SIMILARITY:
        message = f"similarity must be between {LOWEST_SIMILARITY} and {HIGHEST_SIMILARITY}"
        raise ValueError(message)
    return similarity


class Candidate(BaseModel):
    """A chunk one channel of a search found, and how close its meaning is to the query."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    # The cosine similarity of the chunk's stored vector and the query's
    similarity: float

    @field_validator("similarity")
    @classmethod
    def similarity_must_be_in_range(cls, similarity: float) -> float:
        return similarity_in_range(similarity)


class SearchCandidates(BaseModel):
    """What each channel of a search found, best first, read from one version of the
    index: the chunks nearest the query's vector, and those whose words match its words."""

    model_config = ConfigDict(frozen=True)

    by_vector: list[Candidate]
    by_keyword: list[Candidate]


class SearchHit(BaseModel):
    """A chunk a search returns: where each channel ranked it, and the score fusing those
    ranks, which orders the hits."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    similarity: float
    # Counted from one; None where the channel did not find the chunk
    vector_rank: int | None
    keyword_rank: int | None
    fused_score: float

    @field_validator("similarity")
    @classmethod
    def similarity_must_be_in_range(cls, similarity: float) -> float:
        return similarity_in_range(similarity)


class SourceFile(BaseModel):
    """A file discovery found, with what it was when it was found.

    The index records each file it indexed as the scan found it, never stat'ed again
    later: an edit or a deletion made while the file is chunked and embedded leaves
    the record behind, so the next run sees the file as changed or gone. A file that
    gave no chunk is recorded too, so it is not indexed again on every run.
    """

    model_config = ConfigDict(frozen=True)

    # From the project root
    path: Path
    mtime_ns: int
    size_bytes: int

    @field_validator("path")
    @classmethod
    def path_must_be_relative(cls, path: Path) -> Path:
        return relative_path(path)


class IndexDelta(BaseModel):
    """What an index run has to do: the files to index, and the paths gone from disk."""

    model_config = ConfigDict(frozen=True)

    to_index: list[SourceFile]
    to_delete: list[Path]

    @property
    def has_work(self) -> bool:
        return bool(self.to_index or self.to_delete)


class IndexIdentity(BaseModel):
    """What an index is built from: the model that embeds, by its name and the revision of
    its weights, and the version of the chunker that splits. A project keeps an index for
    each, so a session of another model, revision or version never builds the index of
    this one again, and newer weights under a model's name never meet its old vectors.

    The vectors' width is not part of it: the model fixes it, and the stored vector column
    records it, so nothing needs the model loaded to pick an index.
    """

    model_config = ConfigDict(frozen=True)

    model_name: str
    # None where nothing pins the weights (a local path, a hub model the server does not
    # know), which are then trusted to stay what they were
    model_revision: str | None
    chunker_version: int


class IndexManifest(BaseModel):
    """What built an index, in which layout of the store, and when it was last written."""

    model_config = ConfigDict(frozen=True)

    identity: IndexIdentity
    schema_version: int
    indexed_at: datetime


class IndexUpdate(BaseModel):
    """One write of an index run: whole files with their chunks, which take the place of
    what the index held of them, and the paths removed. A run writes a slice of its files
    at a time; the chunks' vectors travel beside it."""

    model_config = ConfigDict(frozen=True)

    # The files indexed, as the scan found them
    files: list[SourceFile]
    chunks: list[Chunk]
    removed_paths: list[Path]

    @model_validator(mode="after")
    def chunks_belong_to_the_files_written(self) -> IndexUpdate:
        written = {file.path for file in self.files}
        strays = sorted({chunk.path for chunk in self.chunks} - written)
        if strays:
            message = f"chunks of files the write does not hold: {strays}"
            raise ValueError(message)
        return self


class IndexCounts(BaseModel):
    """How much an index holds."""

    model_config = ConfigDict(frozen=True)

    files: int
    chunks: int


class IndexStatus(BaseModel):
    """How far a project's index is from its files."""

    model_config = ConfigDict(frozen=True)

    files_count: int
    chunks_count: int
    # Files the index no longer matches: new, modified or deleted since it was written
    stale_files_count: int
    manifest: IndexManifest | None

    @property
    def is_indexed(self) -> bool:
        return self.files_count > 0


class IndexReport(BaseModel):
    """What one index run did."""

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

    results: list[SearchHit]
    filtered_count: int
    timings: SearchTimings
    # Set when the search had to index stale files first
    index_report: IndexReport | None
