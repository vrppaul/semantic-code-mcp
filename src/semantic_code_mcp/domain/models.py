"""Domain models: chunks, search results, index state and the reports of each operation."""

import os
from datetime import datetime
from enum import StrEnum, auto
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from semantic_code_mcp.domain.globs import read_glob

FIRST_LINE = 1
# The range of a cosine similarity: opposite meanings to the same
LOWEST_SIMILARITY = -1.0
HIGHEST_SIMILARITY = 1.0


class Language(StrEnum):
    """A language the chunkers read."""

    python = auto()
    rust = auto()
    markdown = auto()


# What joins an owner's name to what is defined in it, as each language writes it:
# `Stack.push`, `outer::Stack`, `Guide > Install`
PARENT_SEPARATORS = {Language.python: ".", Language.rust: "::", Language.markdown: " > "}


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
    """A chunk one channel of a search by meaning found, and how close its meaning is to the
    query's."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    # The cosine similarity of the chunk's vector of the model and the query's; None where
    # the chunk has none yet, which only its words found
    similarity: float | None

    @field_validator("similarity")
    @classmethod
    def similarity_must_be_in_range(cls, similarity: float | None) -> float | None:
        return None if similarity is None else similarity_in_range(similarity)


class SearchCandidates(BaseModel):
    """What each channel of a search found, best first, read from one version of the
    index: the chunks nearest the query's vector, and those whose words match its words."""

    model_config = ConfigDict(frozen=True)

    by_vector: list[Candidate]
    by_keyword: list[Candidate]


class SearchScope(BaseModel):
    """Which chunks a search looks among: of these kinds and languages, under the paths
    one glob names and outside those another names, held as `domain/globs.py` reads them,
    in the directory of the repository a search was asked from; an empty list or None
    leaves that part of the project in."""

    model_config = ConfigDict(frozen=True)

    kinds: tuple[ChunkKind, ...]
    languages: tuple[Language, ...]
    path_glob: str | None
    exclude_glob: str | None
    # From the project root
    directory: Path | None

    @field_validator("path_glob", "exclude_glob")
    @classmethod
    def globs_must_be_read(cls, glob: str | None) -> str | None:
        return read_glob(glob)

    @field_validator("directory")
    @classmethod
    def directory_must_be_relative(cls, directory: Path | None) -> Path | None:
        return None if directory is None else relative_path(directory)


# A scope that leaves nothing of the project out
WHOLE_PROJECT = SearchScope(
    kinds=(), languages=(), path_glob=None, exclude_glob=None, directory=None
)


class SearchMode(StrEnum):
    """How a server's searches rank the chunks they find. Its setting picks it, not the
    agent: agents gained nothing from meaning, and when offered it they took it for a first
    search (ADR 009, D2). A way of ranking added later is another member here."""

    # BM25 over the chunks' words, the chunks named as the query first; no model loaded
    words = auto()
    # The model's vectors searched too and fused with the words by rank
    words_and_meaning = auto()


class SearchQuery(BaseModel):
    """What a search is asked: its words, how many hits at most, and where to look."""

    model_config = ConfigDict(frozen=True)

    text: str
    limit: int
    scope: SearchScope


class SearchHit(BaseModel):
    """A chunk a search by meaning returns: where each channel ranked it, and the score
    fusing those ranks, which orders the hits."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    # None where the chunk has no vector of the model yet
    similarity: float | None
    # Counted from one; None where the channel did not find the chunk
    vector_rank: int | None
    keyword_rank: int | None
    fused_score: float

    @field_validator("similarity")
    @classmethod
    def similarity_must_be_in_range(cls, similarity: float | None) -> float | None:
        return None if similarity is None else similarity_in_range(similarity)


class FileStamp(BaseModel):
    """What a file was when a scan found it or the index recorded it: its modification time
    and size, which an edit of it moves."""

    model_config = ConfigDict(frozen=True)

    mtime_ns: int
    size_bytes: int


# A file's path from the project root as POSIX text: what a scan and the index key a file
# by, as a stale check compares every file found with every one the index recorded before
# each search, and a `Path` a file made that check cost several times more (ADR 010, O2)
type ProjectPath = str
# The files of a project, each with its stamp
type FileStamps = dict[ProjectPath, FileStamp]

# The number of no watch, which an event of the whole queue carries, as one of changes lost
NO_WATCH = -1


class WatchChange(StrEnum):
    """What an event of a watched directory says happened."""

    # An entry made, or moved in
    entry_added = auto()
    # An entry deleted, or moved out
    entry_removed = auto()
    # An entry written, closed after writing, or its attributes changed; or the directory's own
    entry_changed = auto()
    # The watched directory itself deleted, moved, unmounted, or its watch removed
    directory_gone = auto()
    # Changes went untold: the queue overflowed, the system dropped them, or it could not show
    # them all as they came
    changes_lost = auto()


class WatchEvent(BaseModel):
    """One event a watcher of directories hands back."""

    model_config = ConfigDict(frozen=True)

    # The number of the watch it came from; NO_WATCH for changes lost, which name none
    watch: int
    change: WatchChange
    # The entry's name in the watched directory, as the file system spells it; empty for an
    # event of the directory itself
    name: str
    is_directory: bool


class Place(BaseModel):
    """Where a directory lies: its device and inode, which its path keeps while nothing moves
    it or a directory above it."""

    model_config = ConfigDict(frozen=True)

    device: int
    inode: int

    @classmethod
    def of(cls, status: os.stat_result) -> Place:
        """Where what a `stat` read lies."""
        return cls(device=status.st_dev, inode=status.st_ino)


class SourceFile(BaseModel):
    """A file an index run indexes, with what it was when the scan found it.

    The index records each file it indexed as the scan found it, never stat'ed again
    later: an edit or a deletion made while the file is chunked and embedded leaves
    the record behind, so the next run sees the file as changed or gone. A file that
    gave no chunk is recorded too, so it is not indexed again on every run.
    """

    model_config = ConfigDict(frozen=True)

    # From the project root
    path: Path
    stamp: FileStamp

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
    """What a project's index of words is built from: the version of the chunker that
    splits. A project keeps an index for each, so a session of another version never builds
    the index of this one again.

    The model is not part of it: the index keeps each model's vectors apart, by the
    identity of what made them (`VectorIdentity`), so a change of model embeds again
    without chunking again, and a setting of a model never used builds nothing.
    """

    model_config = ConfigDict(frozen=True)

    chunker_version: int


class VectorIdentity(BaseModel):
    """What makes a chunk's vector: the model, by its name and the revision of its weights,
    and the version of how the embedder reads it. An index keeps the vectors of each apart,
    so newer weights under a model's name, or a new reading of them, never meet its old
    vectors.

    The vectors' width is not part of it: the model fixes it, and the stored vectors record
    it, so nothing needs the model loaded to find them.
    """

    model_config = ConfigDict(frozen=True)

    model_name: str
    # None where nothing pins the weights (a local path, a hub model the server does not
    # know), which are then trusted to stay what they were
    model_revision: str | None
    embedder_version: int


class StoreKind(StrEnum):
    """What keeps a project's indexes; each kind keeps indexes of its own."""

    sqlite = auto()
    lancedb = auto()


class IndexManifest(BaseModel):
    """What built an index, in which store and layout of it, and when it was last written."""

    model_config = ConfigDict(frozen=True)

    identity: IndexIdentity
    store_kind: StoreKind
    schema_version: int
    indexed_at: datetime


class IndexUpdate(BaseModel):
    """One write of an index run: whole files with their chunks, which take the place of
    what the index held of them, and the paths removed. A run writes a slice of its files
    at a time."""

    model_config = ConfigDict(frozen=True)

    # The files indexed, as the scan found them
    files: list[SourceFile]
    chunks: list[Chunk]
    removed_paths: list[Path]
    # Whether no write of the run follows: a store with work at a run's end does it inside
    # this write, so a run cancelled as it lands still leaves the index whole
    last_of_run: bool

    @model_validator(mode="after")
    def chunks_belong_to_the_files_written(self) -> IndexUpdate:
        written = {file.path for file in self.files}
        strays = sorted({chunk.path for chunk in self.chunks} - written)
        if strays:
            message = f"chunks of files the write does not hold: {strays}"
            raise ValueError(message)
        return self


class IndexedFiles(BaseModel):
    """What the index holds of the project's files: the stamp of each file it indexed, as
    the scan that indexed it found it, and the paths it holds chunks of with no record,
    which a write stopped between a file's chunks and its record leaves in a store that lays
    them apart; each path from the project root as POSIX text."""

    model_config = ConfigDict(frozen=True)

    records: FileStamps
    unrecorded_paths: list[ProjectPath]


class IndexCounts(BaseModel):
    """How much an index holds: files, chunks, and the chunks with a model's vectors."""

    model_config = ConfigDict(frozen=True)

    files: int
    chunks: int
    chunks_with_vectors: int


class IndexFreshness(BaseModel):
    """What a search checks of a project's index before it searches: whether there is one,
    and how many files changed since it was written."""

    model_config = ConfigDict(frozen=True)

    files_count: int
    # Files the index no longer matches: new, modified or deleted since it was written
    stale_files_count: int

    @property
    def is_indexed(self) -> bool:
        return self.files_count > 0


class IndexStatus(BaseModel):
    """How far a project's index is from its files."""

    model_config = ConfigDict(frozen=True)

    files_count: int
    chunks_count: int
    # The model a search by meaning embeds with, and how many chunks have its vectors
    model: VectorIdentity
    chunks_with_vectors_count: int
    # Files the index no longer matches: new, modified or deleted since it was written
    stale_files_count: int
    manifest: IndexManifest | None
    # Whether a run, this session's or another's, holds the index as the status is read, so
    # its counts may be what the run has written so far
    being_built: bool

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
    """What one search found, best first, how much it dropped, and what it cost."""

    model_config = ConfigDict(frozen=True)

    results: list[Chunk]
    filtered_count: int
    timings: SearchTimings
    # Set when the search had to index stale files first
    index_report: IndexReport | None
