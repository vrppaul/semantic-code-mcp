"""Tests for the LanceDB store: what is written comes back, by vector and by words."""

import os
import re
from collections.abc import Callable
from pathlib import Path

import lancedb
import numpy
import pyarrow
import pytest
from filelock import FileLock, Timeout
from hypothesis import HealthCheck, given, settings, strategies

from semantic_code_mcp.domain.models import (
    WHOLE_PROJECT,
    Candidate,
    Chunk,
    ChunkKind,
    IndexCounts,
    IndexIdentity,
    IndexUpdate,
    Language,
    SearchQuery,
    SearchScope,
    SourceFile,
)
from semantic_code_mcp.storage.globs import glob_pattern
from semantic_code_mcp.storage.lancedb import (
    CHUNKS_TABLE,
    MANIFEST_FILE,
    NO_CANDIDATES,
    REMOVING_PREFIX,
    SCHEMA_VERSION,
    UPKEEPS_PER_FULL_TEXT_REBUILD,
    USE_MARK_INTERVAL,
    USE_STAMP_FILE,
    LanceDBIndexStore,
)
from tests.fakes import EMBEDDING_DIMENSION
from tests.helpers import chunks_outside_the_full_text_index, newest_full_text_index_files

TOWARD_QUERY = [0.9] * EMBEDDING_DIMENSION
QUERY = numpy.full(EMBEDDING_DIMENSION, 1.0, dtype=numpy.float32)
AWAY_FROM_QUERY = [-0.9] * EMBEDDING_DIMENSION
ALTERNATING = [0.1, -0.1] * (EMBEDDING_DIMENSION // 2)
LIMIT = 10
# A text no chunk's words match, so a search finds by vector alone
NO_WORDS_MATCH = "zzzz"
LOCK_TIMEOUT_SECONDS = 5.0
# Long enough to tell a write that waits from one that does not, short for a test
SHORT_LOCK_TIMEOUT_SECONDS = 0.2
# Modes for a table's files: read and enter only, so a write to it fails; and writable again
READ_ONLY = 0o555
WRITABLE = 0o755
IDENTITY = IndexIdentity(model_name="fake-model", model_revision=None, chunker_version=1)
OTHER_IDENTITY = IndexIdentity(model_name="other-model", model_revision=None, chunker_version=1)
# A time long before any test runs: an index last written then is long unused
LONG_AGO_SECONDS = 1_000_000_000
# What 0.4 kept in a project's index directory: its chunks table, and its record of mtimes
LEGACY_TABLE = "chunks"
LEGACY_RECORDS = "file_mtimes.json"
LEGACY_ROW = {
    "vector": [0.5] * EMBEDDING_DIMENSION,
    "file_path": "/elsewhere/old.py",
    "line_start": 1,
    "line_end": 2,
    "content": "def old(): pass",
    "chunk_type": "function",
    "name": "old",
}


def function_chunk(name: str, path: str, content: str) -> Chunk:
    return Chunk(
        chunk_id=f"{path}#{name}",
        path=Path(path),
        language=Language.python,
        kind=ChunkKind.function,
        name=name,
        parent=None,
        line_start=1,
        line_end=3,
        content=content,
    )


def record_of(path: str) -> SourceFile:
    return SourceFile(path=Path(path), mtime_ns=1, size_bytes=1)


def update_of(chunks: list[Chunk], removed_paths: list[Path]) -> IndexUpdate:
    return IndexUpdate(
        files=[record_of(chunk.path.as_posix()) for chunk in chunks],
        chunks=chunks,
        removed_paths=removed_paths,
    )


def vectors_of(*rows: list[float]) -> numpy.ndarray:
    return numpy.array(rows, dtype=numpy.float32)


def query_of(text: str) -> SearchQuery:
    """A query of the whole project."""
    return SearchQuery(text=text, limit=LIMIT, scope=WHOLE_PROJECT)


def nearest(store: LanceDBIndexStore, query: numpy.ndarray = QUERY) -> list[Candidate]:
    """What the vector channel of a search finds."""
    return store.search_candidates(query, query_of(NO_WORDS_MATCH), LIMIT).by_vector


def matching(store: LanceDBIndexStore, text: str) -> list[Candidate]:
    """What the keyword channel of a search for the text finds."""
    return store.search_candidates(QUERY, query_of(text), LIMIT).by_keyword


@pytest.fixture
def store(tmp_path: Path) -> LanceDBIndexStore:
    return LanceDBIndexStore(
        tmp_path / "index", IDENTITY, lock_timeout_seconds=LOCK_TIMEOUT_SECONDS
    )


@pytest.fixture
def filled_store(store: LanceDBIndexStore) -> LanceDBIndexStore:
    chunks = [
        function_chunk("near", "near.py", "def near(): return elapsed_time()"),
        function_chunk(
            "keyword",
            "indexer.py",
            'log.debug("timing", duration_ms=round(time.time() - started))',
        ),
        function_chunk("far", "far.py", "def far(): return None"),
    ]
    store.write(
        update_of(chunks, removed_paths=[]),
        vectors_of(TOWARD_QUERY, ALTERNATING, AWAY_FROM_QUERY),
    )
    return store


def test_vector_search_ranks_the_nearest_chunk_first(filled_store: LanceDBIndexStore):
    assert [candidate.chunk.name for candidate in nearest(filled_store)] == [
        "near",
        "keyword",
        "far",
    ]


def test_a_chunk_comes_back_from_a_search_as_it_was_written(store: LanceDBIndexStore):
    method = Chunk(
        chunk_id="stack-push",
        path=Path("src/stack.rs"),
        language=Language.rust,
        kind=ChunkKind.method,
        name="push",
        parent="Stack",
        line_start=4,
        line_end=6,
        content="fn push(&mut self, item: T) {}",
    )

    store.write(update_of([method], removed_paths=[]), vectors_of(TOWARD_QUERY))

    assert [candidate.chunk for candidate in nearest(store)] == [method]


def test_full_text_search_finds_an_exact_identifier(filled_store: LanceDBIndexStore):
    assert [candidate.chunk.name for candidate in matching(filled_store, "duration_ms")] == [
        "keyword"
    ]


@pytest.mark.parametrize(
    "words",
    [
        pytest.param("item", id="of its name in camel case"),
        pytest.param("bounded stack", id="of the class it is a method of"),
    ],
)
def test_full_text_search_finds_a_chunk_by_the_words_of_its_name_and_parent(
    store: LanceDBIndexStore, words: str
):
    method = Chunk(
        chunk_id="stack-add",
        path=Path("src/stack.rs"),
        language=Language.rust,
        kind=ChunkKind.method,
        name="addItem",
        parent="BoundedStack",
        line_start=4,
        line_end=6,
        content="fn addItem(&mut self) { self.count += 1; }",
    )
    store.write(update_of([method], removed_paths=[]), vectors_of(TOWARD_QUERY))

    assert [candidate.chunk.name for candidate in matching(store, words)] == ["addItem"]


def test_full_text_search_finds_chunks_of_a_later_write_and_not_removed_ones(
    filled_store: LanceDBIndexStore,
):
    later = function_chunk("later", "later.py", "def later(): return retry_budget")
    filled_store.write(
        update_of([later], removed_paths=[Path("indexer.py")]), vectors_of(TOWARD_QUERY)
    )

    assert [candidate.chunk.name for candidate in matching(filled_store, "retry_budget")] == [
        "later"
    ]
    assert matching(filled_store, "duration_ms") == []


def test_a_word_lancedb_drops_by_default_still_finds_a_chunk(store: LanceDBIndexStore):
    """The index keeps the words LanceDB drops by default, as `not` here."""
    chunk = function_chunk("is_empty", "check.py", "def is_empty(items): return not items")
    store.write(update_of([chunk], removed_paths=[]), vectors_of(TOWARD_QUERY))

    assert [candidate.chunk.name for candidate in matching(store, "not")] == ["is_empty"]


def test_each_candidate_carries_the_similarity_of_its_vector_to_the_query(
    filled_store: LanceDBIndexStore,
):
    candidates = filled_store.search_candidates(QUERY, query_of("duration_ms"), LIMIT)

    assert [
        candidate.similarity for candidate in [*candidates.by_vector, *candidates.by_keyword]
    ] == pytest.approx([1.0, 0.0, -1.0, 0.0])


# More chunks near the query than a channel brings, all code under src/
NEAR_CODE_CHUNKS = 20
NARROW_CHANNEL = 5
# The guide holds GUIDE_QUESTION's "guide"; the code holds both its words, so the guide ranks
# below every code chunk unfiltered, or only "shared", so the guide is the one chunk "guide"
# matches, which LanceDB 0.27's filtered full-text search lost
GUIDE_QUESTION = "shared guide"
CODE_BELOW_THE_CUT = "def f(): return shared guide"
CODE_SHARING_ONE_WORD = "def f(): return shared"


@pytest.fixture
def code_and_a_guide(store: LanceDBIndexStore, request: pytest.FixtureRequest) -> LanceDBIndexStore:
    """Code chunks near the query, more than a channel brings, each holding the code the test
    asks for, and one section of a guide far from it."""
    code = [
        function_chunk(f"function_{number}", f"src/module_{number}.py", request.param)
        for number in range(NEAR_CODE_CHUNKS)
    ]
    guide = Chunk(
        chunk_id="guide",
        path=Path("docs/guide.md"),
        language=Language.markdown,
        kind=ChunkKind.section,
        name="guide",
        parent=None,
        line_start=1,
        line_end=3,
        content="A guide.",
    )
    store.write(
        update_of([*code, guide], removed_paths=[]),
        vectors_of(*([TOWARD_QUERY] * NEAR_CODE_CHUNKS), AWAY_FROM_QUERY),
    )
    return store


@pytest.mark.parametrize(
    "code_and_a_guide",
    [
        pytest.param(CODE_BELOW_THE_CUT, id="the guide below the cut"),
        pytest.param(CODE_SHARING_ONE_WORD, id="the guide alone matching a word"),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "scope",
    [
        pytest.param(
            SearchScope(
                kinds=[ChunkKind.section],
                languages=[],
                path_glob=None,
                exclude_glob=None,
                directory=None,
            ),
            id="by kind",
        ),
        pytest.param(
            SearchScope(
                kinds=[],
                languages=[Language.markdown],
                path_glob=None,
                exclude_glob=None,
                directory=None,
            ),
            id="by language",
        ),
        pytest.param(
            SearchScope(
                kinds=[], languages=[], path_glob="docs/**", exclude_glob=None, directory=None
            ),
            id="by a glob",
        ),
        pytest.param(
            SearchScope(kinds=[], languages=[], path_glob=None, exclude_glob="src", directory=None),
            id="by a glob left out",
        ),
        pytest.param(
            SearchScope(
                kinds=[], languages=[], path_glob=None, exclude_glob=None, directory=Path("docs")
            ),
            id="by a directory",
        ),
    ],
)
def test_both_channels_find_within_the_scope_before_they_cut(
    code_and_a_guide: LanceDBIndexStore, scope: SearchScope
):
    """The guide is further than every chunk a narrow channel would bring unfiltered by
    vector, and by words too in the first case."""
    query = SearchQuery(text=GUIDE_QUESTION, limit=NARROW_CHANNEL, scope=scope)

    candidates = code_and_a_guide.search_candidates(QUERY, query, NARROW_CHANNEL)

    assert [
        [candidate.chunk.name for candidate in candidates.by_vector],
        [candidate.chunk.name for candidate in candidates.by_keyword],
    ] == [["guide"], ["guide"]]


# Paths whose characters a regex, SQL or a glob reads apart, each in a file of its own
AWKWARD_PATHS = [
    "src/module.py",
    "src/deep/er/module.py",
    "src/[ab].py",
    "src/a-b.py",
    "src/a^b&&c~~d.py",
    "src/a\\b.py",
    "lib(1)/it's.rs",
    "with space/100%_done.md",
    "ünïcode/x.md",
]
# What a part of a glob between slashes is written from: wildcards and whatever a regex or
# SQL reads apart, but no character a glob is refused for
GLOB_PART_CHARACTERS = "ab.*?]}^-&~()|+$#' %_ü"
# A part of a glob as it is read: no blanks around it, not a step to this directory or the
# one above it
READ_PARTS = strategies.text(alphabet=GLOB_PART_CHARACTERS, min_size=1, max_size=6).filter(
    lambda part: part == part.strip() and part not in {".", ".."}
)
# A glob as it is read: a name at any depth, or a path from the directory searched
READ_GLOBS = READ_PARTS | strategies.lists(READ_PARTS, min_size=1, max_size=3).map(
    lambda parts: "/" + "/".join(parts)
)
# More than every chunk of AWKWARD_PATHS, so a channel brings all it matches
EVERY_CHUNK = 20


@pytest.fixture
def awkward_store(store: LanceDBIndexStore) -> LanceDBIndexStore:
    chunks = [
        function_chunk(f"function_{number}", path, "def f(): return shared")
        for number, path in enumerate(AWKWARD_PATHS)
    ]
    store.write(update_of(chunks, removed_paths=[]), vectors_of(*([TOWARD_QUERY] * len(chunks))))
    return store


# The store is only read, so one for every example does
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(glob=READ_GLOBS)
def test_the_store_reads_a_glob_as_its_pattern_says(awkward_store: LanceDBIndexStore, glob: str):
    """LanceDB's regex is Rust's, which refuses or reads apart what Python's would read."""
    scope = SearchScope(kinds=[], languages=[], path_glob=glob, exclude_glob=None, directory=None)
    query = SearchQuery(text="shared", limit=EVERY_CHUNK, scope=scope)

    candidates = awkward_store.search_candidates(QUERY, query, EVERY_CHUNK)

    pattern = re.compile(glob_pattern(glob, base=None))
    named = {path: bool(pattern.fullmatch(path)) for path in AWKWARD_PATHS}
    found = {candidate.chunk.path.as_posix() for candidate in candidates.by_vector}
    found_by_words = {candidate.chunk.path.as_posix() for candidate in candidates.by_keyword}
    assert [
        scope.path_glob,
        {path: path in found for path in AWKWARD_PATHS},
        {path: path in found_by_words for path in AWKWARD_PATHS},
    ] == [glob, named, named]


def test_a_search_warns_of_nothing(
    filled_store: LanceDBIndexStore, capfd: pytest.CaptureFixture[str]
):
    """LanceDB writes to the process's standard error, which an MCP client logs."""
    capfd.readouterr()

    filled_store.search_candidates(QUERY, query_of("duration_ms"), LIMIT)

    assert capfd.readouterr().err == ""


def test_an_empty_store_finds_nothing(store: LanceDBIndexStore):
    assert store.search_candidates(QUERY, query_of("anything"), LIMIT) == NO_CANDIDATES


def test_removing_files_takes_only_their_chunks_and_records(filled_store: LanceDBIndexStore):
    filled_store.write(
        update_of([], removed_paths=[Path("near.py"), Path("it's quoted.py")]),
        numpy.empty((0, 0), dtype=numpy.float32),
    )

    assert sorted(str(file.path) for file in filled_store.indexed_files()) == [
        "far.py",
        "indexer.py",
    ]
    assert filled_store.counts().chunks == 2


def test_a_write_records_what_built_the_index(store: LanceDBIndexStore):
    before = store.manifest()
    store.write(update_of([], removed_paths=[]), numpy.empty((0, 0), dtype=numpy.float32))
    after = store.manifest()

    assert (before, after.identity, after.schema_version) == (None, IDENTITY, SCHEMA_VERSION)


def test_a_write_waits_while_another_process_writes(tmp_path: Path):
    store = LanceDBIndexStore(
        tmp_path / "index", IDENTITY, lock_timeout_seconds=SHORT_LOCK_TIMEOUT_SECONDS
    )
    store.directory.mkdir(parents=True)
    chunk = function_chunk("held", "held.py", "def held(): pass")

    with FileLock(store.write_lock), pytest.raises(Timeout):
        store.write(update_of([chunk], removed_paths=[]), vectors_of(TOWARD_QUERY))

    assert store.counts().chunks == 0


def test_a_store_sees_what_another_store_on_its_directory_wrote(tmp_path: Path):
    """Two server processes on one project: each reads what the other has written."""
    directory = tmp_path / "index"
    reader = LanceDBIndexStore(directory, IDENTITY, lock_timeout_seconds=LOCK_TIMEOUT_SECONDS)
    writer = LanceDBIndexStore(directory, IDENTITY, lock_timeout_seconds=LOCK_TIMEOUT_SECONDS)
    first = function_chunk("first", "first.py", "def first(): pass")
    second = function_chunk("second", "second.py", "def second(): pass")
    writer.write(update_of([first], removed_paths=[]), vectors_of(TOWARD_QUERY))
    before = reader.counts().chunks

    writer.write(update_of([second], removed_paths=[]), vectors_of(AWAY_FROM_QUERY))

    assert (before, reader.counts().chunks) == (1, 2)


def test_vectors_of_another_width_replace_the_index_and_leave_the_other_files_unrecorded(
    filled_store: LanceDBIndexStore,
):
    """A model replaced under a name nothing pins gives vectors of another width: the index starts
    again from the files written, and the next run indexes the others, unrecorded now."""
    edited = function_chunk("near", "near.py", "def near(): return elapsed_time()")

    filled_store.write(update_of([edited], removed_paths=[]), vectors_of(TOWARD_QUERY * 2))

    wide_query = numpy.full(2 * EMBEDDING_DIMENSION, 1.0, dtype=numpy.float32)
    assert filled_store.counts() == IndexCounts(files=1, chunks=1)
    assert [candidate.chunk.name for candidate in nearest(filled_store, wide_query)] == ["near"]


def test_an_index_of_another_identity_on_the_project_is_kept_apart(
    filled_store: LanceDBIndexStore, tmp_path: Path
):
    """Two models, or two versions of the chunker, on one project never write each other's index."""
    other = LanceDBIndexStore(
        tmp_path / "index", OTHER_IDENTITY, lock_timeout_seconds=LOCK_TIMEOUT_SECONDS
    )
    chunk = function_chunk("other", "other.py", "def other(): pass")

    other.write(update_of([chunk], removed_paths=[]), vectors_of(AWAY_FROM_QUERY))

    assert (filled_store.counts(), other.counts()) == (
        IndexCounts(files=3, chunks=3),
        IndexCounts(files=1, chunks=1),
    )


def leave_an_index_of_another_identity(project_directory: Path) -> list[str]:
    """Another model's index of the project; the names it made in the project's directory."""
    other = LanceDBIndexStore(
        project_directory, OTHER_IDENTITY, lock_timeout_seconds=LOCK_TIMEOUT_SECONDS
    )
    chunk = function_chunk("other", "other.py", "def other(): pass")
    other.write(update_of([chunk], removed_paths=[]), vectors_of(AWAY_FROM_QUERY))
    return [other.directory.name]


def leave_an_index_of_version_0_4(project_directory: Path) -> list[str]:
    """What 0.4 kept in the project's directory: a chunks table and a record of mtimes."""
    lancedb.connect(str(project_directory)).create_table(
        LEGACY_TABLE, data=pyarrow.Table.from_pylist([LEGACY_ROW])
    )
    (project_directory / LEGACY_RECORDS).write_text("{}")
    return [f"{LEGACY_TABLE}.lance", LEGACY_RECORDS]


LEFT_INDEXES = [
    pytest.param(leave_an_index_of_another_identity, id="another identity"),
    pytest.param(leave_an_index_of_version_0_4, id="version 0.4"),
]


@pytest.mark.parametrize("leave_index", LEFT_INDEXES)
def test_an_index_written_within_the_week_is_left_alone(
    store: LanceDBIndexStore, tmp_path: Path, leave_index: Callable[[Path], list[str]]
):
    """Another session may still use it; this store reads and writes its own index only."""
    left = leave_index(tmp_path / "index")
    chunk = function_chunk("own", "own.py", "def own(): pass")

    store.write(update_of([chunk], removed_paths=[]), vectors_of(TOWARD_QUERY))

    assert set(left) <= set(names_in(tmp_path / "index"))
    assert store.counts() == IndexCounts(files=1, chunks=1)


def write_own_index(store: LanceDBIndexStore) -> None:
    chunk = function_chunk("own", "own.py", "def own(): pass")
    store.write(update_of([chunk], removed_paths=[]), vectors_of(TOWARD_QUERY))


def read_own_index(store: LanceDBIndexStore) -> None:
    store.counts()


@pytest.mark.parametrize("leave_index", LEFT_INDEXES)
@pytest.mark.parametrize(
    "use_own_index",
    [pytest.param(write_own_index, id="written"), pytest.param(read_own_index, id="read")],
)
def test_an_index_nobody_used_for_a_week_is_removed_when_another_is_used(
    store: LanceDBIndexStore,
    tmp_path: Path,
    leave_index: Callable[[Path], list[str]],
    use_own_index: Callable[[LanceDBIndexStore], None],
):
    """A week after both were last used, a session of this store's index reads or writes it."""
    write_own_index(
        LanceDBIndexStore(tmp_path / "index", IDENTITY, lock_timeout_seconds=LOCK_TIMEOUT_SECONDS)
    )
    left = leave_index(tmp_path / "index")
    set_mtime_of_tree(tmp_path / "index", LONG_AGO_SECONDS)

    use_own_index(store)

    assert set(names_in(tmp_path / "index")).isdisjoint(left)


def test_an_index_only_read_within_the_week_is_left_alone(store: LanceDBIndexStore, tmp_path: Path):
    """A session that searches a project nobody edits keeps its index, unwritten for weeks."""
    left = leave_an_index_of_another_identity(tmp_path / "index")
    read_as_another_session(tmp_path / "index")
    set_mtime_of_tree(tmp_path / "index", LONG_AGO_SECONDS)
    read_as_another_session(tmp_path / "index")

    write_own_index(store)

    assert set(left) <= set(names_in(tmp_path / "index"))


def read_as_another_session(project_directory: Path) -> None:
    """A search of the other identity's index, by a server process of its own."""
    LanceDBIndexStore(
        project_directory, OTHER_IDENTITY, lock_timeout_seconds=LOCK_TIMEOUT_SECONDS
    ).counts()


def test_a_store_marks_its_index_again_once_its_stamp_is_older_than_the_interval(
    filled_store: LanceDBIndexStore,
):
    """However long the machine slept since this store last looked: the stamp tells."""
    stamp = filled_store.directory / USE_STAMP_FILE
    filled_store.counts()
    long_before = stamp.stat().st_mtime - 2 * USE_MARK_INTERVAL.total_seconds()
    os.utime(stamp, (long_before, long_before))

    filled_store.counts()

    assert stamp.stat().st_mtime > long_before + USE_MARK_INTERVAL.total_seconds()


def test_an_unused_index_that_another_process_is_writing_is_left_alone(
    store: LanceDBIndexStore, tmp_path: Path
):
    other = LanceDBIndexStore(
        tmp_path / "index", OTHER_IDENTITY, lock_timeout_seconds=LOCK_TIMEOUT_SECONDS
    )
    write_own_index(other)
    set_mtime_of_tree(tmp_path / "index", LONG_AGO_SECONDS)

    with FileLock(other.write_lock):
        write_own_index(store)

    assert other.directory.name in names_in(tmp_path / "index")


def test_a_removal_stopped_midway_is_finished_by_the_next_use(
    store: LanceDBIndexStore, tmp_path: Path
):
    """An index is renamed before it is deleted; a crash between the two leaves the name."""
    half_removed = tmp_path / "index" / f"{REMOVING_PREFIX}schema-1-0123456789abcdef"
    half_removed.mkdir(parents=True)
    (half_removed / MANIFEST_FILE).write_text("{}")

    write_own_index(store)

    assert half_removed.name not in names_in(tmp_path / "index")


def names_in(directory: Path) -> list[str]:
    return sorted(entry.name for entry in directory.iterdir())


def set_mtime_of_tree(directory: Path, seconds: int) -> None:
    """Set the modification time of a directory and of everything under it."""
    for path in [directory, *directory.rglob("*")]:
        os.utime(path, (seconds, seconds))


def write_bytes_that_are_not_utf8(manifest: Path) -> None:
    manifest.write_bytes(b'{"identity": "\xff"}')


def make_a_directory(manifest: Path) -> None:
    manifest.mkdir()


@pytest.mark.parametrize(
    "spoil",
    [
        pytest.param(write_bytes_that_are_not_utf8, id="not UTF-8"),
        pytest.param(make_a_directory, id="a directory"),
    ],
)
def test_a_manifest_that_cannot_be_read_counts_as_none(
    filled_store: LanceDBIndexStore, spoil: Callable[[Path], None]
):
    manifest = filled_store.directory / MANIFEST_FILE
    manifest.unlink()
    spoil(manifest)

    assert filled_store.manifest() is None


def test_a_write_whose_chunks_fail_leaves_its_files_records_as_they_were(
    filled_store: LanceDBIndexStore,
):
    """Chunks go before records: a write stopped between them leaves the old records, which
    the next run reads as files changed, never new records over chunks that are not there."""
    chunks_table_directory = filled_store.directory / f"{CHUNKS_TABLE}.lance"
    set_mode_of_tree(chunks_table_directory, READ_ONLY)
    edited = function_chunk("near", "near.py", "def near(): return elapsed_time() * 2")
    update = IndexUpdate(
        files=[SourceFile(path=Path("near.py"), mtime_ns=2, size_bytes=2)],
        chunks=[edited],
        removed_paths=[],
    )

    with pytest.raises((OSError, RuntimeError)):
        filled_store.write(update, vectors_of(TOWARD_QUERY))
    set_mode_of_tree(chunks_table_directory, WRITABLE)

    assert sorted(filled_store.indexed_files(), key=lambda file: file.path) == [
        record_of("far.py"),
        record_of("indexer.py"),
        record_of("near.py"),
    ]


def test_a_file_s_new_chunks_take_the_place_of_its_old_ones_in_one_commit(
    filled_store: LanceDBIndexStore,
):
    """A reader sees the file's old chunks or its new ones, never neither."""
    chunks_table = lancedb.connect(str(filled_store.directory)).open_table(CHUNKS_TABLE)
    version_before = chunks_table.version
    edited = function_chunk("near", "near.py", "def near(): return elapsed_time() * 2")

    filled_store.write(update_of([edited], removed_paths=[]), vectors_of(TOWARD_QUERY))

    chunks_table.checkout_latest()
    assert (chunks_table.version, filled_store.counts().chunks) == (version_before + 1, 3)


def test_a_chunk_written_again_under_its_id_stands_where_it_now_is(
    filled_store: LanceDBIndexStore,
):
    """Lines inserted above a function move it, its text and id unchanged."""
    near = nearest(filled_store)[0].chunk
    moved = Chunk(
        chunk_id=near.chunk_id,
        path=near.path,
        language=near.language,
        kind=near.kind,
        name=near.name,
        parent="Timer",
        line_start=near.line_start + 2,
        line_end=near.line_end + 2,
        content=near.content,
    )

    filled_store.write(update_of([moved], removed_paths=[]), vectors_of(TOWARD_QUERY))

    assert nearest(filled_store)[0].chunk == moved


def test_the_vectors_of_stored_chunks_come_back_by_their_ids(filled_store: LanceDBIndexStore):
    found = filled_store.stored_vectors(["near.py#near", "nowhere.py#missing"])

    assert list(found) == ["near.py#near"]
    assert numpy.array_equal(found["near.py#near"], vectors_of(TOWARD_QUERY)[0])


def test_the_paths_the_chunks_come_from_come_back(filled_store: LanceDBIndexStore):
    assert filled_store.chunked_paths() == {Path("near.py"), Path("indexer.py"), Path("far.py")}


def test_keeping_up_folds_every_chunk_into_the_full_text_index(filled_store: LanceDBIndexStore):
    later = function_chunk("later", "later.py", "def later(): return retry_budget")
    filled_store.write(update_of([later], removed_paths=[]), vectors_of(TOWARD_QUERY))

    filled_store.keep_up()

    assert chunks_outside_the_full_text_index(filled_store) == 0


def set_mode_of_tree(directory: Path, mode: int) -> None:
    """Set the mode of a directory, everything under it, and their directories."""
    for path in [directory, *directory.rglob("*")]:
        path.chmod(mode)


def test_upkeep_rebuilds_the_full_text_index_before_its_parts_pile_up(
    filled_store: LanceDBIndexStore,
):
    """Each upkeep folds new rows in as a part of their own, which no later upkeep merges."""
    filled_store.keep_up()
    rebuilt = newest_full_text_index_files(filled_store)

    for edit in range(UPKEEPS_PER_FULL_TEXT_REBUILD):
        near = function_chunk("near", "near.py", f"def near(): return {edit}")
        filled_store.write(update_of([near], removed_paths=[]), vectors_of(TOWARD_QUERY))
        filled_store.keep_up()

    assert newest_full_text_index_files(filled_store) == rebuilt
