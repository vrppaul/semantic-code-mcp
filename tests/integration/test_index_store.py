"""Tests of the store's contract, held over every store: what is written comes back, by
vector and by words, within a search's scope; a write replaces, removes, records what built
the index and waits for another writer; reads and writes mark the index as used, so the
project's indexes nobody uses go. Where the index lies, and how unused ones go, is the
index directory's (`test_index_directories.py`)."""

import os
import re
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy
import pytest
from filelock import FileLock, Timeout
from hypothesis import HealthCheck, given, settings, strategies

from semantic_code_mcp.domain.interfaces import IndexStore
from semantic_code_mcp.domain.models import (
    Chunk,
    ChunkKind,
    IndexCounts,
    Language,
    SearchQuery,
    SearchScope,
)
from semantic_code_mcp.storage.globs import glob_pattern
from semantic_code_mcp.storage.index_directories import IndexDirectory
from tests.fakes import EMBEDDING_DIMENSION
from tests.stores import (
    ALTERNATING,
    AWAY_FROM_QUERY,
    IDENTITY,
    LIMIT,
    NO_CANDIDATES,
    NO_WORDS_MATCH,
    OTHER_IDENTITY,
    QUERY,
    TOWARD_QUERY,
    StoreClass,
    function_chunk,
    index_directory_of,
    matching,
    nearest,
    query_of,
    update_of,
    vectors_of,
)


@pytest.fixture
def index_directory(store_class: StoreClass, tmp_path: Path) -> IndexDirectory:
    return index_directory_of(store_class, tmp_path / "index", IDENTITY)


@pytest.fixture
def store(store_class: StoreClass, index_directory: IndexDirectory) -> IndexStore:
    return store_class(index_directory)


# The chunks a filled store holds, nearest the query first
FILLED_CHUNK_NAMES = ("near", "keyword", "far")


@pytest.fixture
def filled_store(store: IndexStore) -> IndexStore:
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


# Long enough to tell a write that waits from one that does not, short for a test
SHORT_LOCK_TIMEOUT_SECONDS = 0.2
# A time long before any test runs: an index last used then is long unused
LONG_AGO_SECONDS = 1_000_000_000


def test_vector_search_ranks_the_nearest_chunk_first(filled_store: IndexStore):
    assert [candidate.chunk.name for candidate in nearest(filled_store)] == [
        "near",
        "keyword",
        "far",
    ]


# How many of the chunks below a channel takes, fewer than there are
FEWER_THAN_STORED = 3


def test_vector_search_brings_the_nearest_chunks_of_more_than_it_takes(store: IndexStore):
    """Written in an order unlike their nearness: each chunk's vector agrees with the query
    in as many components as its number says, the rest opposed."""
    agreeing_components = [3, 7, 1, 8, 5, 2, 6, 4]
    chunks = [
        function_chunk(f"agrees_in_{count}", f"agrees_in_{count}.py", "def f(): pass")
        for count in agreeing_components
    ]
    vectors = vectors_of(
        *([1.0] * count + [-1.0] * (EMBEDDING_DIMENSION - count) for count in agreeing_components)
    )
    store.write(update_of(chunks, removed_paths=[]), vectors)

    candidates = store.search_candidates(QUERY, query_of(NO_WORDS_MATCH), FEWER_THAN_STORED)

    assert [candidate.chunk.name for candidate in candidates.by_vector] == [
        "agrees_in_8",
        "agrees_in_7",
        "agrees_in_6",
    ]


def test_a_chunk_comes_back_from_a_search_as_it_was_written(store: IndexStore):
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


def test_full_text_search_finds_an_exact_identifier(filled_store: IndexStore):
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
    store: IndexStore, words: str
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


def test_a_question_wholly_in_quotes_finds_a_chunk_by_its_words(filled_store: IndexStore):
    """An agent quotes an identifier it knows; LanceDB read such a text as a phrase, which
    an index without word positions cannot answer, and the search failed."""
    assert [candidate.chunk.name for candidate in matching(filled_store, '"duration_ms"')] == [
        "keyword"
    ]


# How many questions of any text a store is asked
ANY_QUESTION_EXAMPLES = 200


# The store is only read, so one for every example does
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    max_examples=ANY_QUESTION_EXAMPLES,
)
@given(question=strategies.text())
def test_a_question_of_any_text_gets_an_answer(filled_store: IndexStore, question: str):
    """An agent's question is never read as a store's own query syntax: quotes, operators,
    brackets and words like OR, NEAR or NOT are only text."""
    candidates = filled_store.search_candidates(QUERY, query_of(question), LIMIT)

    assert len(candidates.by_vector) == len(FILLED_CHUNK_NAMES)


def test_full_text_search_finds_chunks_of_a_later_write_and_not_removed_ones(
    filled_store: IndexStore,
):
    later = function_chunk("later", "later.py", "def later(): return retry_budget")
    filled_store.write(
        update_of([later], removed_paths=[Path("indexer.py")]), vectors_of(TOWARD_QUERY)
    )

    assert [candidate.chunk.name for candidate in matching(filled_store, "retry_budget")] == [
        "later"
    ]
    assert matching(filled_store, "duration_ms") == []


def test_a_stop_word_still_finds_a_chunk(store: IndexStore):
    """The index keeps stop words, which LanceDB drops by default, as `not` here."""
    chunk = function_chunk("is_empty", "check.py", "def is_empty(items): return not items")
    store.write(update_of([chunk], removed_paths=[]), vectors_of(TOWARD_QUERY))

    assert [candidate.chunk.name for candidate in matching(store, "not")] == ["is_empty"]


def test_each_candidate_carries_the_similarity_of_its_vector_to_the_query(
    filled_store: IndexStore,
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
def code_and_a_guide(store: IndexStore, request: pytest.FixtureRequest) -> IndexStore:
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
    code_and_a_guide: IndexStore, scope: SearchScope
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
def awkward_store(store: IndexStore) -> IndexStore:
    chunks = [
        function_chunk(f"function_{number}", path, "def f(): return shared")
        for number, path in enumerate(AWKWARD_PATHS)
    ]
    store.write(update_of(chunks, removed_paths=[]), vectors_of(*([TOWARD_QUERY] * len(chunks))))
    return store


# The store is only read, so one for every example does
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(glob=READ_GLOBS)
def test_the_store_reads_a_glob_as_its_pattern_says(awkward_store: IndexStore, glob: str):
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


def test_an_empty_store_finds_nothing(store: IndexStore):
    assert store.search_candidates(QUERY, query_of("anything"), LIMIT) == NO_CANDIDATES


def test_removing_files_takes_only_their_chunks_and_records(filled_store: IndexStore):
    filled_store.write(
        update_of([], removed_paths=[Path("near.py"), Path("it's quoted.py")]),
        numpy.empty((0, 0), dtype=numpy.float32),
    )

    assert sorted(str(file.path) for file in filled_store.indexed_files().records) == [
        "far.py",
        "indexer.py",
    ]
    assert filled_store.counts().chunks == 2


def test_a_write_waits_while_another_process_writes(tmp_path: Path, store_class: StoreClass):
    directory = index_directory_of(
        store_class, tmp_path / "index", IDENTITY, SHORT_LOCK_TIMEOUT_SECONDS
    )
    store = store_class(directory)
    directory.path.mkdir(parents=True)
    chunk = function_chunk("held", "held.py", "def held(): pass")

    with FileLock(directory.write_lock), pytest.raises(Timeout):
        store.write(update_of([chunk], removed_paths=[]), vectors_of(TOWARD_QUERY))

    assert store.counts().chunks == 0


def test_a_store_sees_what_another_store_on_its_directory_wrote(
    store_class: StoreClass, index_directory: IndexDirectory
):
    """Two server processes on one project: each reads what the other has written."""
    reader = store_class(index_directory)
    writer = store_class(index_directory)
    first = function_chunk("first", "first.py", "def first(): pass")
    second = function_chunk("second", "second.py", "def second(): pass")
    writer.write(update_of([first], removed_paths=[]), vectors_of(TOWARD_QUERY))
    before = reader.counts().chunks

    writer.write(update_of([second], removed_paths=[]), vectors_of(AWAY_FROM_QUERY))

    assert (before, reader.counts().chunks) == (1, 2)


# Nearer the query than every filled chunk but the nearest
PARTLY_TOWARD_QUERY = [0.9] * (EMBEDDING_DIMENSION // 2) + [0.1] * (EMBEDDING_DIMENSION // 2)


def a_file_added_and_the_last_written_removed(writer: IndexStore) -> None:
    """The file whose chunk was written last goes, so a store that gave its row's id again
    would give it to the chunk added."""
    added = function_chunk("added", "added.py", "def added(): pass")
    writer.write(
        update_of([added], removed_paths=[Path("far.py")]), vectors_of(PARTLY_TOWARD_QUERY)
    )


def a_file_removed_alone(writer: IndexStore) -> None:
    writer.write(
        update_of([], removed_paths=[Path("near.py")]), numpy.empty((0, 0), dtype=numpy.float32)
    )


@pytest.mark.parametrize(
    ("edit", "found"),
    [
        pytest.param(
            a_file_added_and_the_last_written_removed, ["near", "added", "keyword"], id="added"
        ),
        pytest.param(a_file_removed_alone, ["keyword", "far"], id="removed alone"),
    ],
)
def test_a_search_sees_what_another_store_on_its_directory_wrote_since_its_last_search(
    store_class: StoreClass,
    index_directory: IndexDirectory,
    filled_store: IndexStore,
    edit: Callable[[IndexStore], None],
    found: list[str],
):
    """Another session's server writes the same index: what a reader keeps of it between
    searches never hides the change."""
    nearest(filled_store)

    edit(store_class(index_directory))

    assert [candidate.chunk.name for candidate in nearest(filled_store)] == found


def test_a_search_after_its_index_was_removed_and_built_again_sees_the_new_one_alone(
    store_class: StoreClass, index_directory: IndexDirectory, filled_store: IndexStore
):
    """A cache cleared, or another session's sweep, while a session held the old index: the
    new one starts its rows again, and what the session kept of the old must not stand."""
    nearest(filled_store)
    shutil.rmtree(index_directory.path)
    rebuilt = [
        function_chunk(f"new_{number}", f"new_{number}.py", "def f(): pass")
        for number in range(len(FILLED_CHUNK_NAMES) + 1)
    ]

    store_class(index_directory).write(
        update_of(rebuilt, removed_paths=[]),
        vectors_of(AWAY_FROM_QUERY, TOWARD_QUERY, ALTERNATING, PARTLY_TOWARD_QUERY),
    )

    assert [candidate.chunk.name for candidate in nearest(filled_store)] == [
        "new_1",
        "new_3",
        "new_2",
        "new_0",
    ]


def test_vectors_of_another_width_replace_the_index_and_leave_the_other_files_unrecorded(
    filled_store: IndexStore,
):
    """A model replaced under a name nothing pins gives vectors of another width: the index starts
    again from the files written, and the next run indexes the others, unrecorded now."""
    nearest(filled_store)
    edited = function_chunk("near", "near.py", "def near(): return elapsed_time()")

    filled_store.write(update_of([edited], removed_paths=[]), vectors_of(TOWARD_QUERY * 2))

    wide_query = numpy.full(2 * EMBEDDING_DIMENSION, 1.0, dtype=numpy.float32)
    assert filled_store.counts() == IndexCounts(files=1, chunks=1)
    assert [candidate.chunk.name for candidate in nearest(filled_store, wide_query)] == ["near"]


def test_a_write_records_what_built_the_index(store_class: StoreClass, store: IndexStore):
    before = store.manifest()
    store.write(update_of([], removed_paths=[]), numpy.empty((0, 0), dtype=numpy.float32))
    after = store.manifest()

    assert (before, after.identity, after.schema_version) == (
        None,
        IDENTITY,
        store_class.LAYOUT.schema_version,
    )


def write_own_index(store: IndexStore) -> None:
    chunk = function_chunk("own", "own.py", "def own(): pass")
    store.write(update_of([chunk], removed_paths=[]), vectors_of(TOWARD_QUERY))


def count_own_index(store: IndexStore) -> None:
    store.counts()


def list_own_files(store: IndexStore) -> None:
    store.indexed_files()


def search_own_index(store: IndexStore) -> None:
    store.search_candidates(QUERY, query_of("own"), LIMIT)


def read_own_vectors(store: IndexStore) -> None:
    store.stored_vectors(["own.py#own"])


@pytest.mark.parametrize(
    "use_own_index",
    [
        pytest.param(write_own_index, id="written"),
        pytest.param(count_own_index, id="counted"),
        pytest.param(list_own_files, id="its files listed"),
        pytest.param(search_own_index, id="searched"),
        pytest.param(read_own_vectors, id="its vectors read"),
    ],
)
def test_an_index_nobody_used_for_a_week_is_removed_when_another_is_used(
    store_class: StoreClass,
    store: IndexStore,
    tmp_path: Path,
    use_own_index: Callable[[IndexStore], None],
):
    """A week after both were last used, a session of this store's index uses it: every read
    and write marks the index as used, and with it removes the project's unused ones."""
    write_own_index(store)
    other = index_directory_of(store_class, tmp_path / "index", OTHER_IDENTITY)
    write_own_index(store_class(other))
    set_mtime_of_tree(tmp_path / "index", LONG_AGO_SECONDS)

    use_own_index(store)

    assert not other.exists()


def set_mtime_of_tree(directory: Path, seconds: int) -> None:
    """Set the modification time of a directory and of everything under it."""
    for path in [directory, *directory.rglob("*")]:
        os.utime(path, (seconds, seconds))


def test_a_chunk_written_again_under_its_id_stands_where_it_now_is(
    filled_store: IndexStore,
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


def test_the_vectors_of_stored_chunks_come_back_by_their_ids(filled_store: IndexStore):
    found = filled_store.stored_vectors(["near.py#near", "nowhere.py#missing"])

    assert list(found) == ["near.py#near"]
    assert numpy.array_equal(found["near.py#near"], vectors_of(TOWARD_QUERY)[0])
