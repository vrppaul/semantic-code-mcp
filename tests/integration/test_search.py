"""Tests for search: by words, the chunks named as the query first; by meaning, what it keeps
of the store's hits; and the indexing and embedding it does first."""

import asyncio
import threading
from pathlib import Path

import pytest

from semantic_code_mcp.app import build_chunker
from semantic_code_mcp.config import Settings
from semantic_code_mcp.domain.models import (
    LOWEST_SIMILARITY,
    WHOLE_PROJECT,
    Candidate,
    Chunk,
    ChunkKind,
    Language,
    SearchCandidates,
    SearchQuery,
    SearchScope,
)
from semantic_code_mcp.services.progress import ProgressReporter
from semantic_code_mcp.services.search import SearchService, SimilarityFloors
from tests.fakes import FAKE_MODEL, FakeEmbedder, FakeIndexStore, ProgressRecorder
from tests.helpers import indexing_service, registry_of, vector_service
from tests.stores import NO_CANDIDATES

NO_PROGRESS = ProgressReporter(None)
LIMIT = 10
CLOSE = 0.9
FLOORS = SimilarityFloors(meaning_only=0.2, words_match=0.1)
# Too far in meaning for a hit no keyword matched, near enough for one whose words match
BETWEEN_THE_FLOORS = (FLOORS.meaning_only + FLOORS.words_match) / 2
BELOW_BOTH_FLOORS = FLOORS.words_match / 2
# A file no project of these tests holds
ELSEWHERE = "nowhere/old.py"
# The parent the chunk of `push` sits in, as each query of it names that parent
STACK_PARENT = {"Stack.push": "Stack", "stack::push": "Stack", "outer.Stack.push": "outer.Stack"}


def candidate(
    name: str, similarity: float | None, path: str = ELSEWHERE, parent: str | None = None
) -> Candidate:
    chunk = Chunk(
        chunk_id=f"{path}#{parent}#{name}",
        path=Path(path),
        language=Language.python,
        kind=ChunkKind.function,
        name=name,
        parent=parent,
        line_start=1,
        line_end=2,
        content=f"def {name}(): pass",
    )
    return Candidate(chunk=chunk, similarity=similarity)


def words_query(text: str) -> SearchQuery:
    """A query of the whole project, by words alone."""
    return SearchQuery(text=text, limit=LIMIT, scope=WHOLE_PROJECT, by_meaning=False)


def meaning_query(text: str) -> SearchQuery:
    """A query of the whole project, by meaning too."""
    return SearchQuery(text=text, limit=LIMIT, scope=WHOLE_PROJECT, by_meaning=True)


def found_by_vector(*candidates: Candidate) -> SearchCandidates:
    return SearchCandidates(by_vector=list(candidates), by_keyword=[])


def search_service(
    store: FakeIndexStore, settings: Settings, embedder: FakeEmbedder
) -> SearchService:
    return SearchService(
        store=store,
        embedder=embedder,
        model=FAKE_MODEL,
        indexing=indexing_service(store, settings, build_chunker()),
        vectors=vector_service(store, embedder),
        floors=FLOORS,
    )


def names_of(chunks: list[Chunk]) -> list[str]:
    return [chunk.name for chunk in chunks]


async def test_a_search_by_words_returns_what_the_words_ranked_without_the_model(
    test_settings: Settings, sample_project: Path
):
    """No floor holds a hit by words back, however far its meaning."""
    store = FakeIndexStore(
        SearchCandidates(
            by_vector=[],
            by_keyword=[candidate("first", BELOW_BOTH_FLOORS), candidate("second", None)],
        )
    )
    embedder = FakeEmbedder(loaded=False)
    service = search_service(store, test_settings, embedder)

    report = await service.search(words_query("anything"), sample_project, progress=NO_PROGRESS)

    assert (names_of(report.results), embedder.is_loaded) == (["first", "second"], False)


async def test_the_chunks_named_as_the_query_come_first_then_the_rest_by_words(
    test_settings: Settings, sample_project: Path
):
    """Those of the name the words ranked first in that order, then those they did not."""
    named_unranked = candidate("parse", CLOSE, "b.py")
    store = FakeIndexStore(
        SearchCandidates(
            by_vector=[named_unranked],
            by_keyword=[
                candidate("caller", CLOSE, "c.py"),
                candidate("parse", CLOSE, "a.py"),
                candidate("parse_config", CLOSE, "d.py"),
            ],
        )
    )
    service = search_service(store, test_settings, FakeEmbedder())

    report = await service.search(words_query(" Parse "), sample_project, progress=NO_PROGRESS)

    assert [(chunk.name, chunk.path.as_posix()) for chunk in report.results] == [
        ("parse", "a.py"),
        ("parse", "b.py"),
        ("caller", "c.py"),
        ("parse_config", "d.py"),
    ]


@pytest.mark.parametrize(
    "query",
    [
        pytest.param("Stack.push", id="as Python writes it"),
        pytest.param("stack::push", id="as Rust writes it, in another case"),
        pytest.param("outer.Stack.push", id="under a parent qualified itself"),
    ],
)
async def test_a_qualified_name_brings_the_chunks_of_that_name_under_that_parent_first(
    test_settings: Settings, sample_project: Path, query: str
):
    store = FakeIndexStore(
        SearchCandidates(
            by_vector=[],
            by_keyword=[
                candidate("push", CLOSE, "queue.py", parent="Queue"),
                candidate("caller", CLOSE, "caller.py"),
                candidate("push", CLOSE, "stack.py", parent=STACK_PARENT[query]),
            ],
        )
    )
    service = search_service(store, test_settings, FakeEmbedder())

    report = await service.search(words_query(query), sample_project, progress=NO_PROGRESS)

    assert [chunk.path.as_posix() for chunk in report.results] == [
        "stack.py",
        "queue.py",
        "caller.py",
    ]


async def test_a_name_holding_one_separator_under_a_parent_of_another_is_read_by_each(
    test_settings: Settings, sample_project: Path
):
    """`Setup > config.json` is a section named for a file under a heading."""
    store = FakeIndexStore(
        SearchCandidates(
            by_vector=[],
            by_keyword=[
                candidate("caller", CLOSE, "caller.py"),
                candidate("config.json", CLOSE, "guide.md", parent="Setup"),
            ],
        )
    )
    service = search_service(store, test_settings, FakeEmbedder())

    report = await service.search(
        words_query("Setup > config.json"), sample_project, progress=NO_PROGRESS
    )

    assert names_of(report.results) == ["config.json", "caller"]


@pytest.mark.parametrize(
    ("by_meaning", "chunks_embedded"),
    [pytest.param(False, 0, id="by words"), pytest.param(True, 6, id="by meaning")],
)
async def test_a_search_by_meaning_embeds_the_chunks_without_vectors_first(
    test_settings: Settings, sample_project: Path, *, by_meaning: bool, chunks_embedded: int
):
    store = FakeIndexStore(NO_CANDIDATES)
    service = search_service(store, test_settings, FakeEmbedder())
    query = SearchQuery(text="greet", limit=LIMIT, scope=WHOLE_PROJECT, by_meaning=by_meaning)

    await service.search(query, sample_project, progress=NO_PROGRESS)

    assert store.counts(FAKE_MODEL).chunks_with_vectors == chunks_embedded


async def test_a_hit_far_in_meaning_is_dropped_and_one_whose_words_match_further(
    test_settings: Settings, sample_project: Path
):
    """One without a vector yet stays: only its words found it."""
    named = candidate("named", BETWEEN_THE_FLOORS)
    store = FakeIndexStore(
        SearchCandidates(
            by_vector=[candidate("close", CLOSE), candidate("far", BETWEEN_THE_FLOORS), named],
            by_keyword=[
                named,
                candidate("word_by_chance", BELOW_BOTH_FLOORS),
                candidate("not_embedded", None),
            ],
        )
    )
    service = search_service(store, test_settings, FakeEmbedder())

    report = await service.search(meaning_query("anything"), sample_project, progress=NO_PROGRESS)

    assert (names_of(report.results), report.filtered_count) == (
        ["named", "close", "not_embedded"],
        2,
    )


async def test_a_hit_named_as_the_query_stays_however_far_in_meaning(
    test_settings: Settings, sample_project: Path
):
    """An agent that asks for an identifier it knows wants its definition, which a model
    may find far from the bare name; a name the query is only part of gets no such pass."""
    store = FakeIndexStore(
        SearchCandidates(
            by_vector=[],
            by_keyword=[
                candidate("main", BELOW_BOTH_FLOORS),
                candidate("main_loop", BELOW_BOTH_FLOORS),
            ],
        )
    )
    service = search_service(store, test_settings, FakeEmbedder())

    report = await service.search(meaning_query(" Main "), sample_project, progress=NO_PROGRESS)

    assert names_of(report.results) == ["main"]


@pytest.mark.parametrize(
    "by_meaning", [pytest.param(False, id="by words"), pytest.param(True, id="by meaning")]
)
async def test_no_more_hits_than_the_limit(
    test_settings: Settings, sample_project: Path, *, by_meaning: bool
):
    found = [candidate(f"function_{rank}", CLOSE) for rank in range(LIMIT)]
    store = FakeIndexStore(SearchCandidates(by_vector=found, by_keyword=found))
    service = search_service(store, test_settings, FakeEmbedder())

    report = await service.search(
        SearchQuery(text="anything", limit=3, scope=WHOLE_PROJECT, by_meaning=by_meaning),
        sample_project,
        progress=NO_PROGRESS,
    )

    assert len(report.results) == 3


async def test_hits_by_meaning_come_in_fused_order_whichever_file_they_are_in(
    test_settings: Settings, sample_project: Path
):
    store = FakeIndexStore(
        found_by_vector(
            candidate("best", CLOSE, "nowhere/a.py"),
            candidate("second", CLOSE, "nowhere/b.py"),
            candidate("third", CLOSE, "nowhere/a.py"),
        )
    )
    service = search_service(store, test_settings, FakeEmbedder())

    report = await service.search(meaning_query("anything"), sample_project, progress=NO_PROGRESS)

    assert names_of(report.results) == ["best", "second", "third"]


async def test_a_project_never_indexed_is_indexed_before_the_search(
    test_settings: Settings, sample_project: Path
):
    store = FakeIndexStore(NO_CANDIDATES)
    service = search_service(store, test_settings, FakeEmbedder())

    report = await service.search(words_query("anything"), sample_project, progress=NO_PROGRESS)

    assert report.index_report is not None
    assert report.index_report.files_indexed == 2


async def test_an_index_up_to_date_is_not_touched(test_settings: Settings, sample_project: Path):
    store = FakeIndexStore(NO_CANDIDATES)
    service = search_service(store, test_settings, FakeEmbedder())
    await service.search(words_query("first"), sample_project, progress=NO_PROGRESS)

    report = await service.search(words_query("second"), sample_project, progress=NO_PROGRESS)

    assert report.index_report is None


@pytest.mark.parametrize(
    "query",
    [
        pytest.param(words_query("anything"), id="by words"),
        pytest.param(meaning_query("anything"), id="by meaning"),
    ],
)
async def test_progress_of_a_search_that_indexes_only_moves_forward(
    test_settings: Settings, sample_project: Path, query: SearchQuery
):
    recorder = ProgressRecorder()
    service = search_service(FakeIndexStore(NO_CANDIDATES), test_settings, FakeEmbedder())

    await service.search(query, sample_project, progress=ProgressReporter(recorder))

    assert recorder.progress_values == sorted(recorder.progress_values)


async def test_a_project_with_nothing_indexed_is_answered_without_the_model(
    test_settings: Settings, tmp_path: Path
):
    """Even by meaning: nothing to embed, nothing to find."""
    empty_project = tmp_path / "empty"
    empty_project.mkdir()
    embedder = FakeEmbedder(loaded=False)
    service = search_service(FakeIndexStore(NO_CANDIDATES), test_settings, embedder)

    report = await service.search(meaning_query("anything"), empty_project, progress=NO_PROGRESS)

    assert (report.results, embedder.is_loaded) == ([], False)


async def test_a_fresh_server_loads_the_model_for_the_query_off_the_event_loop(
    test_settings: Settings, sample_project: Path
):
    """An index up to date and embedded, and a model not yet loaded: the first search by
    meaning after a restart."""
    store = FakeIndexStore(NO_CANDIDATES)
    await search_service(store, test_settings, FakeEmbedder()).search(
        meaning_query("anything"), sample_project, progress=NO_PROGRESS
    )
    fresh_embedder = FakeEmbedder(loaded=False)
    service = search_service(store, test_settings, fresh_embedder)

    await service.search(meaning_query("anything"), sample_project, progress=NO_PROGRESS)

    assert fresh_embedder.calling_threads
    assert threading.main_thread() not in fresh_embedder.calling_threads


async def test_two_first_searches_at_once_both_find_the_code(
    test_settings: Settings, sample_project: Path
):
    """The second waits for the first one's index run, finds its work done, and searches."""
    store = FakeIndexStore(found_by_vector(candidate("found", CLOSE)))
    service = search_service(store, test_settings, FakeEmbedder())

    reports = await asyncio.gather(
        service.search(meaning_query("anything"), sample_project, progress=NO_PROGRESS),
        service.search(meaning_query("anything"), sample_project, progress=NO_PROGRESS),
    )

    assert [names_of(report.results) for report in reports] == [["found"], ["found"]]


async def test_a_search_by_meaning_keeps_what_the_floors_of_the_settings_let_through(
    test_settings: Settings, sample_project: Path
):
    """Floors at the lowest similarity let every chunk of the project through, those the
    query's word matches among them."""
    settings = Settings(
        cache_dir=test_settings.cache_dir,
        min_similarity=LOWEST_SIMILARITY,
        min_similarity_when_words_match=LOWEST_SIMILARITY,
    )
    search = registry_of(settings, FakeEmbedder(), build_chunker()).services_for(sample_project)

    report = await search.search.search(
        meaning_query("greet"), sample_project, progress=NO_PROGRESS
    )

    assert len(report.results) == 6


async def test_a_search_hands_its_scope_to_the_store(test_settings: Settings, sample_project: Path):
    store = FakeIndexStore(NO_CANDIDATES)
    scope = SearchScope(
        kinds=[ChunkKind.klass],
        languages=[],
        path_glob="src/**",
        exclude_glob=None,
        directory=Path("src"),
    )
    service = search_service(store, test_settings, FakeEmbedder())

    await service.search(
        SearchQuery(text="anything", limit=LIMIT, scope=scope, by_meaning=False),
        sample_project,
        progress=NO_PROGRESS,
    )

    assert [query.scope for query in store.searched] == [scope]
