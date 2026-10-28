"""Tests for the harness: the scoring of search quality, and the edits of the speed run."""

from pathlib import Path

import pytest
from evals.__main__ import DEFAULT_EDITS, parse_arguments
from evals.corpus import TOP_LEVEL, Answer, Corpus, CorpusSource, Query, QuestionKind
from evals.quality import (
    Hit,
    KindScores,
    QueryRanks,
    RankScores,
    chunks_by_file,
    first_rank,
    hit_answers,
    is_ambiguous,
    is_held,
    missing_answer_chunks,
    relative_hit,
    sample_names,
    score,
    scores_by_kind,
    shares_answer_words,
)
from evals.server import (
    BY_MEANING_VARIABLE,
    INDEX_STORE_VARIABLE,
    SearchHit,
    ServerOptions,
    server_environment,
)
from evals.speed import Edit, edit_text, edits_in_turn

from semantic_code_mcp.app import build_chunker
from semantic_code_mcp.domain.models import Chunk, ChunkKind, Language, StoreKind
from semantic_code_mcp.storage.lancedb import UPKEEPS_PER_COMPACTION


def test_first_rank_counts_from_one():
    assert first_rank([False, True, True]) == 2


def test_first_rank_is_none_when_nothing_matched():
    assert first_rank([False, False]) is None


def test_score_counts_hits_and_averages_reciprocal_ranks_over_every_question():
    scores = score([1, 3, None, 2])

    assert scores == RankScores(hits_at_1=1, hits_at_3=3, mean_reciprocal_rank=0.458)


def ranks_of(kind: QuestionKind, file_rank: int | None, chunk_rank: int | None) -> QueryRanks:
    return QueryRanks(
        query="a question",
        kind=kind,
        shares_answer_words=False,
        file_rank=file_rank,
        chunk_rank=chunk_rank,
        hits=(),
    )


def test_scores_are_kept_apart_for_each_kind_of_question():
    ranks = [
        ranks_of(QuestionKind.code, 1, 2),
        ranks_of(QuestionKind.documentation, 4, None),
        ranks_of(QuestionKind.code, None, None),
    ]

    assert scores_by_kind(ranks) == (
        KindScores(
            kind=QuestionKind.code,
            queries=2,
            file=RankScores(hits_at_1=1, hits_at_3=1, mean_reciprocal_rank=0.5),
            chunk=RankScores(hits_at_1=0, hits_at_3=1, mean_reciprocal_rank=0.25),
        ),
        KindScores(
            kind=QuestionKind.documentation,
            queries=1,
            file=RankScores(hits_at_1=0, hits_at_3=0, mean_reciprocal_rank=0.25),
            chunk=RankScores(hits_at_1=0, hits_at_3=0, mean_reciprocal_rank=0.0),
        ),
    )


def defined(
    name: str, path: str, kind: ChunkKind = ChunkKind.function, parent: str | None = None
) -> Chunk:
    return Chunk(
        chunk_id=f"{path}:{name}",
        path=Path(path),
        language=Language.python,
        kind=kind,
        name=name,
        parent=parent,
        line_start=1,
        line_end=1,
        content=f"def {name}(): ...",
    )


def test_a_bare_name_is_asked_of_every_fifth_name_code_outside_its_tests_defines():
    """A name of several files is asked once; a test's name, a name that is no identifier
    and a heading are not asked. Each such name here sorts where the sample would take it."""
    chunks = [
        *(defined(f"name_{number:02}", "a.py") for number in range(12)),
        defined("name_03", "a.py", ChunkKind.method),
        defined("aaa", "a.py"),
        defined("aaa", "b.py"),
        defined("aab", "tests/helpers.py"),
        defined("aac", "src/lib.rs", parent="outer::tests"),
        defined("aad", "src/a_test.py"),
        defined("aae", "src/test_a.py"),
        defined("Display for Path", "a.py", ChunkKind.klass),
        defined("Install", "a.md", ChunkKind.section),
    ]

    assert sample_names(chunks) == ["aaa", "name_04", "name_09"]


@pytest.mark.parametrize(
    ("question", "answer", "shares"),
    [
        pytest.param(
            "check if a regex pattern has an uppercase letter",
            Answer(path="src/regex_helper.rs", symbol="pattern_has_uppercase_char"),
            True,
            id="a word of the name",
        ),
        pytest.param(
            "make the search list hidden files",
            Answer(path="README.md", symbol="Hidden and ignored files"),
            True,
            id="a word of the heading, plural",
        ),
        pytest.param(
            "list the files",
            Answer(path="src/reading.py", symbol="file_reader"),
            True,
            id="a word of the name, its plural",
        ),
        pytest.param(
            "parse a size filter such as +10k",
            Answer(path="src/filter/size.rs", symbol="from_string"),
            False,
            id="no word",
        ),
    ],
)
def test_a_question_shares_its_answer_s_words_when_a_stem_is_both_s(
    question: str, answer: Answer, shares: bool
):
    assert shares_answer_words(question, (answer,)) is shares


@pytest.mark.parametrize(
    ("answer", "hit", "matches"),
    [
        pytest.param(
            Answer(path="types.py", symbol="convert"),
            Hit(path="types.py", symbol="convert", parent="File"),
            True,
            id="no parent named, any parent",
        ),
        pytest.param(
            Answer(path="types.py", symbol="convert", parent="BoolParamType"),
            Hit(path="types.py", symbol="convert", parent="BoolParamType"),
            True,
            id="the parent named",
        ),
        pytest.param(
            Answer(path="types.py", symbol="convert", parent="BoolParamType"),
            Hit(path="types.py", symbol="convert", parent="File"),
            False,
            id="another parent",
        ),
        pytest.param(
            Answer(path="walk.rs", symbol="scan", parent=TOP_LEVEL),
            Hit(path="walk.rs", symbol="scan", parent=TOP_LEVEL),
            True,
            id="the top level named, a top-level hit",
        ),
        pytest.param(
            Answer(path="walk.rs", symbol="scan", parent=TOP_LEVEL),
            Hit(path="walk.rs", symbol="scan", parent="WorkerState"),
            False,
            id="the top level named, a method's hit",
        ),
        pytest.param(
            Answer(path="types.py", symbol="convert"),
            Hit(path="types.py", symbol="to_info_dict", parent=TOP_LEVEL),
            False,
            id="another name",
        ),
        pytest.param(
            Answer(path="types.py", symbol="convert"),
            Hit(path="core.py", symbol="convert", parent=TOP_LEVEL),
            False,
            id="another file",
        ),
    ],
)
def test_a_hit_answers_when_its_file_name_and_any_parent_named_are_the_answer_s(
    answer: Answer, hit: Hit, matches: bool
):
    assert hit_answers(hit, answer) is matches


@pytest.mark.parametrize(
    ("answer", "ambiguous"),
    [
        pytest.param(Answer(path="types.py", symbol="convert"), True, id="several parents"),
        pytest.param(
            Answer(path="types.py", symbol="convert", parent="File"), False, id="its parent named"
        ),
        pytest.param(Answer(path="types.py", symbol="prompt"), False, id="overloads of one parent"),
    ],
)
def test_an_answer_names_its_parent_where_chunks_of_several_share_its_name(
    answer: Answer, ambiguous: bool
):
    """Else a key judged for one chunk would count every chunk of that name in the file."""
    chunks = [
        defined("convert", "types.py", ChunkKind.method, parent="BoolParamType"),
        defined("convert", "types.py", ChunkKind.method, parent="File"),
        defined("prompt", "types.py"),
        defined("prompt", "types.py"),
    ]

    assert is_ambiguous(answer, chunks) is ambiguous


@pytest.mark.parametrize(
    ("answer", "held"),
    [
        pytest.param(Answer(path="types.py", symbol="convert", parent="File"), True, id="held"),
        pytest.param(
            Answer(path="types.py", symbol="convert", parent="Path"), False, id="a parent none has"
        ),
        pytest.param(
            Answer(path="types.py", symbol="open_url", parent=TOP_LEVEL),
            True,
            id="the top level named",
        ),
    ],
)
def test_an_answer_is_held_when_a_chunk_of_its_file_is_it(answer: Answer, held: bool):
    """A mistyped parent would otherwise score as a miss forever."""
    chunks = [
        defined("convert", "types.py", ChunkKind.method, parent="File"),
        defined("open_url", "types.py"),
    ]

    assert is_held(answer, chunks) is held


def test_an_answer_in_a_file_the_server_does_not_index_is_a_mistake():
    """It could never be hit, and would score as a miss forever."""
    indexed = Answer(path="types.py", symbol="convert")
    unindexed = Answer(path="build/types.py", symbol="convert")
    question = Query(text="convert a value", kind=QuestionKind.code, answers=(indexed, unindexed))
    corpus = Corpus(
        corpus=CorpusSource(name="click", repository="click.git", commit="abc"),
        queries=(question,),
        held_out=(),
    )

    mistakes = missing_answer_chunks(corpus, chunks_by_file([defined("convert", "types.py")]))

    assert mistakes == [unindexed]


@pytest.mark.parametrize(
    ("parent", "named"),
    [
        pytest.param("WorkerState", "WorkerState", id="a method"),
        pytest.param(None, TOP_LEVEL, id="the top level"),
    ],
)
def test_relative_hit_names_the_file_from_the_corpus_root_and_the_parent(
    tmp_path: Path, parent: str | None, named: str
):
    hit = SearchHit(
        file_path=str(tmp_path / "src" / "walk.rs"),
        name="scan",
        kind="function",
        parent=parent,
        line_start=1,
        line_end=9,
    )

    assert relative_hit(hit, tmp_path) == Hit(path="src/walk.rs", symbol="scan", parent=named)


PYTHON_SOURCE = '''"""A module."""


def first():
    return 1
'''
RUST_SOURCE = """/// A function.
fn first() -> u32 {
    1
}
"""
MARKDOWN_SOURCE = """# Guide

Intro.

## Install

Steps.
"""


@pytest.mark.parametrize(
    ("name", "source"),
    [
        pytest.param("module.py", PYTHON_SOURCE, id="python"),
        pytest.param("lib.rs", RUST_SOURCE, id="rust"),
        pytest.param("guide.md", MARKDOWN_SOURCE, id="markdown"),
    ],
)
def test_an_edit_of_the_speed_run_adds_one_chunk_and_changes_no_other(
    tmp_path: Path, name: str, source: str
):
    """So the search after it re-embeds one chunk once vectors are reused."""
    edited = tmp_path / name
    edited.write_text(source)
    before = build_chunker().chunk_file(tmp_path, Path(name))

    edited.write_text(source + edit_text(edited, 1))
    after = build_chunker().chunk_file(tmp_path, Path(name))

    assert [chunk.chunk_id for chunk in after][:-1] == [chunk.chunk_id for chunk in before]


def question_answered_in(path: str, text: str) -> Query:
    return Query(
        text=text,
        kind=QuestionKind.code,
        answers=(Answer(path=path, symbol="answer"),),
    )


def test_the_speed_run_edits_the_files_that_answer_its_questions_in_turn():
    """Edits of one file never show upkeep compacting what another edit left; each edit asks
    the first question its file answers."""
    queries = [
        question_answered_in("first.py", "first question"),
        question_answered_in("first.py", "second question"),
        question_answered_in("second.py", "third question"),
        question_answered_in("guide.md", "fourth question"),
    ]

    edits = edits_in_turn(queries, 5)

    assert edits == [
        Edit(path=Path("first.py"), question="first question"),
        Edit(path=Path("second.py"), question="third question"),
        Edit(path=Path("guide.md"), question="fourth question"),
        Edit(path=Path("first.py"), question="first question"),
        Edit(path=Path("second.py"), question="third question"),
    ]


def test_the_speed_run_edits_enough_for_its_last_edit_to_meet_a_compaction():
    """The slowest search after an edit shows LanceDB's compaction only if the run reaches it."""
    assert DEFAULT_EDITS == UPKEEPS_PER_COMPACTION


@pytest.mark.parametrize(
    ("index_store", "store_given"),
    [
        pytest.param(None, None, id="the server's default"),
        pytest.param(StoreKind.sqlite, "sqlite", id="a store the run names"),
    ],
)
def test_a_server_of_the_harness_keeps_its_indexes_in_the_store_a_run_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    index_store: StoreKind | None,
    store_given: str | None,
):
    """The caller's own setting never reaches it."""
    monkeypatch.setenv(INDEX_STORE_VARIABLE, StoreKind.lancedb)

    environment = server_environment(
        tmp_path, ServerOptions(index_store=index_store, by_meaning=False)
    )

    assert environment.get(INDEX_STORE_VARIABLE) == store_given


@pytest.mark.parametrize(
    ("by_meaning", "setting_given"),
    [
        pytest.param(False, None, id="by words, the server's default"),
        pytest.param(True, "true", id="by meaning, as a run asks"),
    ],
)
def test_a_server_of_the_harness_searches_by_meaning_only_where_a_run_asks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, by_meaning: bool, setting_given: str | None
):
    """The caller's own setting never reaches it."""
    monkeypatch.setenv(BY_MEANING_VARIABLE, "true")

    environment = server_environment(
        tmp_path, ServerOptions(index_store=None, by_meaning=by_meaning)
    )

    assert environment.get(BY_MEANING_VARIABLE) == setting_given


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param(["quality", "--store", StoreKind.sqlite], id="quality in a store of its own"),
        pytest.param(["speed", "--store", StoreKind.sqlite], id="speed in a store of its own"),
        pytest.param(["quality", "--by-meaning"], id="quality by meaning"),
    ],
)
def test_a_run_other_than_the_server_s_default_is_never_saved_as_the_baseline(
    arguments: list[str],
):
    with pytest.raises(SystemExit):
        parse_arguments([*arguments, "--save-baseline"])
