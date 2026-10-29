"""Search quality: where the answer to each question lands in what `search_code` returns.

Two scores for each kind of question, as issue #3 measured them: whether the right file
comes back, and whether the right chunk does. Each as hits@1, hits@3 and mean reciprocal
rank, apart for the questions held out, and the chunk score again for the questions that
hold a word of an answer's name or heading and those that do not, as the keyword channel
finds the first more easily. Then whether a search for a bare name of the code finds a
chunk of that name, and how many results questions no corpus answers get, which a good
search keeps at none.
"""

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from evals.corpus import (
    TOP_LEVEL,
    Answer,
    Corpus,
    OffTopicQuestions,
    Query,
    QuestionKind,
    missing_answer_paths,
)
from evals.server import SearchHit, ServerConnection
from semantic_code_mcp.app import build_chunker
from semantic_code_mcp.config import Settings
from semantic_code_mcp.discovery import SourceDiscovery
from semantic_code_mcp.domain.models import Chunk, ChunkKind
from semantic_code_mcp.storage.search_text import words_of

# How many results each question asks for, as an agent's default call does
RESULTS_PER_QUERY = 10
FIRST_RANK = 1
TOP_ONE = 1
TOP_THREE = 3
MRR_DECIMALS = 3
# A bare name is asked of every this many names, in their order
NAME_SAMPLE_STEP = 5
# What code defines under a name an agent could search for
NAMED_KINDS = frozenset({ChunkKind.function, ChunkKind.method, ChunkKind.klass})
# How a test is told by its path or by what it sits in: a directory or a module of tests,
# a file named for them
TEST_DIRECTORIES = frozenset({"test", "tests"})
TEST_FILE_PREFIX = "test_"
TEST_FILE_SUFFIX = "_test"
RUST_PATH_SEPARATOR = "::"
# Chunks of one name under more parents than this need an answer that names its parent
ONE_PARENT = 1
# How many hits a name's rank keeps, to read what came instead of its chunk
NAME_HITS_KEPT = 3
# Words a question and an answer's name or heading share, compared by a crude stem: a
# plural's `s` dropped, then the first five letters; words shorter than four letters skipped
PLURAL_ENDING = "s"
STEM_LENGTH = 5
SHORTEST_WORD = 4
# The groups of wording, in the order the report gives them: questions that hold an
# answer's words first
WORDINGS = (True, False)
# A discovery for the harness alone: no home directory holds the corpora's checkouts
HOME_OF_NO_CORPUS = Path("/home-of-no-corpus")


class Hit(BaseModel):
    """A result as a key is compared with it: its file from the corpus root, its chunk's
    name, and its parent, TOP_LEVEL for a chunk that sits in nothing."""

    model_config = ConfigDict(frozen=True)

    path: str
    symbol: str
    parent: str


class QueryRanks(BaseModel):
    """Where one question's answer landed: the first matching file and the first matching chunk."""

    model_config = ConfigDict(frozen=True)

    query: str
    kind: QuestionKind
    # Whether the question holds a word of an answer's name or heading
    shares_answer_words: bool
    file_rank: int | None
    chunk_rank: int | None
    # What came back, best first, so a miss can be read without running the model again
    hits: tuple[Hit, ...]


class NameRank(BaseModel):
    """Where the first chunk of a name landed, the name alone asked, and the first hits,
    which a miss can be read from."""

    model_config = ConfigDict(frozen=True)

    name: str
    rank: int | None
    hits: tuple[Hit, ...]


class RankScores(BaseModel):
    """Hits at one and three and the mean reciprocal rank of one kind of match."""

    model_config = ConfigDict(frozen=True)

    hits_at_1: int
    hits_at_3: int
    mean_reciprocal_rank: float


class KindScores(BaseModel):
    """Both scores of the questions of one kind."""

    model_config = ConfigDict(frozen=True)

    kind: QuestionKind
    queries: int
    file: RankScores
    chunk: RankScores


class WordingScores(BaseModel):
    """The chunk score of the questions of one kind that do, or do not, hold a word of an
    answer's name or heading."""

    model_config = ConfigDict(frozen=True)

    kind: QuestionKind
    shares_answer_words: bool
    queries: int
    chunk: RankScores


class NameScores(BaseModel):
    """How well a search for a bare name finds the chunk that defines it."""

    model_config = ConfigDict(frozen=True)

    names: int
    chunk: RankScores


class OffTopicResults(BaseModel):
    """What the questions no corpus answers got, all of each list together."""

    model_config = ConfigDict(frozen=True)

    far_questions: int
    far_results: int
    near_questions: int
    near_results: int


class CorpusQuality(BaseModel):
    """The quality of one corpus: the scores, and every question's ranks behind them."""

    model_config = ConfigDict(frozen=True)

    corpus: str
    kinds: tuple[KindScores, ...]
    wordings: tuple[WordingScores, ...]
    held_out: tuple[KindScores, ...]
    names: NameScores
    off_topic: OffTopicResults
    ranks: tuple[QueryRanks, ...]
    held_out_ranks: tuple[QueryRanks, ...]
    name_ranks: tuple[NameRank, ...]


async def measure_quality(
    corpus: Corpus,
    working_tree: Path,
    off_topic: OffTopicQuestions,
    connection: ServerConnection,
) -> CorpusQuality:
    """Index a corpus, ask every question and every sampled name, and score where the
    answers landed; then ask the off-topic questions and count what came back."""
    missing_files = missing_answer_paths(corpus, working_tree)
    if missing_files:
        message = f"{corpus.corpus.name}: answers name files the checkout lacks: {missing_files}"
        raise CorpusMistakeError(message)
    chunks = chunks_of(working_tree)
    chunks_by_path = chunks_by_file(chunks)
    missing_chunks = missing_answer_chunks(corpus, chunks_by_path)
    if missing_chunks:
        message = (
            f"{corpus.corpus.name}: answers name chunks the server does not index: {missing_chunks}"
        )
        raise CorpusMistakeError(message)
    ambiguous = ambiguous_answers(corpus, chunks_by_path)
    if ambiguous:
        message = f"{corpus.corpus.name}: answers must name a parent of their chunk: {ambiguous}"
        raise CorpusMistakeError(message)
    await connection.index(working_tree)
    ranks = [await rank_query(query, working_tree, connection) for query in corpus.queries]
    held_out_ranks = [
        await rank_query(query, working_tree, connection) for query in corpus.held_out
    ]
    names = sample_names(chunks)
    name_ranks = [await rank_name(name, working_tree, connection) for name in names]
    return CorpusQuality(
        corpus=corpus.corpus.name,
        kinds=scores_by_kind(ranks),
        wordings=scores_by_wording(ranks),
        held_out=scores_by_kind(held_out_ranks),
        names=NameScores(
            names=len(name_ranks), chunk=score([name_rank.rank for name_rank in name_ranks])
        ),
        off_topic=OffTopicResults(
            far_questions=len(off_topic.far),
            far_results=await results_for(off_topic.far, working_tree, connection),
            near_questions=len(off_topic.near),
            near_results=await results_for(off_topic.near, working_tree, connection),
        ),
        ranks=tuple(ranks),
        held_out_ranks=tuple(held_out_ranks),
        name_ranks=tuple(name_ranks),
    )


async def rank_query(query: Query, working_tree: Path, connection: ServerConnection) -> QueryRanks:
    """Ask one question and find where its answers landed."""
    answer = await connection.search(working_tree, query.text, RESULTS_PER_QUERY)
    relative_hits = [relative_hit(hit, working_tree) for hit in answer.results]
    answer_paths = {expected.path for expected in query.answers}
    return QueryRanks(
        query=query.text,
        kind=query.kind,
        shares_answer_words=shares_answer_words(query.text, query.answers),
        file_rank=first_rank([hit.path in answer_paths for hit in relative_hits]),
        chunk_rank=first_rank(
            [any(hit_answers(hit, expected) for expected in query.answers) for hit in relative_hits]
        ),
        hits=tuple(relative_hits),
    )


async def rank_name(name: str, working_tree: Path, connection: ServerConnection) -> NameRank:
    """Ask a bare name and find where the first chunk of that name landed, in any file."""
    answer = await connection.search(working_tree, name, RESULTS_PER_QUERY)
    relative_hits = [relative_hit(hit, working_tree) for hit in answer.results]
    return NameRank(
        name=name,
        rank=first_rank([hit.symbol == name for hit in relative_hits]),
        hits=tuple(relative_hits[:NAME_HITS_KEPT]),
    )


async def results_for(
    questions: Sequence[str], working_tree: Path, connection: ServerConnection
) -> int:
    """How many results the questions got, all of them together."""
    answers = [
        await connection.search(working_tree, question, RESULTS_PER_QUERY) for question in questions
    ]
    return sum(len(answer.results) for answer in answers)


def scores_by_kind(ranks: Sequence[QueryRanks]) -> tuple[KindScores, ...]:
    """Both scores of each kind of question the ranks hold, in the order of the kinds."""
    kinds_asked = [
        kind for kind in QuestionKind if any(query_ranks.kind == kind for query_ranks in ranks)
    ]
    return tuple(
        kind_scores(kind, [query_ranks for query_ranks in ranks if query_ranks.kind == kind])
        for kind in kinds_asked
    )


def scores_by_wording(ranks: Sequence[QueryRanks]) -> tuple[WordingScores, ...]:
    """The chunk score of each kind's questions that hold an answer's words, then of those
    that do not, for each group the ranks hold."""
    return tuple(
        wording_scores(kind, ranks, shares=shares)
        for kind in QuestionKind
        for shares in WORDINGS
        if any(is_in_group(query_ranks, kind, shares=shares) for query_ranks in ranks)
    )


def wording_scores(
    kind: QuestionKind, ranks: Sequence[QueryRanks], *, shares: bool
) -> WordingScores:
    """The chunk score of one kind's questions that do, or do not, hold an answer's words."""
    group = [
        query_ranks.chunk_rank
        for query_ranks in ranks
        if is_in_group(query_ranks, kind, shares=shares)
    ]
    return WordingScores(
        kind=kind, shares_answer_words=shares, queries=len(group), chunk=score(group)
    )


def is_in_group(query_ranks: QueryRanks, kind: QuestionKind, *, shares: bool) -> bool:
    """Whether a question is of the kind, and holds an answer's words or not, as asked."""
    return query_ranks.kind == kind and query_ranks.shares_answer_words == shares


def shares_answer_words(question: str, answers: Sequence[Answer]) -> bool:
    """Whether the question holds a word of an answer's name or heading, by stem."""
    asked = stems_of(question)
    return any(asked & stems_of(answer.symbol) for answer in answers)


def stems_of(text: str) -> set[str]:
    """The crude stems of a text's words, identifiers split: `pattern_has_uppercase_char`
    gives patte, upper (from uppercase), char."""
    words = [word.lower().removesuffix(PLURAL_ENDING) for word in words_of(text)]
    return {word[:STEM_LENGTH] for word in words if len(word) >= SHORTEST_WORD}


def kind_scores(kind: QuestionKind, ranks: Sequence[QueryRanks]) -> KindScores:
    """Both scores of the questions of one kind."""
    return KindScores(
        kind=kind,
        queries=len(ranks),
        file=score([query_ranks.file_rank for query_ranks in ranks]),
        chunk=score([query_ranks.chunk_rank for query_ranks in ranks]),
    )


def sample_names(chunks: Sequence[Chunk]) -> list[str]:
    """Every NAME_SAMPLE_STEP-th name an agent could search for, in their order: of a
    function, a method or a class outside the tests, an identifier (not Rust's `Display
    for Path`); a name several files define is one name, which a chunk of any of them
    answers."""
    named = {
        chunk.name
        for chunk in chunks
        if chunk.kind in NAMED_KINDS and chunk.name.isidentifier() and not is_test(chunk)
    }
    return sorted(named)[::NAME_SAMPLE_STEP]


def is_test(chunk: Chunk) -> bool:
    """Whether a chunk is a test's, by a directory or a file named for tests, or a Rust
    module of tests it sits in."""
    stem = chunk.path.stem
    enclosing = (chunk.parent or "").split(RUST_PATH_SEPARATOR)
    return (
        bool(TEST_DIRECTORIES & set(chunk.path.parts[:-1]))
        or stem.startswith(TEST_FILE_PREFIX)
        or stem.endswith(TEST_FILE_SUFFIX)
        or bool(TEST_DIRECTORIES & set(enclosing))
    )


def chunks_of(working_tree: Path) -> list[Chunk]:
    """Every chunk of the files the server indexes in the checkout, by its own discovery
    with the default settings and its chunker, without the model."""
    chunker = build_chunker()
    settings = Settings(cache_dir=working_tree)
    discovery = SourceDiscovery(
        extensions=chunker.supported_extensions,
        ignore_patterns=settings.ignore_patterns,
        use_gitignore=settings.use_gitignore,
        git_timeout_seconds=settings.git_timeout_seconds,
        home_directory=HOME_OF_NO_CORPUS,
    )
    return [
        chunk
        for path in discovery.discover(working_tree)
        for chunk in chunker.chunk_file(working_tree, Path(path))
    ]


def chunks_by_file(chunks: Sequence[Chunk]) -> dict[str, list[Chunk]]:
    """The chunks of each file, by its path from the corpus root."""
    grouped: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.path.as_posix()].append(chunk)
    return dict(grouped)


def missing_answer_chunks(corpus: Corpus, chunks_by_path: dict[str, list[Chunk]]) -> list[Answer]:
    """Answers that no chunk of their file, as the server discovers and chunks it, matches.

    Answers name chunks as the chunker names them, so a mistyped name, or a file the server
    never indexes, would score as a miss forever; this asks the server's own parts, without
    the model.
    """
    return sorted(
        (
            answer
            for answer in every_answer(corpus)
            if not is_held(answer, chunks_by_path.get(answer.path, []))
        ),
        key=answer_order,
    )


def ambiguous_answers(corpus: Corpus, chunks_by_path: dict[str, list[Chunk]]) -> list[Answer]:
    """Answers that name no parent where chunks of several parents share their name; chunks
    of one name under one parent (overloads, a Rust type and its impl) all count."""
    return sorted(
        (
            answer
            for answer in every_answer(corpus)
            if is_ambiguous(answer, chunks_by_path.get(answer.path, []))
        ),
        key=answer_order,
    )


def is_held(answer: Answer, chunks: Sequence[Chunk]) -> bool:
    """Whether a chunk of the answer's file is the answer."""
    return any(hit_answers(chunk_hit(chunk), answer) for chunk in chunks)


def is_ambiguous(answer: Answer, chunks: Sequence[Chunk]) -> bool:
    """Whether an answer names no parent though chunks of several parents in its file share
    its name, so that a key judged for one would count them all."""
    parents = {chunk.parent for chunk in chunks if chunk.name == answer.symbol}
    return answer.parent is None and len(parents) > ONE_PARENT


def hit_answers(hit: Hit, expected: Answer) -> bool:
    """Whether a hit is the expected answer: its file and name, and its parent where the
    answer names one."""
    return (
        hit.path == expected.path
        and hit.symbol == expected.symbol
        and expected.parent in (None, hit.parent)
    )


def every_answer(corpus: Corpus) -> set[Answer]:
    return {answer for query in corpus.every_query() for answer in query.answers}


def chunk_hit(chunk: Chunk) -> Hit:
    """A chunk as a hit of it is compared with a key."""
    return Hit(path=chunk.path.as_posix(), symbol=chunk.name, parent=chunk.parent or TOP_LEVEL)


def answer_order(answer: Answer) -> tuple[str, str, str]:
    """The order mistakes are reported in; a key of any parent sorts beside the top level."""
    return (answer.path, answer.symbol, answer.parent or TOP_LEVEL)


def relative_hit(hit: SearchHit, working_tree: Path) -> Hit:
    """A result as a key is compared with it: its path from the corpus root, its chunk name,
    and its parent, TOP_LEVEL where it sits in nothing."""
    path = Path(hit.file_path).resolve().relative_to(working_tree.resolve()).as_posix()
    return Hit(path=path, symbol=hit.name, parent=hit.parent or TOP_LEVEL)


def first_rank(matches: Sequence[bool]) -> int | None:
    """The rank of the first matching result, counted from one; None when nothing matched."""
    return next((rank for rank, matched in enumerate(matches, start=FIRST_RANK) if matched), None)


def score(ranks: Sequence[int | None]) -> RankScores:
    """Hits at one and three and the mean reciprocal rank of a list of ranks.

    A question whose answer never came back counts as a miss and adds nothing to the mean.
    """
    found = [rank for rank in ranks if rank is not None]
    reciprocal_rank_sum = sum(1 / rank for rank in found)
    return RankScores(
        hits_at_1=sum(rank <= TOP_ONE for rank in found),
        hits_at_3=sum(rank <= TOP_THREE for rank in found),
        mean_reciprocal_rank=round(reciprocal_rank_sum / len(ranks), MRR_DECIMALS),
    )


class CorpusMistakeError(ValueError):
    """A corpus file names an answer that cannot be right."""
