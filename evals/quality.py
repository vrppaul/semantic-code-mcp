"""Search quality: where the answer to each question lands in what `search_code` returns.

Two scores for each kind of question, as issue #3 measured them: whether the right file
comes back, and whether the right chunk does. Each as hits@1, hits@3 and mean reciprocal
rank, apart for the questions held out. Then whether a search for a bare name finds the
chunk that defines it, and how many results questions no corpus answers get, which a good
search keeps at none.
"""

import subprocess  # nosec B404
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from evals.corpus import (
    Answer,
    Corpus,
    OffTopicQuestions,
    Query,
    QuestionKind,
    missing_answer_paths,
)
from evals.server import SearchHit, ServerConnection
from semantic_code_mcp.app import build_chunker
from semantic_code_mcp.domain.models import Chunk, ChunkKind

# How many results each question asks for, as an agent's default call does
RESULTS_PER_QUERY = 10
FIRST_RANK = 1
TOP_ONE = 1
TOP_THREE = 3
MRR_DECIMALS = 3
# A bare name is asked of every this many names, in the order of their files and names
NAME_SAMPLE_STEP = 5
# What code defines under a name an agent could search for
NAMED_KINDS = frozenset({ChunkKind.function, ChunkKind.method, ChunkKind.klass})
# A name defined in this many files has one chunk to find
ONE_FILE = 1
# `git ls-files -z` separates paths with NUL
GIT_PATH_SEPARATOR = "\0"
# How many hits a name's rank keeps, to read what came instead of its chunk
NAME_HITS_KEPT = 3


class QueryRanks(BaseModel):
    """Where one question's answer landed: the first matching file and the first matching chunk."""

    model_config = ConfigDict(frozen=True)

    query: str
    kind: QuestionKind
    file_rank: int | None
    chunk_rank: int | None
    # What came back, best first, so a miss can be read without running the model again
    hits: tuple[Answer, ...]


class NameRank(BaseModel):
    """Where the chunk that defines a name landed, the name alone asked, and the first
    hits, which a miss can be read from."""

    model_config = ConfigDict(frozen=True)

    name: Answer
    rank: int | None
    hits: tuple[Answer, ...]


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
    missing_chunks = missing_answer_chunks(corpus, working_tree)
    if missing_chunks:
        message = f"{corpus.corpus.name}: answers name chunks no file holds: {missing_chunks}"
        raise CorpusMistakeError(message)
    await connection.index(working_tree)
    ranks = [await rank_query(query, working_tree, connection) for query in corpus.queries]
    held_out_ranks = [
        await rank_query(query, working_tree, connection) for query in corpus.held_out
    ]
    names = sample_names(chunks_of(working_tree))
    name_ranks = [await rank_name(name, working_tree, connection) for name in names]
    return CorpusQuality(
        corpus=corpus.corpus.name,
        kinds=scores_by_kind(ranks),
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
        file_rank=first_rank([hit.path in answer_paths for hit in relative_hits]),
        chunk_rank=first_rank([hit in query.answers for hit in relative_hits]),
        hits=tuple(relative_hits),
    )


async def rank_name(name: Answer, working_tree: Path, connection: ServerConnection) -> NameRank:
    """Ask a bare name and find where the chunk that defines it landed."""
    answer = await connection.search(working_tree, name.symbol, RESULTS_PER_QUERY)
    relative_hits = [relative_hit(hit, working_tree) for hit in answer.results]
    return NameRank(
        name=name,
        rank=first_rank([hit == name for hit in relative_hits]),
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


def kind_scores(kind: QuestionKind, ranks: Sequence[QueryRanks]) -> KindScores:
    """Both scores of the questions of one kind."""
    return KindScores(
        kind=kind,
        queries=len(ranks),
        file=score([query_ranks.file_rank for query_ranks in ranks]),
        chunk=score([query_ranks.chunk_rank for query_ranks in ranks]),
    )


def sample_names(chunks: Sequence[Chunk]) -> list[Answer]:
    """Every NAME_SAMPLE_STEP-th name an agent could search for whose chunk is one file's:
    of a function, a method or a class, an identifier (not Rust's `Display for Path`),
    defined in no other file, as a name in many files has no one chunk to find."""
    named = {
        Answer(path=chunk.path.as_posix(), symbol=chunk.name)
        for chunk in chunks
        if chunk.kind in NAMED_KINDS and chunk.name.isidentifier()
    }
    files_defining = Counter(answer.symbol for answer in named)
    defined_once = sorted(
        (answer for answer in named if files_defining[answer.symbol] == ONE_FILE),
        key=lambda answer: (answer.path, answer.symbol),
    )
    return defined_once[::NAME_SAMPLE_STEP]


def chunks_of(working_tree: Path) -> list[Chunk]:
    """Every chunk the server's chunker gives the checkout's tracked files, without the model."""
    chunker = build_chunker()
    listing = subprocess.run(  # nosec B603, B607
        ["git", "-C", str(working_tree), "ls-files", "-z"],
        check=True,
        capture_output=True,
        text=True,
    )
    paths = [path for path in listing.stdout.split(GIT_PATH_SEPARATOR) if path]
    supported = tuple(chunker.supported_extensions)
    return [
        chunk
        for path in paths
        if path.endswith(supported)
        for chunk in chunker.chunk_file(working_tree, Path(path))
    ]


def missing_answer_chunks(corpus: Corpus, working_tree: Path) -> list[Answer]:
    """Answers whose chunk name the server's chunker never gives in that file.

    Answers name chunks as the chunker names them, so a mistyped name would score as a
    miss forever; this asks the chunker itself, without the model.
    """
    chunker = build_chunker()
    answers = {answer for query in corpus.every_query() for answer in query.answers}
    names_by_path = {
        path: {chunk.name for chunk in chunker.chunk_file(working_tree, Path(path))}
        for path in {answer.path for answer in answers}
    }
    return sorted(
        (answer for answer in answers if answer.symbol not in names_by_path[answer.path]),
        key=lambda answer: (answer.path, answer.symbol),
    )


def relative_hit(hit: SearchHit, working_tree: Path) -> Answer:
    """A hit as an answer is written: its path from the corpus root and its chunk name."""
    path = Path(hit.file_path).resolve().relative_to(working_tree.resolve()).as_posix()
    return Answer(path=path, symbol=hit.name)


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
