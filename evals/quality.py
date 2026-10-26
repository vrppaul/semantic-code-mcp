"""Search quality: where the answer to each question lands in what `search_code` returns.

Two scores per corpus, as issue #3 measured them: whether the right file comes back, and
whether the right chunk does. Each as hits@1, hits@3 and mean reciprocal rank. And how
many results questions no corpus answers get, which a good search keeps at none.
"""

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from evals.corpus import Answer, Corpus, Query, missing_answer_paths
from evals.server import SearchHit, ServerConnection
from semantic_code_mcp.app import build_chunker

# How many results each question asks for, as an agent's default call does
RESULTS_PER_QUERY = 10
FIRST_RANK = 1
TOP_ONE = 1
TOP_THREE = 3
MRR_DECIMALS = 3


class QueryRanks(BaseModel):
    """Where one question's answer landed: the first matching file and the first matching chunk."""

    model_config = ConfigDict(frozen=True)

    query: str
    file_rank: int | None
    chunk_rank: int | None
    # What came back, best first, so a miss can be read without running the model again
    hits: tuple[Answer, ...]


class RankScores(BaseModel):
    """Hits at one and three and the mean reciprocal rank of one kind of match."""

    model_config = ConfigDict(frozen=True)

    hits_at_1: int
    hits_at_3: int
    mean_reciprocal_rank: float


class CorpusQuality(BaseModel):
    """The quality of one corpus: both scores, and every question's ranks behind them."""

    model_config = ConfigDict(frozen=True)

    corpus: str
    queries: int
    file: RankScores
    chunk: RankScores
    ranks: tuple[QueryRanks, ...]
    off_topic_questions: int
    # What the off-topic questions got, all of them together
    off_topic_results: int


async def measure_quality(
    corpus: Corpus, working_tree: Path, off_topic: tuple[str, ...], connection: ServerConnection
) -> CorpusQuality:
    """Index a corpus, ask every question, and score where the answers landed; then ask
    the off-topic questions and count what came back."""
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
    off_topic_answers = [
        await connection.search(working_tree, question, RESULTS_PER_QUERY) for question in off_topic
    ]
    return CorpusQuality(
        corpus=corpus.corpus.name,
        queries=len(ranks),
        file=score([query_ranks.file_rank for query_ranks in ranks]),
        chunk=score([query_ranks.chunk_rank for query_ranks in ranks]),
        ranks=tuple(ranks),
        off_topic_questions=len(off_topic),
        off_topic_results=sum(len(answer.results) for answer in off_topic_answers),
    )


async def rank_query(query: Query, working_tree: Path, connection: ServerConnection) -> QueryRanks:
    """Ask one question and find where its answers landed."""
    answer = await connection.search(working_tree, query.text, RESULTS_PER_QUERY)
    relative_hits = [relative_hit(hit, working_tree) for hit in answer.results]
    answer_paths = {expected.path for expected in query.answers}
    return QueryRanks(
        query=query.text,
        file_rank=first_rank([hit.path in answer_paths for hit in relative_hits]),
        chunk_rank=first_rank([hit in query.answers for hit in relative_hits]),
        hits=tuple(relative_hits),
    )


def missing_answer_chunks(corpus: Corpus, working_tree: Path) -> list[Answer]:
    """Answers whose chunk name the server's chunker never gives in that file.

    Answers name chunks as the chunker names them, so a mistyped name would score as a
    miss forever; this asks the chunker itself, without the model.
    """
    chunker = build_chunker()
    answers = {answer for query in corpus.queries for answer in query.answers}
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
