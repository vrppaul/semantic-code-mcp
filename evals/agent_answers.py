"""An agent's answer scored against a question's keys, through the chunks of the file it
names (ADR 008, D1).

An agent writes its answer's name as it pleases: bare (`convert`), by its class
(`Choice.convert`) or its module (`click.termui.progressbar`), as a call
(`put_if_absent()`), a heading whole (`11. Glossary`). So the name is first read as the
chunks of its file it names, as the server chunks that file, and those chunks are matched
with the keys as `evals quality` matches a hit, by the parent where a key names one. A
member of a keyed class, or the class of a keyed method, counts as right and is counted
apart. What the chunks cannot settle, a bare name of several parents that the keys tell
apart or a name no chunk has, is left to a judge, whose verdict goes into the corpus's keys,
and a kept run is scored again without new sessions.
"""

from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum, auto
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from evals.corpus import Answer
from evals.quality import chunk_hit, hit_answers
from semantic_code_mcp.app import build_chunker
from semantic_code_mcp.domain.models import PARENT_SEPARATORS, Chunk
from semantic_code_mcp.services.search import name_asked

# What an answer may wrap its name in: a fence's backticks, a call's parentheses, the marks
# of a Markdown heading
CODE_FENCE = "`"
CALL_PARENTHESES = "()"
HEADING_MARK = "#"
# What prefixes a path written from the repository's root
CURRENT_DIRECTORY_PREFIX = "./"


class SessionAnswer(BaseModel):
    """What a session answered: a file and the symbol in it, as the agent wrote them."""

    model_config = ConfigDict(frozen=True)

    path: str
    symbol: str


class AnswerVerdict(StrEnum):
    """How the chunks an answer names meet the question's keys."""

    # A key names the chunk the answer names
    right = auto()
    # The answer names a member of a keyed chunk, or the chunk a keyed one sits in
    member = auto()
    # The answer names a chunk no key names, nor a member or the owner of one
    other_chunk = auto()
    # The answer's name is chunks of several parents, which the keys tell apart
    ambiguous = auto()
    # No chunk of the answer's file, as the server chunks it, has the answer's name
    unresolved = auto()


# The verdicts a run counts as right answers
RIGHT_VERDICTS = frozenset({AnswerVerdict.right, AnswerVerdict.member})
# The verdicts a judge settles, as no key does
VERDICTS_TO_JUDGE = frozenset(
    {AnswerVerdict.other_chunk, AnswerVerdict.ambiguous, AnswerVerdict.unresolved}
)


class AnswerScore(BaseModel):
    """An answer scored: its file from the repository's root, as the keys name files,
    whether a key names that file, and how the chunks the answer names meet the keys."""

    model_config = ConfigDict(frozen=True)

    path: str
    file_right: bool
    verdict: AnswerVerdict


class AnswerScorer:
    """Scores the answers of one corpus's sessions, each file chunked once, as the answers of
    a run name a few files many times."""

    def __init__(self, working_tree: Path) -> None:
        self._working_tree = working_tree
        self._chunker = build_chunker()
        self._chunks_by_path: dict[str, list[Chunk]] = {}

    def score(self, answer: SessionAnswer, keys: Sequence[Answer]) -> AnswerScore:
        """An answer scored against a question's keys."""
        paths = [answered_path(answer, self._working_tree), *(key.path for key in keys)]
        return score_answer(answer, keys, self._chunks_of(paths), self._working_tree)

    def _chunks_of(self, paths: Iterable[str]) -> dict[str, list[Chunk]]:
        for path in paths:
            if path not in self._chunks_by_path:
                self._chunks_by_path[path] = self._chunked(path)
        return self._chunks_by_path

    def _chunked(self, path: str) -> list[Chunk]:
        """A file's chunks as the server chunks it; none for a path outside the checkout or
        no file at all, which an agent may answer."""
        checkout = self._working_tree.resolve()
        full_path = (checkout / path).resolve()
        if not full_path.is_relative_to(checkout) or not full_path.is_file():
            return []
        return self._chunker.chunk_file(checkout, full_path.relative_to(checkout))


def score_answer(
    answer: SessionAnswer,
    keys: Sequence[Answer],
    chunks_by_path: Mapping[str, Sequence[Chunk]],
    working_tree: Path,
) -> AnswerScore:
    """An answer scored against a question's keys, through the chunks of the files it and the
    keys name."""
    path = answered_path(answer, working_tree)
    named = chunks_named(answer.symbol, chunks_by_path.get(path, ()))
    keyed = [
        chunk
        for key in keys
        for chunk in chunks_by_path.get(key.path, ())
        if hit_answers(chunk_hit(chunk), key)
    ]
    return AnswerScore(
        path=path,
        file_right=any(key.path == path for key in keys),
        verdict=verdict_of(named, keyed),
    )


def answered_path(answer: SessionAnswer, working_tree: Path) -> str:
    """The answer's file from the repository's root, as the keys name files."""
    path = Path(answer.path)
    if path.is_absolute() and path.is_relative_to(working_tree):
        return path.relative_to(working_tree).as_posix()
    return answer.path.removeprefix(CURRENT_DIRECTORY_PREFIX)


def chunks_named(symbol: str, chunks: Sequence[Chunk]) -> list[Chunk]:
    """The chunks of a file an answer's symbol names: those of its whole name, as a heading
    with dots is one; else, split at the last separator a language qualifies a name by, those
    of the name whose parent the qualifier ends with, or, where no chunk of the name sits in
    it, those of the name at the top of the file, the qualifier read as the module's path."""
    asked = name_asked(written_name(symbol))
    whole = [chunk for chunk in chunks if chunk.name.casefold() == asked.whole]
    if whole:
        return whole
    named: list[Chunk] = []
    for qualified in asked.qualified:
        of_name = [chunk for chunk in chunks if chunk.name.casefold() == qualified.name]
        in_parent = [
            chunk
            for chunk in of_name
            if chunk.parent is not None and ends_with_parent(qualified.parent, chunk.parent)
        ]
        named += in_parent or [chunk for chunk in of_name if chunk.parent is None]
    return list({chunk.chunk_id: chunk for chunk in named}.values())


def written_name(symbol: str) -> str:
    """The name an answer writes, out of the fence, the call's parentheses or the heading's
    marks it may wrap it in."""
    return (
        symbol.strip().strip(CODE_FENCE).removesuffix(CALL_PARENTHESES).lstrip(HEADING_MARK).strip()
    )


def ends_with_parent(qualifier: str, parent: str) -> bool:
    """Whether a qualifier names a parent, whole or as its last parts: `ProgressBar` and
    `click._termui_impl.ProgressBar` both name `ProgressBar`."""
    folded = parent.casefold()
    return qualifier == folded or any(
        qualifier.endswith(separator + folded) for separator in PARENT_SEPARATORS.values()
    )


def verdict_of(named: Sequence[Chunk], keyed: Sequence[Chunk]) -> AnswerVerdict:
    """How the chunks an answer names meet the keyed chunks: one verdict where every chunk
    named has the same; ambiguous where they differ."""
    match [*{chunk_verdict(chunk, keyed) for chunk in named}]:
        case []:
            return AnswerVerdict.unresolved
        case [verdict]:
            return verdict
        case _:
            return AnswerVerdict.ambiguous


def chunk_verdict(chunk: Chunk, keyed: Sequence[Chunk]) -> AnswerVerdict:
    """How one chunk an answer names meets the keyed chunks."""
    if any(chunk.chunk_id == key.chunk_id for key in keyed):
        return AnswerVerdict.right
    if any(sits_in(chunk, key) or sits_in(key, chunk) for key in keyed):
        return AnswerVerdict.member
    return AnswerVerdict.other_chunk


def sits_in(chunk: Chunk, owner: Chunk) -> bool:
    """Whether a chunk is defined right in another of its file: a method in its class, a
    section under its heading."""
    return chunk.path == owner.path and chunk.parent == qualified_name(owner)


def qualified_name(chunk: Chunk) -> str:
    """A chunk's name qualified by its parent's, as its language joins them."""
    if chunk.parent is None:
        return chunk.name
    return f"{chunk.parent}{PARENT_SEPARATORS[chunk.language]}{chunk.name}"
