"""Pinned corpora: which repository at which commit, and the questions asked of it.

Each corpus is a TOML file under `evals/corpora/`. A question says what it asks for, code
or its documentation, and names every answer that counts — a file and the chunk name
inside it — so a score never depends on taste. The questions under `held_out` are asked
and reported apart, and never looked at while a weight or a floor is chosen.
"""

import shutil
import subprocess  # nosec B404
import tomllib
from collections.abc import Sequence
from enum import StrEnum, auto
from pathlib import Path

from pydantic import BaseModel, ConfigDict

CORPORA_DIRECTORY = Path(__file__).parent / "corpora"
# Questions no corpus answers, asked of every corpus
OFF_TOPIC_FILE = Path(__file__).parent / "off_topic.toml"
CORPUS_FILE_SUFFIX = ".toml"
# A checkout's directory is named after the corpus and this many characters of its commit
COMMIT_PREFIX_LENGTH = 12
# Where a clone is made before it is moved into place; a failed clone or checkout is removed
PARTIAL_CHECKOUT_SUFFIX = ".partial"


class Answer(BaseModel):
    """A result that answers a question: a file, relative to the corpus root, and a chunk name."""

    model_config = ConfigDict(frozen=True)

    path: str
    symbol: str


class QuestionKind(StrEnum):
    """What a question asks for: the code that does a thing, or the documentation of it."""

    code = auto()
    documentation = auto()


class Query(BaseModel):
    """A question in plain words, what it asks for, and every answer that counts."""

    model_config = ConfigDict(frozen=True)

    text: str
    kind: QuestionKind
    answers: tuple[Answer, ...]


class CorpusSource(BaseModel):
    """Where a corpus comes from: a repository pinned to one commit."""

    model_config = ConfigDict(frozen=True)

    name: str
    repository: str
    commit: str


class Corpus(BaseModel):
    """A corpus file as written: its source and its questions."""

    model_config = ConfigDict(frozen=True)

    corpus: CorpusSource
    queries: tuple[Query, ...]
    held_out: tuple[Query, ...]

    def every_query(self) -> tuple[Query, ...]:
        """The questions chosen by and the questions held out, together."""
        return (*self.queries, *self.held_out)


class OffTopicQuestions(BaseModel):
    """Questions no corpus answers, as the file holds them: of other fields than
    programming, and of programming."""

    model_config = ConfigDict(frozen=True)

    far: tuple[str, ...]
    near: tuple[str, ...]


def load_off_topic_questions() -> OffTopicQuestions:
    """The questions no corpus answers, which a good search answers with nothing."""
    return OffTopicQuestions.model_validate(tomllib.loads(OFF_TOPIC_FILE.read_text()))


def available_corpus_names() -> list[str]:
    """The names of every corpus under `evals/corpora/`."""
    return sorted(path.stem for path in CORPORA_DIRECTORY.glob(f"*{CORPUS_FILE_SUFFIX}"))


def load_corpus(name: str) -> Corpus:
    """Read one corpus file by name."""
    path = CORPORA_DIRECTORY / f"{name}{CORPUS_FILE_SUFFIX}"
    return Corpus.model_validate(tomllib.loads(path.read_text()))


def load_corpora(names: Sequence[str]) -> list[Corpus]:
    """Read the named corpora, or every corpus when no name is given."""
    return [load_corpus(name) for name in names or available_corpus_names()]


def checkout(source: CorpusSource, checkouts_root: Path) -> Path:
    """A clean working tree of the corpus at its pinned commit, cloned on first use.

    A checkout kept from an earlier run is restored first, so edits a killed speed run
    left behind are not indexed.
    """
    target = checkouts_root / f"{source.name}-{source.commit[:COMMIT_PREFIX_LENGTH]}"
    if target.is_dir():
        restore(target)
        return target

    partial = target.with_name(target.name + PARTIAL_CHECKOUT_SUFFIX)
    checkouts_root.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(partial, ignore_errors=True)
    try:
        run_git(["clone", "--quiet", "--filter=blob:none", source.repository, str(partial)])
        run_git(["-C", str(partial), "checkout", "--quiet", "--detach", source.commit])
    except subprocess.CalledProcessError:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    partial.rename(target)
    return target


def missing_answer_paths(corpus: Corpus, working_tree: Path) -> list[str]:
    """Answer paths that name no file in the checkout: a mistake in the corpus file."""
    answer_paths = {answer.path for query in corpus.every_query() for answer in query.answers}
    return sorted(path for path in answer_paths if not (working_tree / path).is_file())


def restore(working_tree: Path) -> None:
    """Undo every change to the tracked files of a checkout."""
    run_git(["-C", str(working_tree), "checkout", "--quiet", "--", "."])


def run_git(arguments: list[str]) -> None:
    """Run git with a fixed argument list, failing loudly."""
    subprocess.run(["git", *arguments], check=True, capture_output=True)  # nosec B603, B607
