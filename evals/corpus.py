"""Pinned corpora: which repository at which commit, and the questions asked of it; or a
repository made here, of modules spread over directories, where a large one cannot be cloned.

Each corpus is a TOML file under `evals/corpora/`; one made here lies under `evals/generated/`,
apart, as only a speed run asks it (`--corpus-file`). A question says what it asks for, code
or its documentation, and names every answer that counts — a file, the chunk name inside
it, and its parent where chunks of several parents share that name — so a score never
depends on taste. The questions under `held_out` are asked and reported apart, and never
looked at while a weight or a floor is chosen. A question may also hold the answers agents
gave that a judge settled, right or wrong, as the agents wrote them: what no key names, such
as the same rule in a file the server does not chunk.
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
# The version of what a generated corpus's modules hold and where they lie, in its checkout's
# name: a change of either makes the checkout anew
GENERATOR_VERSION = 1
# A generated corpus's directories and modules, by their numbers, and its one commit's author
GENERATED_DIRECTORY = "package_{number:02}"
GENERATED_MODULE = "module_{number:05}.py"
GENERATED_AUTHOR = ("-c", "user.name=evals", "-c", "user.email=evals@localhost")
# What each generated module is of, by its number in turn, which its names and words carry
GENERATED_SUBJECTS = ("invoice", "order", "payment", "refund", "shipment", "receipt", "account")
GENERATED_SOURCE = '''"""Module {number} of a generated corpus: the ledger of each {subject}."""


def settle_{subject}_{number}(amount):
    """Settle what one {subject} owes, as module {number} reckons it."""
    return amount + {number}


class {title}Ledger{number}:
    """The {subject}s settled so far."""

    def post(self, entry):
        """Post one {subject} to the ledger."""
        return entry
'''


# How a key, and every hit, names the parent of a chunk that sits in nothing
TOP_LEVEL = ""


class Answer(BaseModel):
    """A chunk that answers a question, as its key names it: a file, relative to the corpus
    root, a chunk name, and the chunk's parent where chunks of several parents in the file
    share that name."""

    model_config = ConfigDict(frozen=True)

    path: str
    symbol: str
    # Named only where the name alone would take in another parent's chunk, TOP_LEVEL for a
    # chunk that sits in nothing; None takes a chunk of any parent
    parent: str | None = None


class JudgedAnswer(BaseModel):
    """An agent's answer a judge settled where no key does: its file and its symbol as the
    agent wrote them, whether it is right, and why."""

    model_config = ConfigDict(frozen=True)

    path: str
    symbol: str
    right: bool
    why: str


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
    # Agents' answers a judge settled; none in a corpus no agent has been asked
    judged: tuple[JudgedAnswer, ...] = ()


class PinnedSource(BaseModel):
    """Where a corpus comes from: a repository pinned to one commit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    repository: str
    commit: str

    def checkout_name(self) -> str:
        """The name of its checkout's directory, and of the index kept for it."""
        return f"{self.name}-{self.commit[:COMMIT_PREFIX_LENGTH]}"


class GeneratedSource(BaseModel):
    """Where a corpus comes from: a repository made here, its modules spread over directories,
    each holding so many modules or so many directories, levels deep enough for them all."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    modules: int
    modules_per_directory: int
    directories_per_directory: int

    def checkout_name(self) -> str:
        """The name of its checkout's directory, and of the index kept for it."""
        return (
            f"{self.name}-{self.modules}-{self.modules_per_directory}"
            f"-{self.directories_per_directory}-v{GENERATOR_VERSION}"
        )

    def module_path(self, number: int) -> str:
        """Where the numbered module lies, from the root: in the directory its number falls in,
        named by that directory's number's digits, one a level."""
        levels = digits_in(self.last_directory(), self.directories_per_directory)
        directory = number // self.modules_per_directory
        names: list[str] = []
        for _ in range(levels):
            directory, digit = divmod(directory, self.directories_per_directory)
            names.append(GENERATED_DIRECTORY.format(number=digit))
        return "/".join([*reversed(names), GENERATED_MODULE.format(number=number)])

    def last_directory(self) -> int:
        """The number of the directory the last module lies in."""
        return (self.modules - 1) // self.modules_per_directory


def digits_in(number: int, base: int) -> int:
    """How many digits a number has, written in a base; one for zero."""
    digits = 1
    while number >= base:
        number //= base
        digits += 1
    return digits


def generated_module_source(number: int) -> str:
    """What the numbered module of a generated corpus holds: a function, a class and its
    method, named and worded by the module's subject."""
    subject = GENERATED_SUBJECTS[number % len(GENERATED_SUBJECTS)]
    return GENERATED_SOURCE.format(number=number, subject=subject, title=subject.title())


class Corpus(BaseModel):
    """A corpus file as written: its source and its questions."""

    model_config = ConfigDict(frozen=True)

    corpus: PinnedSource | GeneratedSource
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
    """Read one corpus file of `evals/corpora/` by name."""
    return load_corpus_file(corpus_file_of(name))


def corpus_file_of(name: str) -> Path:
    """The file of a corpus of `evals/corpora/`, by its name."""
    return CORPORA_DIRECTORY / f"{name}{CORPUS_FILE_SUFFIX}"


def load_corpus_file(path: Path) -> Corpus:
    """Read a corpus file wherever it lies, as one of the owner's private repositories keeps
    out of this one."""
    return Corpus.model_validate(tomllib.loads(path.read_text()))


def load_corpora(names: Sequence[str]) -> list[Corpus]:
    """Read the named corpora, or every corpus when no name is given."""
    return [load_corpus(name) for name in names or available_corpus_names()]


def checkout(source: PinnedSource | GeneratedSource, checkouts_root: Path) -> Path:
    """A clean working tree of the corpus, at its pinned commit, cloned on first use, or made
    and committed on first use.

    A checkout kept from an earlier run is restored first, so edits a killed speed run
    left behind are not indexed.
    """
    target = checkouts_root / source.checkout_name()
    if target.is_dir():
        restore(target)
        return target

    partial = target.with_name(target.name + PARTIAL_CHECKOUT_SUFFIX)
    checkouts_root.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(partial, ignore_errors=True)
    try:
        match source:
            case PinnedSource():
                cloned(source, partial)
            case GeneratedSource():
                generated(source, partial)
    except subprocess.CalledProcessError:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    partial.rename(target)
    return target


def cloned(source: PinnedSource, working_tree: Path) -> None:
    """Clone a pinned corpus into a working tree, at its commit."""
    run_git(["clone", "--quiet", "--filter=blob:none", source.repository, str(working_tree)])
    run_git(["-C", str(working_tree), "checkout", "--quiet", "--detach", source.commit])


def generated(source: GeneratedSource, working_tree: Path) -> None:
    """Make a generated corpus's modules in a working tree, and commit them."""
    for number in range(source.modules):
        module = working_tree / source.module_path(number)
        module.parent.mkdir(parents=True, exist_ok=True)
        module.write_text(generated_module_source(number))
    run_git(["-C", str(working_tree), "init", "--quiet"])
    run_git(["-C", str(working_tree), "add", "--all"])
    run_git(
        ["-C", str(working_tree), *GENERATED_AUTHOR, "commit", "--quiet", "--message", "generated"]
    )


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
