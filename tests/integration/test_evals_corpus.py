"""Tests of a corpus's checkout: a corpus made here holds each of its modules where its path says,
committed, so a speed run times a repository of a large one's shape where none can be cloned."""

import subprocess
from pathlib import Path

from evals.corpus import GeneratedSource, checkout, load_corpus_file

# A corpus of ten modules, three a directory, two directories a directory: two levels
SMALL = GeneratedSource(
    name="small", modules=10, modules_per_directory=3, directories_per_directory=2
)
SENTRY_SHAPED = Path(__file__).parents[2] / "evals" / "generated" / "sentry-shaped.toml"


def git_output(working_tree: Path, *arguments: str) -> list[str]:
    run = subprocess.run(
        ["git", "-C", str(working_tree), *arguments], check=True, capture_output=True, text=True
    )
    return run.stdout.splitlines()


def test_a_generated_corpus_holds_each_module_where_its_path_says_committed(tmp_path: Path):
    working_tree = checkout(SMALL, tmp_path)

    assert (
        sorted(git_output(working_tree, "ls-files")),
        git_output(working_tree, "status", "--porcelain"),
    ) == (
        sorted([".gitignore", *(SMALL.module_path(number) for number in range(SMALL.modules))]),
        [],
    )


def test_the_sentry_shaped_corpus_s_answers_lie_where_its_modules_do():
    """Its questions name the modules by their paths, which the generator decides."""
    corpus = load_corpus_file(SENTRY_SHAPED)
    source = corpus.corpus
    answers = sorted(answer.path for query in corpus.queries for answer in query.answers)

    assert isinstance(source, GeneratedSource)
    assert set(answers) <= {source.module_path(number) for number in range(source.modules)}
