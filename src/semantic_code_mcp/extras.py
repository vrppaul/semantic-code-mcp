"""The package's optional extras, and the one way a setting that needs one says it is missing.

A store or a search mode may need libraries the base install leaves out: the module that
imports them is imported only when the settings pick what needs it, inside `needing`, which
turns a missing library of the extra into an error naming what installs it. A module of the
package itself missing is a broken install, and stays the error it is, as installing the
extra would not mend it.
"""

from collections.abc import Generator
from contextlib import contextmanager

from pydantic import BaseModel, ConfigDict

# What parts a module's package from its name
MODULE_NAME_SEPARATOR = "."


class OptionalExtra(BaseModel):
    """An optional extra of the package: what installs it, the top-level modules it
    installs, and what needs them."""

    model_config = ConfigDict(frozen=True)

    requirement: str
    libraries: frozenset[str]
    needed_by: str


LANCEDB_EXTRA = OptionalExtra(
    requirement="semantic-code-mcp[lancedb]",
    libraries=frozenset({"lancedb", "lance", "pyarrow"}),
    needed_by="the LanceDB store",
)
MEANING_EXTRA = OptionalExtra(
    requirement="semantic-code-mcp[meaning]",
    libraries=frozenset({"onnxruntime", "tokenizers", "huggingface_hub"}),
    needed_by="a search by meaning",
)


class ExtraMissingError(RuntimeError):
    """The settings pick what needs an optional extra, and its libraries are not installed."""


@contextmanager
def needing(extra: OptionalExtra) -> Generator[None]:
    """Imports made inside need the extra: a library of it missing is an error naming what
    installs it; any other missing module stays the error it is."""
    try:
        yield
    except ModuleNotFoundError as error:
        missing = (error.name or "").partition(MODULE_NAME_SEPARATOR)[0]
        if missing not in extra.libraries:
            raise
        message = f"{extra.needed_by} needs its libraries: install {extra.requirement}"
        raise ExtraMissingError(message) from error
