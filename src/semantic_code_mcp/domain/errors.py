"""The failures of the core that an agent is told about, each with what to do about it.

The tools turn a `SemanticCodeError` into a tool error that carries its message.
"""


class SemanticCodeError(Exception):
    """A failure the agent hears about by its message."""


class IndexRebuiltElsewhereError(SemanticCodeError):
    """Another process built the index again, from another model, chunker or layout,
    after this run planned its write; the run plans again rather than mix the two."""


class IndexWriteError(SemanticCodeError):
    """The index cannot take what a run writes, such as vectors of another width than
    those it holds; a forced re-index builds it again."""
