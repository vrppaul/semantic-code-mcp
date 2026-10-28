"""The embedder the services share: a model the settings name, loaded on first use.

Loading the model reads its files, fetched once from the Hugging Face hub at a pinned
revision, and starts an inference session: time an `index_status` call never needs to
pay. So nothing loads until the first text is embedded, and that happens in a worker
thread (the services embed through `asyncio.to_thread`), never on the event loop. A lock
keeps two first calls from loading the model twice.

What names a model's vectors, its revision and the embedder's version, is known without
the model: a server that searches by words alone reads it for the index's counts and
never loads the model. The libraries that load it (onnxruntime, tokenizers,
huggingface_hub) come with the extra `semantic-code-mcp[meaning]`, so `onnx_model.py`,
which needs them, is imported only when a model loads, and a server set to search by
meaning without them is told which extra installs them (ADR 009, D2).
"""

import threading
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, override

import structlog

from semantic_code_mcp.domain.interfaces import Embedder, Vector, VectorMatrix

if TYPE_CHECKING:
    from semantic_code_mcp.embedding.onnx_model import OnnxModel

log = structlog.get_logger()

# The version of how the embedder makes a text's vector of a model's files: bumped by any
# change to that, so an index of the vectors it made before is never written to again
EMBEDDER_VERSION = 1
# The default model's weights as the harness measured them (`evals/baseline/`)
ALL_MINILM_L6_V2_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
# The revision each model the server knows loads, under every name it goes by: a hub
# model's name follows its newest revision, which can change its vectors under an index
PINNED_REVISIONS: Mapping[str, str] = MappingProxyType(
    {
        "all-MiniLM-L6-v2": ALL_MINILM_L6_V2_REVISION,
        "sentence-transformers/all-MiniLM-L6-v2": ALL_MINILM_L6_V2_REVISION,
    }
)
# What installs the libraries that load a model, an optional extra of the package, and the
# top-level modules it installs
MEANING_EXTRA = "semantic-code-mcp[meaning]"
MEANING_LIBRARIES = frozenset({"onnxruntime", "tokenizers", "huggingface_hub"})
# What parts a module's package from its name
MODULE_NAME_SEPARATOR = "."

# Loads a model by its name and the revision of its weights, None where nothing pins them
type ModelLoader = Callable[[str, str | None], OnnxModel]


class MeaningUnavailableError(RuntimeError):
    """A model is to load, and the libraries that load one are not installed."""


class OnnxEmbedder(Embedder):
    """An embedder over a sentence-transformers model's ONNX export, loaded when first
    needed: a model of the hub, by its name, or one in a directory, by its path."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._revision = PINNED_REVISIONS.get(model_name)
        self._model: OnnxModel | None = None
        self._loading = threading.Lock()

    @property
    @override
    def model_name(self) -> str:
        return self._model_name

    @property
    @override
    def revision(self) -> str | None:
        return self._revision

    @property
    @override
    def version(self) -> int:
        return EMBEDDER_VERSION

    @property
    @override
    def is_loaded(self) -> bool:
        return self._model is not None

    @override
    def load(self) -> None:
        self._loaded_model()

    @override
    def embed_query(self, text: str) -> Vector:
        return self._loaded_model().embed_query(text)

    @override
    def embed_documents(self, texts: list[str]) -> VectorMatrix:
        log.debug("embedding_documents", count=len(texts))
        return self._loaded_model().embed_documents(texts)

    def _loaded_model(self) -> OnnxModel:
        """The model, loading it on the first call."""
        with self._loading:
            if self._model is None:
                self._model = model_loader()(self._model_name, self._revision)
            return self._model


def model_loader() -> ModelLoader:
    """What loads a model, imported only now, as the extra installs the libraries it needs;
    or an error naming the extra where they are missing."""
    try:
        from semantic_code_mcp.embedding.onnx_model import load_model  # noqa: PLC0415
    except ModuleNotFoundError as error:
        missing = (error.name or "").partition(MODULE_NAME_SEPARATOR)[0]
        if missing not in MEANING_LIBRARIES:
            raise
        message = f"a search by meaning needs the model's libraries: install {MEANING_EXTRA}"
        raise MeaningUnavailableError(message) from error
    return load_model
