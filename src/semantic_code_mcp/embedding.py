"""Embeddings with onnxruntime: a sentence-transformers model's ONNX export and its
tokenizer, loaded on first use.

Loading the model reads its files, fetched once from the Hugging Face hub at a pinned
revision, and starts an inference session: time an `index_status` call never needs to
pay. So nothing loads until the first text is embedded, and that happens in a worker
thread (the services embed through `asyncio.to_thread`), never on the event loop. A lock
keeps two first calls from loading the model twice.

The vectors are those sentence-transformers makes of the same model, to a few parts in a
million, so an index it built keeps answering: the tokenizer cuts a text at the model's
token limit, the model's token states are averaged over the text's tokens, and the mean
is made unit length, as the model's own modules say. sentence-transformers itself, and
torch under it, are not installed: about 0.8 GB less (ADR 006, stage 7).
"""

import threading
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import override

import numpy
import onnxruntime
import structlog
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from tokenizers import Tokenizer

from semantic_code_mcp.domain.interfaces import Embedder, Vector, VectorMatrix

log = structlog.get_logger()

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

# The hub's owner of the models sentence-transformers names without one
SENTENCE_TRANSFORMERS_OWNER = "sentence-transformers"
OWNER_SEPARATOR = "/"
# What the embedder reads of a model, by its place in the model's repository or directory
TOKENIZER_FILE = "tokenizer.json"
ONNX_MODEL_FILE = "onnx/model.onnx"
SENTENCE_BERT_CONFIG_FILE = "sentence_bert_config.json"
POOLING_CONFIG_FILE = "1_Pooling/config.json"
MODEL_FILES = (TOKENIZER_FILE, ONNX_MODEL_FILE, SENTENCE_BERT_CONFIG_FILE, POOLING_CONFIG_FILE)
# The model's inputs, as its ONNX export names them; one text's token types are all zero
INPUT_IDS = "input_ids"
ATTENTION_MASK = "attention_mask"
TOKEN_TYPE_IDS = "token_type_ids"  # nosec B105 - an input's name, not a secret
# The token states are the export's first output, a batch of texts by tokens by width
TOKEN_STATES_OUTPUT = 0
TOKEN_AXIS = 1
# A row of a matrix of vectors runs along this axis
VECTOR_AXIS = 1
# Texts embedded at once, longest first, as sentence-transformers batches them, so that a
# batch, padded to its longest text, pads little
BATCH_SIZE = 32
# What a text of no tokens, or a vector of no length, is divided by instead, as
# sentence-transformers does
LEAST_TOKEN_COUNT = 1e-9
LEAST_LENGTH = 1e-12
CPU_PROVIDER = "CPUExecutionProvider"


class UnsupportedModelError(ValueError):
    """A model whose vectors the embedder cannot make as the model's own library would."""


class SentenceBertConfig(BaseModel):
    """What a sentence-transformers model says of the texts it reads."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    # The longest text it reads, in tokens; the tokenizer's own limit can be another
    max_seq_length: int


class PoolingConfig(BaseModel):
    """How a sentence-transformers model makes one vector of a text's token states."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    pooling_mode_mean_tokens: bool = False
    pooling_mode_cls_token: bool = False
    pooling_mode_max_tokens: bool = False
    pooling_mode_mean_sqrt_len_tokens: bool = False

    @property
    def is_mean(self) -> bool:
        """Whether it takes the mean of the states alone, the one pooling read here."""
        others = (
            self.pooling_mode_cls_token,
            self.pooling_mode_max_tokens,
            self.pooling_mode_mean_sqrt_len_tokens,
        )
        return self.pooling_mode_mean_tokens and not any(others)


class OnnxModel:
    """A model's tokenizer and its inference session, ready to embed."""

    def __init__(self, tokenizer: Tokenizer, session: onnxruntime.InferenceSession) -> None:
        self._tokenizer = tokenizer
        self._session = session
        self._takes_token_types = TOKEN_TYPE_IDS in {node.name for node in session.get_inputs()}

    def embed(self, texts: list[str]) -> VectorMatrix:
        """The unit vectors of at least one text, a row each, in their order."""
        longest_first = sorted(range(len(texts)), key=lambda index: -len(texts[index]))
        batches = [
            self._embed_batch([texts[index] for index in longest_first[start : start + BATCH_SIZE]])
            for start in range(0, len(texts), BATCH_SIZE)
        ]
        embedded = numpy.concatenate(batches)
        vectors = numpy.empty_like(embedded)
        vectors[longest_first] = embedded
        return vectors

    def _embed_batch(self, texts: list[str]) -> VectorMatrix:
        encodings = self._tokenizer.encode_batch(texts)
        token_ids = numpy.array([encoding.ids for encoding in encodings], dtype=numpy.int64)
        mask = numpy.array([encoding.attention_mask for encoding in encodings], dtype=numpy.int64)
        inputs = {INPUT_IDS: token_ids, ATTENTION_MASK: mask}
        if self._takes_token_types:
            inputs[TOKEN_TYPE_IDS] = numpy.zeros_like(token_ids)
        states = self._session.run(None, inputs)[TOKEN_STATES_OUTPUT]
        return unit_rows(mean_over_tokens(numpy.asarray(states, dtype=numpy.float32), mask))


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
    def is_loaded(self) -> bool:
        return self._model is not None

    @override
    def load(self) -> None:
        self._loaded_model()

    @override
    def embed_query(self, text: str) -> Vector:
        return self._loaded_model().embed([text])[0]

    @override
    def embed_documents(self, texts: list[str]) -> VectorMatrix:
        log.debug("embedding_documents", count=len(texts))
        return self._loaded_model().embed(texts)

    def _loaded_model(self) -> OnnxModel:
        """The model, loading it on the first call."""
        with self._loading:
            if self._model is None:
                self._model = load_model(self._model_name, self._revision)
            return self._model


def load_model(model_name: str, revision: str | None) -> OnnxModel:
    """Read a model's files, from its directory or the hub at the revision given, and start
    an inference session over its ONNX export."""
    log.info("loading_embedding_model", model=model_name, revision=revision)
    # The hub draws a bar on stderr as it fetches a file, among the server's log lines
    disable_progress_bars()
    files = {
        relative_path: model_file(model_name, revision, relative_path)
        for relative_path in MODEL_FILES
    }
    pooling = PoolingConfig.model_validate_json(files[POOLING_CONFIG_FILE].read_bytes())
    if not pooling.is_mean:
        message = (
            f"{model_name} pools a text's token states otherwise than by their mean alone, "
            "which this embedder does not read"
        )
        raise UnsupportedModelError(message)
    limits = SentenceBertConfig.model_validate_json(files[SENTENCE_BERT_CONFIG_FILE].read_bytes())
    tokenizer = Tokenizer.from_file(str(files[TOKENIZER_FILE]))
    # The model's limit, and padding to a batch's longest text, as sentence-transformers
    # sets them over what the tokenizer's file holds (all-MiniLM-L6-v2's: 128 tokens both)
    tokenizer.enable_truncation(max_length=limits.max_seq_length)
    tokenizer.enable_padding()
    session = onnxruntime.InferenceSession(str(files[ONNX_MODEL_FILE]), providers=[CPU_PROVIDER])
    log.info("embedding_model_loaded", model=model_name)
    return OnnxModel(tokenizer, session)


def model_file(model_name: str, revision: str | None, relative_path: str) -> Path:
    """Where one of a model's files lies: in the model's directory when its name is one,
    else in the hub's cache, fetched there once."""
    directory = Path(model_name)
    if directory.is_dir():
        return directory / relative_path
    return Path(hf_hub_download(hub_repository(model_name), relative_path, revision=revision))


def hub_repository(model_name: str) -> str:
    """The hub repository a model's name names: sentence-transformers' own models go by
    their name alone."""
    if OWNER_SEPARATOR in model_name:
        return model_name
    return f"{SENTENCE_TRANSFORMERS_OWNER}{OWNER_SEPARATOR}{model_name}"


def mean_over_tokens(states: NDArray[numpy.float32], mask: NDArray[numpy.int64]) -> VectorMatrix:
    """Each text's token states averaged over its own tokens, its padding left out."""
    weights = mask[:, :, numpy.newaxis].astype(numpy.float32)
    totals = (states * weights).sum(axis=TOKEN_AXIS)
    counts = numpy.maximum(weights.sum(axis=TOKEN_AXIS), LEAST_TOKEN_COUNT)
    return totals / counts


def unit_rows(vectors: VectorMatrix) -> VectorMatrix:
    """The vectors scaled to unit length, as the model's last module makes them."""
    lengths = numpy.linalg.norm(vectors, axis=VECTOR_AXIS, keepdims=True)
    return vectors / numpy.maximum(lengths, LEAST_LENGTH)
