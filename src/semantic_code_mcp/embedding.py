"""Embeddings with onnxruntime: a sentence-transformers model's ONNX export and its
tokenizer, loaded on first use.

Loading the model reads its files, fetched once from the Hugging Face hub at a pinned
revision, and starts an inference session: time an `index_status` call never needs to
pay. So nothing loads until the first text is embedded, and that happens in a worker
thread (the services embed through `asyncio.to_thread`), never on the event loop. A lock
keeps two first calls from loading the model twice.

The vectors are those sentence-transformers makes of the same model, to a few parts in a
million, so an index it built keeps answering: the tokenizer lowercases a text where the
model says to and cuts it at the model's token limit, the model's token states are
averaged over the text's tokens, and the mean is made unit length, as the model's own
modules say. sentence-transformers itself, and torch under it, are not installed: about
0.9 GB less (ADR 006, stage 7).
"""

import threading
from collections.abc import Mapping
from enum import StrEnum, auto
from pathlib import Path
from types import MappingProxyType
from typing import override

import numpy
import onnxruntime
import structlog
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError, RemoteEntryNotFoundError
from huggingface_hub.utils import disable_progress_bars
from numpy.typing import NDArray

# onnxruntime's errors live in its compiled module, which ships no stubs
from onnxruntime.capi import onnxruntime_pybind11_state  # ty: ignore[unresolved-import]
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    RootModel,
    ValidationError,
)
from tokenizers import Tokenizer
from tokenizers.normalizers import Lowercase, Normalizer, Sequence

from semantic_code_mcp.domain.interfaces import Embedder, Vector, VectorMatrix

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

# The hub's owner of the models sentence-transformers names without one
SENTENCE_TRANSFORMERS_OWNER = "sentence-transformers"
OWNER_SEPARATOR = "/"
# What the embedder reads of a model, by its place in the model's repository or directory:
# the configs first, which say whether it can make the model's vectors, then the rest
MODULES_FILE = "modules.json"
POOLING_CONFIG_FILE = "1_Pooling/config.json"
SENTENCE_BERT_CONFIG_FILE = "sentence_bert_config.json"
TRANSFORMER_CONFIG_FILE = "config.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
TOKENIZER_FILE = "tokenizer.json"
ONNX_MODEL_FILE = "onnx/model.onnx"


class ModuleKind(StrEnum):
    """The modules of a sentence-transformers model whose work the embedder does."""

    transformer = auto()
    pooling = auto()
    normalize = auto()


# Each module whose work the embedder does, under every type sentence-transformers saves it
# as: in `sentence_transformers.models`, or where version 5.4 moved it
MODULE_KINDS: Mapping[str, ModuleKind] = MappingProxyType(
    {
        "sentence_transformers.models.Transformer": ModuleKind.transformer,
        "sentence_transformers.base.modules.transformer.Transformer": ModuleKind.transformer,
        "sentence_transformers.models.Pooling": ModuleKind.pooling,
        "sentence_transformers.sentence_transformer.modules.pooling.Pooling": ModuleKind.pooling,
        "sentence_transformers.models.Normalize": ModuleKind.normalize,
        "sentence_transformers.sentence_transformer.modules.normalize.Normalize": (
            ModuleKind.normalize
        ),
    }
)
# The chains of modules whose vectors the embedder makes: the transformer, the mean of its
# token states, and a unit length or none. It makes every vector unit length, so a model
# without that last module gets sentence-transformers' vectors in direction alone, which
# the store's cosine reads alike
READ_MODULE_CHAINS = (
    (ModuleKind.transformer, ModuleKind.pooling),
    (ModuleKind.transformer, ModuleKind.pooling, ModuleKind.normalize),
)
MEAN_POOLING = "mean"
# How a message lists names: a chain's types, an export's inputs
LIST_SEPARATOR = ", "
# The model's inputs, as its ONNX export names them; one text's token types are all zero
INPUT_IDS = "input_ids"
ATTENTION_MASK = "attention_mask"
TOKEN_TYPE_IDS = "token_type_ids"  # nosec B105 - an input's name, not a secret
# The inputs the embedder sends: these two always, and the token types where the export
# takes them
NEEDED_INPUTS = frozenset({INPUT_IDS, ATTENTION_MASK})
SENT_INPUTS = frozenset({INPUT_IDS, ATTENTION_MASK, TOKEN_TYPE_IDS})
# The token states are the export's first output, a batch of texts by tokens by width
TOKEN_STATES_OUTPUT = 0
# A short text the export runs once when it loads, so the shape of its states is seen
# before any search depends on them
EXPORT_PROBE_TEXT = "def greet(name): return name"
ONE_TEXT = 1
# The keys a pooling's config gives its states' width under: sentence-transformers 5.4's,
# and the older one it renames
WIDTH_KEYS = AliasChoices("embedding_dimension", "word_embedding_dimension")
TOKEN_AXIS = 1
# A row of a matrix of vectors runs along this axis
# A query's one row of the matrix its embedding gives
QUERY_ROW = 0
VECTOR_AXIS = 1
# Texts embedded at once, longest first, as sentence-transformers batches them, so that a
# batch, padded to its longest text, pads little
BATCH_SIZE = 32
# What a text of no tokens, or a vector of no length, is divided by instead, as
# sentence-transformers does
LEAST_TOKEN_COUNT = 1e-9
LEAST_LENGTH = 1e-12
CPU_PROVIDER = "CPUExecutionProvider"
# onnxruntime's threads otherwise spin for about half a second after each run, so a 2 ms
# query burned some 6 CPU-seconds (a review's probe); off, a chunk embeds as fast
SPINNING_CONFIG_KEY = "session.intra_op.allow_spinning"
NO_SPINNING = "0"
# onnxruntime's memory arena keeps the buffers of the largest batch it ran for the life of
# the process, so a server that indexed held about 1,080 MB until it exited. A batch of
# documents hands back what the arena holds free once it ends, which holds about 540 MB
# and indexes as fast, where an embedder without the arena indexed a fifth slower; a query
# keeps its small buffers for the next (`evals speed`, in turns)
ARENA_SHRINKAGE_CONFIG_KEY = "memory.enable_memory_arena_shrinkage"
SHRINK_CPU_ARENA = "cpu:0"
# What onnxruntime raises, each a bare Exception of its own, when an export starts no
# session or does not run: a file of no ONNX model (InvalidProtobuf) and weights in a file
# beside it that is missing (Fail), both probed, and by their names the rest an export can
# bring about: an argument it refuses, a file or a model it cannot find, a graph it cannot
# read, an operator it cannot run, an exception while it starts the session
EXPORT_ERRORS = (
    onnxruntime_pybind11_state.Fail,
    onnxruntime_pybind11_state.InvalidArgument,
    onnxruntime_pybind11_state.InvalidGraph,
    onnxruntime_pybind11_state.InvalidProtobuf,
    onnxruntime_pybind11_state.NoModel,
    onnxruntime_pybind11_state.NoSuchFile,
    onnxruntime_pybind11_state.NotFound,
    onnxruntime_pybind11_state.NotImplemented,
    onnxruntime_pybind11_state.RuntimeException,
)


class UnusableModelError(ValueError):
    """A model the embedder cannot load, or whose vectors it cannot make as the model's own
    library would; the message names the file and says why."""


class ModuleEntry(BaseModel):
    """One module of a sentence-transformers model's chain, as `modules.json` lists it."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    type: str

    @property
    def kind(self) -> ModuleKind | None:
        """What the module does, where the embedder does its work."""
        return MODULE_KINDS.get(self.type)


class ModuleChain(RootModel[list[ModuleEntry]]):
    """A sentence-transformers model's modules in their order, as `modules.json` lists them."""

    model_config = ConfigDict(frozen=True)

    @property
    def kinds(self) -> tuple[ModuleKind | None, ...]:
        """Each module's kind, in the chain's order."""
        return tuple(module.kind for module in self.root)

    @property
    def types(self) -> str:
        """Each module's type, in the chain's order, as a reader is told them."""
        return LIST_SEPARATOR.join(module.type for module in self.root)


class PoolingConfig(BaseModel):
    """How a sentence-transformers model makes one vector of a text's token states: one
    mode, or several joined, by name since sentence-transformers 5.4, by flags before."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    embedding_dimension: PositiveInt = Field(validation_alias=WIDTH_KEYS)
    pooling_mode: str | list[str] | None = None
    pooling_mode_mean_tokens: bool = False
    pooling_mode_cls_token: bool = False
    pooling_mode_max_tokens: bool = False
    pooling_mode_mean_sqrt_len_tokens: bool = False
    pooling_mode_weightedmean_tokens: bool = False
    pooling_mode_lasttoken: bool = False

    @property
    def is_mean(self) -> bool:
        """Whether it takes the mean of the states alone, the one pooling read here."""
        if self.pooling_mode is not None:
            return self.pooling_mode in (MEAN_POOLING, [MEAN_POOLING])
        others = (
            self.pooling_mode_cls_token,
            self.pooling_mode_max_tokens,
            self.pooling_mode_mean_sqrt_len_tokens,
            self.pooling_mode_weightedmean_tokens,
            self.pooling_mode_lasttoken,
        )
        return self.pooling_mode_mean_tokens and not any(others)


class SentenceBertConfig(BaseModel):
    """What a sentence-transformers model says of the texts it reads: whether to lowercase
    them first, and the longest, in tokens. Models saved by sentence-transformers 5.4 or
    later say neither, leaving both to the tokenizer."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    do_lower_case: bool = False
    max_seq_length: PositiveInt | None = None


class TokenizerConfig(BaseModel):
    """The tokenizer's own limit, which a model's sentence-transformers config can lower."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    model_max_length: int


class TransformerConfig(BaseModel):
    """The transformer's positions, which bound the tokens it reads (RoBERTa reads two
    fewer), where its config names them as BERT's does; positive, as XLNet's -1 for none
    would leave a text uncut."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    max_position_embeddings: PositiveInt | None = None


class OnnxModel:
    """A model's tokenizer and its inference session, ready to embed."""

    def __init__(self, tokenizer: Tokenizer, session: onnxruntime.InferenceSession) -> None:
        self._tokenizer = tokenizer
        self._session = session
        self._shrinking = onnxruntime.RunOptions()
        self._shrinking.add_run_config_entry(ARENA_SHRINKAGE_CONFIG_KEY, SHRINK_CPU_ARENA)
        self._keeping = onnxruntime.RunOptions()
        self._takes_token_types = TOKEN_TYPE_IDS in {node.name for node in session.get_inputs()}

    def embed_documents(self, texts: list[str]) -> VectorMatrix:
        """The unit vectors of at least one document, a row each, in their order; the
        arena hands back each batch's buffers once it ends."""
        return self._embed(texts, self._shrinking)

    def embed_query(self, text: str) -> Vector:
        """The unit vector of a query; the arena keeps its buffers for the next."""
        return self._embed([text], self._keeping)[QUERY_ROW]

    def _embed(self, texts: list[str], run_options: onnxruntime.RunOptions) -> VectorMatrix:
        longest_first = sorted(range(len(texts)), key=lambda index: -len(texts[index]))
        batches = [
            self._embed_batch(
                [texts[index] for index in longest_first[start : start + BATCH_SIZE]], run_options
            )
            for start in range(0, len(texts), BATCH_SIZE)
        ]
        embedded = numpy.concatenate(batches)
        vectors = numpy.empty_like(embedded)
        vectors[longest_first] = embedded
        return vectors

    def token_states(self, texts: list[str]) -> NDArray[numpy.float32]:
        """Each token's state of each text, as the export's first output gives them: texts
        by tokens by width, where the export is one the embedder reads."""
        return self._states(self._inputs(texts), self._keeping)

    def token_count(self, text: str) -> int:
        """How many tokens the model reads of a text, its limit holding."""
        return len(self._tokenizer.encode(text).ids)

    def _embed_batch(self, texts: list[str], run_options: onnxruntime.RunOptions) -> VectorMatrix:
        inputs = self._inputs(texts)
        states = self._states(inputs, run_options)
        return unit_rows(mean_over_tokens(states, inputs[ATTENTION_MASK]))

    def _inputs(self, texts: list[str]) -> dict[str, NDArray[numpy.int64]]:
        """What the export is given of a batch: its tokens, the mask over them, and the
        token types, all zero, where it takes them."""
        encodings = self._tokenizer.encode_batch(texts)
        token_ids = numpy.array([encoding.ids for encoding in encodings], dtype=numpy.int64)
        mask = numpy.array([encoding.attention_mask for encoding in encodings], dtype=numpy.int64)
        inputs = {INPUT_IDS: token_ids, ATTENTION_MASK: mask}
        if self._takes_token_types:
            inputs[TOKEN_TYPE_IDS] = numpy.zeros_like(token_ids)
        return inputs

    def _states(
        self, inputs: dict[str, NDArray[numpy.int64]], run_options: onnxruntime.RunOptions
    ) -> NDArray[numpy.float32]:
        states = self._session.run(None, inputs, run_options)[TOKEN_STATES_OUTPUT]
        return numpy.asarray(states, dtype=numpy.float32)


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
                self._model = load_model(self._model_name, self._revision)
            return self._model


class ModelFiles:
    """Where a model's files are read: its directory, when its name is one, else the hub's
    cache at the revision given, each file fetched there once."""

    def __init__(self, model_name: str, revision: str | None) -> None:
        self._model_name = model_name
        self._revision = revision

    @property
    def model_name(self) -> str:
        return self._model_name

    def path(self, relative_path: str) -> Path:
        """Where one of the model's files lies."""
        directory = Path(self._model_name)
        if directory.is_dir():
            return file_in_directory(directory, relative_path)
        repository = hub_repository(self._model_name)
        try:
            return Path(hf_hub_download(repository, relative_path, revision=self._revision))
        except LocalEntryNotFoundError as error:
            message = (
                f"{self._model_name}'s {relative_path} is neither in the hub's cache nor "
                "fetched from the hub (offline, or the hub did not answer): start the server "
                "once where it reaches the hub"
            )
            raise UnusableModelError(message) from error
        except RemoteEntryNotFoundError as error:
            message = (
                f"{self._model_name} has no {relative_path} on the hub: the embedder reads "
                "a model's ONNX export and its sentence-transformers configs"
            )
            raise UnusableModelError(message) from error

    def config[ConfigModel: BaseModel](
        self, relative_path: str, config_type: type[ConfigModel]
    ) -> ConfigModel:
        """One of the model's JSON configs, read as the type given."""
        try:
            return config_type.model_validate_json(self.path(relative_path).read_bytes())
        except ValidationError as error:
            problems = error.json(include_url=False, include_context=False, include_input=False)
            message = (
                f"{self._model_name}'s {relative_path} does not hold what the embedder reads "
                f"of it: {problems}"
            )
            raise UnusableModelError(message) from error


def load_model(model_name: str, revision: str | None) -> OnnxModel:
    """Read a model's configs, from its directory or the hub at the revision given, check
    that the embedder makes its vectors, then read its tokenizer and start an inference
    session over its ONNX export."""
    log.info("loading_embedding_model", model=model_name, revision=revision)
    # The hub draws a bar on stderr as it fetches a file, among the server's log lines
    disable_progress_bars()
    files = ModelFiles(model_name, revision)
    pooling = checked_pooling(files)
    tokenizer = model_tokenizer(files)
    session = inference_session(files)
    check_inputs(files, session)
    model = OnnxModel(tokenizer, session)
    check_states(files, model, pooling.embedding_dimension)
    log.info("embedding_model_loaded", model=model_name)
    return model


def checked_pooling(files: ModelFiles) -> PoolingConfig:
    """The model's pooling, once its modules are those whose work the embedder does; a
    model whose modules make its vectors otherwise is refused."""
    chain = files.config(MODULES_FILE, ModuleChain)
    if chain.kinds not in READ_MODULE_CHAINS:
        message = (
            f"{files.model_name}'s {MODULES_FILE} chains {chain.types}: the embedder reads a "
            "transformer, its pooling and a unit length, as sentence-transformers names them"
        )
        raise UnusableModelError(message)
    pooling = files.config(POOLING_CONFIG_FILE, PoolingConfig)
    if not pooling.is_mean:
        message = (
            f"{files.model_name} pools a text's token states otherwise than by their mean "
            f"alone ({POOLING_CONFIG_FILE})"
        )
        raise UnusableModelError(message)
    return pooling


def model_tokenizer(files: ModelFiles) -> Tokenizer:
    """The model's tokenizer, set as sentence-transformers sets it over what the tokenizer's
    file holds: lowercasing a text first where the model says to, cutting it at the model's
    limit, and padding a batch to its longest text (all-MiniLM-L6-v2's file says 128 tokens
    for both)."""
    config = files.config(SENTENCE_BERT_CONFIG_FILE, SentenceBertConfig)
    limit = token_limit(files, config)
    tokenizer = read_tokenizer(files)
    if config.do_lower_case:
        tokenizer.normalizer = lowercasing_first(tokenizer.normalizer)
    tokenizer.enable_truncation(max_length=limit)
    tokenizer.enable_padding()
    return tokenizer


def read_tokenizer(files: ModelFiles) -> Tokenizer:
    """The model's tokenizer, as its file holds it."""
    path = files.path(TOKENIZER_FILE)
    try:
        return Tokenizer.from_file(str(path))
    # tokenizers raises a bare Exception for a file it cannot read, and no narrower class;
    # a narrower one, as a MemoryError, is no refusal of the file and goes on as it came
    except Exception as error:
        if type(error) is not Exception:
            raise
        message = f"{files.model_name}'s {TOKENIZER_FILE} does not load in tokenizers: {error}"
        raise UnusableModelError(message) from error


def token_limit(files: ModelFiles, config: SentenceBertConfig) -> int:
    """The longest text the model reads, in tokens, as sentence-transformers sets it: its
    own config's, or where that names none, its tokenizer's held to the positions its
    transformer has (a tokenizer's config can hold transformers' "no limit", 10^30). A
    limit of its own past those positions is refused, as the transformer could not read it;
    so is a model with neither, where sentence-transformers would take the tokenizer's alone,
    which may be no limit at all."""
    positions = files.config(TRANSFORMER_CONFIG_FILE, TransformerConfig).max_position_embeddings
    if config.max_seq_length is None:
        if positions is None:
            message = (
                f"{files.model_name} names no limit of its own ({SENTENCE_BERT_CONFIG_FILE}), "
                f"and its transformer no positions ({TRANSFORMER_CONFIG_FILE})"
            )
            raise UnusableModelError(message)
        tokenizer_limit = files.config(TOKENIZER_CONFIG_FILE, TokenizerConfig).model_max_length
        return min(tokenizer_limit, positions)
    if positions is not None and config.max_seq_length > positions:
        message = (
            f"{files.model_name}'s {SENTENCE_BERT_CONFIG_FILE} cuts a text at "
            f"{config.max_seq_length} tokens, past the {positions} positions its transformer "
            f"has ({TRANSFORMER_CONFIG_FILE})"
        )
        raise UnusableModelError(message)
    return config.max_seq_length


def lowercasing_first(normalizer: Normalizer | None) -> Normalizer:
    """A tokenizer's normalizer with lowercasing put before its own steps, as
    sentence-transformers puts it, unless the normalizer lowercases already: a step before
    its own `Lowercase` may read case, as a `Replace` of a capital does."""
    if normalizer is None:
        return Lowercase()
    if lowercases(normalizer):
        return normalizer
    return Sequence([Lowercase(), normalizer])


def lowercases(normalizer: Normalizer) -> bool:
    """Whether a normalizer is a `Lowercase`, or a sequence holding one, as
    sentence-transformers looks for it."""
    if isinstance(normalizer, Lowercase):
        return True
    return isinstance(normalizer, Sequence) and any(
        isinstance(step, Lowercase) for step in normalizer
    )


def inference_session(files: ModelFiles) -> onnxruntime.InferenceSession:
    """An onnxruntime session over the model's ONNX export, on the CPU."""
    options = onnxruntime.SessionOptions()
    options.add_session_config_entry(SPINNING_CONFIG_KEY, NO_SPINNING)
    export = files.path(ONNX_MODEL_FILE)
    try:
        return onnxruntime.InferenceSession(str(export), options, providers=[CPU_PROVIDER])
    except EXPORT_ERRORS as error:
        message = f"{files.model_name}'s {ONNX_MODEL_FILE} does not load in onnxruntime: {error}"
        raise UnusableModelError(message) from error


def check_inputs(files: ModelFiles, session: onnxruntime.InferenceSession) -> None:
    """Refuse an export the embedder cannot feed, which would fail at the first text."""
    inputs = frozenset(node.name for node in session.get_inputs())
    if not NEEDED_INPUTS <= inputs <= SENT_INPUTS:
        taken = LIST_SEPARATOR.join(sorted(inputs))
        message = (
            f"{files.model_name}'s {ONNX_MODEL_FILE} takes {taken}: the embedder sends "
            f"{INPUT_IDS} and {ATTENTION_MASK}, and {TOKEN_TYPE_IDS} where taken"
        )
        raise UnusableModelError(message)


def check_states(files: ModelFiles, model: OnnxModel, width: int) -> None:
    """Refuse an export whose first output, for one short text, is not that text's token
    states as wide as the pooling's config says: a vector per text, or logits, would give
    vectors that mean nothing, unseen while a search embeds a single text."""
    try:
        states = model.token_states([EXPORT_PROBE_TEXT])
    except EXPORT_ERRORS as error:
        message = f"{files.model_name}'s {ONNX_MODEL_FILE} does not run in onnxruntime: {error}"
        raise UnusableModelError(message) from error
    tokens = model.token_count(EXPORT_PROBE_TEXT)
    expected = (ONE_TEXT, tokens, width)
    if states.shape != expected:
        message = (
            f"{files.model_name}'s {ONNX_MODEL_FILE} gives one text of {tokens} tokens "
            f"a first output of shape {states.shape}: the embedder reads each token's state, "
            f"{expected}, as wide as {POOLING_CONFIG_FILE} says"
        )
        raise UnusableModelError(message)


def file_in_directory(directory: Path, relative_path: str) -> Path:
    """One of the files of a model kept in a directory, which must hold it."""
    path = directory / relative_path
    if not path.is_file():
        message = f"the model in {directory} has no {relative_path}"
        raise UnusableModelError(message)
    return path


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
