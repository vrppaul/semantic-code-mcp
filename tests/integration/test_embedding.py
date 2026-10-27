"""Tests for embeddings with the real model."""

import json
import shutil
from pathlib import Path

import huggingface_hub.constants
import numpy
import onnx
import pytest
from huggingface_hub import hf_hub_download
from onnx import TensorProto, helper

from semantic_code_mcp.embedding import (
    ALL_MINILM_L6_V2_REVISION,
    ATTENTION_MASK,
    INPUT_IDS,
    MODULES_FILE,
    ONNX_MODEL_FILE,
    POOLING_CONFIG_FILE,
    SENTENCE_BERT_CONFIG_FILE,
    TOKEN_TYPE_IDS,
    TOKENIZER_CONFIG_FILE,
    TOKENIZER_FILE,
    TRANSFORMER_CONFIG_FILE,
    ModelFiles,
    OnnxEmbedder,
    UnusableModelError,
)
from tests.conftest import MODEL_NAME
from tests.fakes import EMBEDDING_DIMENSION

# Vectors sentence-transformers gave the default model, before the embedder moved off it;
# one of its texts runs to 1,442 tokens, past every limit a test sets
REFERENCE_VECTORS = Path(__file__).parent.parent / "data" / "all_minilm_l6_v2_vectors.json"
# How far a component may stray from the reference: batching alone moves it by 6e-8
COMPONENT_TOLERANCE = 1e-5
HUB_REPOSITORY = f"sentence-transformers/{MODEL_NAME}"
# The files the embedder reads of a model, which a test copies into a directory
MODEL_FILES = (
    MODULES_FILE,
    POOLING_CONFIG_FILE,
    SENTENCE_BERT_CONFIG_FILE,
    TOKENIZER_CONFIG_FILE,
    TRANSFORMER_CONFIG_FILE,
    TOKENIZER_FILE,
    ONNX_MODEL_FILE,
)
# The default model's own limit, in tokens, and the positions its transformer has
MODEL_TOKEN_LIMIT = 256
TRANSFORMER_POSITIONS = 512
# Chains the embedder does not read: poolings other than the mean alone, which
# sentence-transformers would take another way or join into a wider vector, a layer after
# the pooling that changes the vector, and a transformer of the model's own code
FIRST_TOKEN_POOLING = {
    "word_embedding_dimension": EMBEDDING_DIMENSION,
    "pooling_mode_cls_token": True,
}
MEAN_AND_FIRST_TOKEN_POOLING = {
    "word_embedding_dimension": EMBEDDING_DIMENSION,
    "pooling_mode_mean_tokens": True,
    "pooling_mode_cls_token": True,
}
MEAN_AND_LAST_TOKEN_POOLING = {
    "word_embedding_dimension": EMBEDDING_DIMENSION,
    "pooling_mode_mean_tokens": True,
    "pooling_mode_lasttoken": True,
}
DENSE_AFTER_POOLING = [
    {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
    {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
    {"idx": 2, "name": "2", "path": "2_Dense", "type": "sentence_transformers.models.Dense"},
]
TRANSFORMER_OF_ITS_OWN_CODE = [
    {"idx": 0, "name": "0", "path": "", "type": "custom_st.Transformer"},
    {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
    {
        "idx": 2,
        "name": "2",
        "path": "2_Normalize",
        "type": "sentence_transformers.models.Normalize",
    },
]
# A chain whose module names no type
MODULE_OF_NO_TYPE = [{"idx": 0, "name": "0", "path": ""}]
# How sentence-transformers 5.4 and later save a model: its modules where 5.4 moved them,
# the pooling by name, and the limit left to the tokenizer's config
MODULES_OF_SENTENCE_TRANSFORMERS_5_4 = [
    {
        "idx": 0,
        "name": "0",
        "path": "",
        "type": "sentence_transformers.base.modules.transformer.Transformer",
    },
    {
        "idx": 1,
        "name": "1",
        "path": "1_Pooling",
        "type": "sentence_transformers.sentence_transformer.modules.pooling.Pooling",
    },
    {
        "idx": 2,
        "name": "2",
        "path": "2_Normalize",
        "type": "sentence_transformers.sentence_transformer.modules.normalize.Normalize",
    },
]
POOLING_BY_NAME = {"embedding_dimension": EMBEDDING_DIMENSION, "pooling_mode": "mean"}
NO_LIMIT_OF_ITS_OWN = {"do_lower_case": False}
TOKENIZER_LIMIT = {"model_max_length": MODEL_TOKEN_LIMIT}
# transformers' "no limit", which some tokenizers' configs hold (thenlper/gte-small's),
# and the limit the transformer's positions set
NO_TOKENIZER_LIMIT = {"model_max_length": 10**30}
LIMIT_OF_THE_POSITIONS = {"max_seq_length": TRANSFORMER_POSITIONS}
# A limit past the positions, which the transformer could not read
LIMIT_PAST_THE_POSITIONS = {"max_seq_length": 2 * TRANSFORMER_POSITIONS}
# A transformer's config that names its positions otherwise than BERT's does, as
# nomic-bert's `n_positions`, or not at all
POSITIONS_NAMED_OTHERWISE = {"model_type": "nomic_bert", "n_positions": TRANSFORMER_POSITIONS}
# A model that says to lowercase its texts, as cased RoBERTa models do, or that does not,
# over a tokenizer that keeps their case: by its normalizer, or for want of one
LOWERCASING_CONFIG = {"max_seq_length": MODEL_TOKEN_LIMIT, "do_lower_case": True}
CASE_KEEPING_CONFIG = {"max_seq_length": MODEL_TOKEN_LIMIT, "do_lower_case": False}
CASE_KEEPING_NORMALIZER = {
    "type": "BertNormalizer",
    "clean_text": True,
    "handle_chinese_chars": True,
    "strip_accents": True,
    "lowercase": False,
}
NORMALIZER_FIELD = "normalizer"
TEXTS_WITH_CAPITALS = ["class DatabaseConnection: pass", "Hello World, CamelCase IDENTIFIERS"]
# A normalizer that lowercases on its own, after a step that reads case: lowercasing first
# would hide the capital the step replaces, so sentence-transformers adds no step to it
REPLACING_THEN_LOWERCASING = {
    "type": "Sequence",
    "normalizers": [
        {"type": "Replace", "pattern": {"String": "X"}, "content": "y"},
        {"type": "Lowercase"},
    ],
}
TEXTS_WITH_THE_REPLACED_CAPITAL = ["X marks the spot", "a matrix X of floats"]
# An export whose weights lie in a file beside it, which a directory without that file lacks
EXTERNAL_WEIGHTS_REPOSITORY = "intfloat/multilingual-e5-large"
EXTERNAL_WEIGHTS_REVISION = "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3"
# Exports made by hand, which onnxruntime loads but the embedder cannot use: an opset and
# an IR version onnxruntime reads, a graph that turns the token ids into numbers and
# reshapes them, each 0 of the shape keeping that axis as it is
OPSET = 17
IR_VERSION = 8
TOKEN_STATES = "token_states"
IDS_AS_NUMBERS = "ids_as_numbers"
STATES_SHAPE = "states_shape"
TEXTS_BY_TOKENS = ["texts", "tokens"]
TEXTS_BY_TOKENS_BY_WIDTH = [0, 0, 1]
TEXTS_BY_WIDTH = [0, 0]
INPUT_SHAPE = "input_shape"
# An input some exports take, which the embedder never sends
POSITION_IDS = "position_ids"
ZERO_TOKEN_TYPE = "zero_token_type"
# A file no model's repository holds
ABSENT_FILE = "onnx/absent.onnx"
# The hub's settings a test sets: offline, and the directory of its cache
OFFLINE_SETTING = "HF_HUB_OFFLINE"
CACHE_SETTING = "HF_HUB_CACHE"


def cosine_similarity(first: numpy.ndarray, second: numpy.ndarray) -> float:
    return float(first @ second / (numpy.linalg.norm(first) * numpy.linalg.norm(second)))


def copy_of_the_model(directory: Path) -> Path:
    """The default model's files, as the embedder reads them, in a directory of their own."""
    for relative_path in MODEL_FILES:
        cached = hf_hub_download(HUB_REPOSITORY, relative_path, revision=ALL_MINILM_L6_V2_REVISION)
        (directory / relative_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, directory / relative_path)
    return directory


def copy_with_case(directory: Path, config: object, normalizer: object) -> Path:
    """The default model, with the sentence-transformers config given and its tokenizer's
    normalizer replaced by the one given."""
    copy_of_the_model(directory)
    (directory / SENTENCE_BERT_CONFIG_FILE).write_text(json.dumps(config))
    tokenizer = json.loads((directory / TOKENIZER_FILE).read_text())
    (directory / TOKENIZER_FILE).write_text(json.dumps(tokenizer | {NORMALIZER_FIELD: normalizer}))
    return directory


def write_tiny_export(
    path: Path, input_names: tuple[str, ...], input_type: int, states_shape: list[int]
) -> None:
    """An export taking the inputs named, of the type given, whose first output is the token
    ids as numbers in the shape given: texts by tokens by width, or by fewer axes."""
    graph = helper.make_graph(
        [
            helper.make_node("Cast", [INPUT_IDS], [IDS_AS_NUMBERS], to=TensorProto.FLOAT),
            helper.make_node("Reshape", [IDS_AS_NUMBERS, STATES_SHAPE], [TOKEN_STATES]),
        ],
        "tiny",
        [helper.make_tensor_value_info(name, input_type, TEXTS_BY_TOKENS) for name in input_names],
        [
            helper.make_tensor_value_info(
                TOKEN_STATES, TensorProto.FLOAT, [None] * len(states_shape)
            )
        ],
        [helper.make_tensor(STATES_SHAPE, TensorProto.INT64, [len(states_shape)], states_shape)],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", OPSET)], ir_version=IR_VERSION
    )
    onnx.save(model, str(path))


def drop_token_types(path: Path) -> None:
    """The export at the path, rewritten to take no token types and to make its own, all
    zero, in their place."""
    export = onnx.load(str(path))
    names = [node.name for node in export.graph.input]
    del export.graph.input[names.index(TOKEN_TYPE_IDS)]
    zero = helper.make_tensor(ZERO_TOKEN_TYPE, TensorProto.INT64, [1], [0])
    export.graph.node.insert(
        0, helper.make_node("ConstantOfShape", [INPUT_SHAPE], [TOKEN_TYPE_IDS], value=zero)
    )
    export.graph.node.insert(0, helper.make_node("Shape", [INPUT_IDS], [INPUT_SHAPE]))
    onnx.save(export, str(path))


def test_the_model_is_not_loaded_before_the_first_text():
    assert OnnxEmbedder(MODEL_NAME).is_loaded is False


def test_a_query_becomes_one_vector_of_the_model_dimension(embedder: OnnxEmbedder):
    assert embedder.embed_query("def hello(): pass").shape == (EMBEDDING_DIMENSION,)


def test_documents_become_one_row_per_text(embedder: OnnxEmbedder):
    vectors = embedder.embed_documents(["def hello(): pass", "class Foo: pass", "x = 1"])

    assert vectors.shape == (3, EMBEDDING_DIMENSION)


def test_texts_about_one_thing_are_closer_than_texts_about_another(embedder: OnnxEmbedder):
    addition = embedder.embed_query("def add(a, b): return a + b")
    summation = embedder.embed_query("def sum_numbers(x, y): return x + y")
    database = embedder.embed_query("class DatabaseConnection: pass")

    assert cosine_similarity(addition, summation) > cosine_similarity(addition, database)


def test_the_vectors_are_those_sentence_transformers_gave(embedder: OnnxEmbedder):
    """So an index built before keeps answering; one of the texts runs past the model's
    256 tokens, which both cut alike."""
    reference = json.loads(REFERENCE_VECTORS.read_text())

    vectors = embedder.embed_documents(reference["texts"])

    assert numpy.allclose(vectors, reference["vectors"], atol=COMPONENT_TOLERANCE)


def test_a_model_in_a_directory_embeds_as_the_hub_s(embedder: OnnxEmbedder, tmp_path: Path):
    local = OnnxEmbedder(str(copy_of_the_model(tmp_path)))

    vector = local.embed_query("def add(a, b): return a + b")

    assert numpy.array_equal(vector, embedder.embed_query("def add(a, b): return a + b"))


@pytest.mark.parametrize(
    ("changes", "named"),
    [
        pytest.param({POOLING_CONFIG_FILE: FIRST_TOKEN_POOLING}, "pools", id="the first token"),
        pytest.param(
            {POOLING_CONFIG_FILE: MEAN_AND_FIRST_TOKEN_POOLING},
            "pools",
            id="the mean and the first token",
        ),
        pytest.param(
            {POOLING_CONFIG_FILE: MEAN_AND_LAST_TOKEN_POOLING},
            "pools",
            id="the mean and the last token",
        ),
        pytest.param({MODULES_FILE: DENSE_AFTER_POOLING}, "chains", id="a layer after the pooling"),
        pytest.param(
            {MODULES_FILE: TRANSFORMER_OF_ITS_OWN_CODE},
            "custom_st.Transformer",
            id="a transformer of the model's own code",
        ),
        pytest.param({MODULES_FILE: MODULE_OF_NO_TYPE}, MODULES_FILE, id="a module of no type"),
        pytest.param(
            {SENTENCE_BERT_CONFIG_FILE: LIMIT_PAST_THE_POSITIONS},
            SENTENCE_BERT_CONFIG_FILE,
            id="a limit past the transformer's positions",
        ),
        pytest.param(
            {
                SENTENCE_BERT_CONFIG_FILE: NO_LIMIT_OF_ITS_OWN,
                TRANSFORMER_CONFIG_FILE: POSITIONS_NAMED_OTHERWISE,
            },
            TRANSFORMER_CONFIG_FILE,
            id="no limit of its own, and no positions known",
        ),
    ],
)
def test_a_model_the_embedder_cannot_make_the_vectors_of_is_refused_when_it_loads(
    tmp_path: Path, changes: dict[str, object], named: str
):
    """Its vectors would mean something else than its own library makes of them."""
    directory = copy_of_the_model(tmp_path)
    for relative_path, content in changes.items():
        (directory / relative_path).write_text(json.dumps(content))

    with pytest.raises(UnusableModelError, match=named):
        OnnxEmbedder(str(directory)).load()


def test_a_model_saved_by_sentence_transformers_5_4_or_later_embeds_as_its_older_save(
    embedder: OnnxEmbedder, tmp_path: Path
):
    """It names its modules where 5.4 moved them and its pooling instead of flagging it,
    and leaves the limit to the tokenizer."""
    directory = copy_of_the_model(tmp_path)
    (directory / MODULES_FILE).write_text(json.dumps(MODULES_OF_SENTENCE_TRANSFORMERS_5_4))
    (directory / POOLING_CONFIG_FILE).write_text(json.dumps(POOLING_BY_NAME))
    (directory / SENTENCE_BERT_CONFIG_FILE).write_text(json.dumps(NO_LIMIT_OF_ITS_OWN))
    (directory / TOKENIZER_CONFIG_FILE).write_text(json.dumps(TOKENIZER_LIMIT))
    texts = json.loads(REFERENCE_VECTORS.read_text())["texts"]

    vectors = OnnxEmbedder(str(directory)).embed_documents(texts)

    assert numpy.array_equal(vectors, embedder.embed_documents(texts))


def test_a_model_whose_transformer_names_its_positions_otherwise_reads_to_its_own_limit(
    embedder: OnnxEmbedder, tmp_path: Path
):
    """sentence-transformers holds a limit of the model's own to no positions it cannot
    find in the transformer's config."""
    directory = copy_of_the_model(tmp_path)
    (directory / TRANSFORMER_CONFIG_FILE).write_text(json.dumps(POSITIONS_NAMED_OTHERWISE))
    texts = json.loads(REFERENCE_VECTORS.read_text())["texts"]

    vectors = OnnxEmbedder(str(directory)).embed_documents(texts)

    assert numpy.array_equal(vectors, embedder.embed_documents(texts))


def test_a_tokenizer_of_no_limit_reads_as_many_tokens_as_the_transformer_has_positions(
    tmp_path: Path,
):
    """As sentence-transformers cuts it."""
    directory = copy_of_the_model(tmp_path / "unlimited")
    (directory / SENTENCE_BERT_CONFIG_FILE).write_text(json.dumps(NO_LIMIT_OF_ITS_OWN))
    (directory / TOKENIZER_CONFIG_FILE).write_text(json.dumps(NO_TOKENIZER_LIMIT))
    limited = copy_of_the_model(tmp_path / "limited")
    (limited / SENTENCE_BERT_CONFIG_FILE).write_text(json.dumps(LIMIT_OF_THE_POSITIONS))
    texts = json.loads(REFERENCE_VECTORS.read_text())["texts"]

    vectors = OnnxEmbedder(str(directory)).embed_documents(texts)

    assert numpy.array_equal(vectors, OnnxEmbedder(str(limited)).embed_documents(texts))


@pytest.mark.parametrize(
    ("config", "normalizer", "lowercased"),
    [
        pytest.param(
            LOWERCASING_CONFIG,
            CASE_KEEPING_NORMALIZER,
            True,
            id="said, over a normalizer that keeps case",
        ),
        pytest.param(LOWERCASING_CONFIG, None, True, id="said, for want of a normalizer"),
        pytest.param(
            CASE_KEEPING_CONFIG,
            CASE_KEEPING_NORMALIZER,
            False,
            id="not said, over a normalizer that keeps case",
        ),
    ],
)
def test_a_model_s_config_says_whether_capitals_embed_as_a_lowercasing_tokenizer_s(
    embedder: OnnxEmbedder, tmp_path: Path, config: object, normalizer: object, lowercased: bool
):
    """sentence-transformers lowercases a text first where the model's config says so, and
    only there."""
    directory = copy_with_case(tmp_path, config, normalizer)

    vectors = OnnxEmbedder(str(directory)).embed_documents(TEXTS_WITH_CAPITALS)

    assert numpy.array_equal(vectors, embedder.embed_documents(TEXTS_WITH_CAPITALS)) is lowercased


def test_a_model_that_says_to_lowercase_over_a_normalizer_that_lowercases_embeds_as_unsaid(
    tmp_path: Path,
):
    """sentence-transformers adds no lowercasing to a normalizer that holds one."""
    said = copy_with_case(tmp_path / "said", LOWERCASING_CONFIG, REPLACING_THEN_LOWERCASING)
    unsaid = copy_with_case(tmp_path / "unsaid", CASE_KEEPING_CONFIG, REPLACING_THEN_LOWERCASING)

    vectors = OnnxEmbedder(str(said)).embed_documents(TEXTS_WITH_THE_REPLACED_CAPITAL)

    unsaid_vectors = OnnxEmbedder(str(unsaid)).embed_documents(TEXTS_WITH_THE_REPLACED_CAPITAL)
    assert numpy.array_equal(vectors, unsaid_vectors)


def test_a_model_directory_without_its_onnx_export_is_refused_naming_the_file(tmp_path: Path):
    """Else the agent would read only that the tool failed."""
    directory = copy_of_the_model(tmp_path)
    (directory / ONNX_MODEL_FILE).unlink()

    with pytest.raises(UnusableModelError, match=ONNX_MODEL_FILE):
        OnnxEmbedder(str(directory)).load()


@pytest.mark.parametrize(
    ("target", "repository", "relative_path", "revision"),
    [
        pytest.param(
            TOKENIZER_FILE,
            HUB_REPOSITORY,
            ONNX_MODEL_FILE,
            ALL_MINILM_L6_V2_REVISION,
            id="a tokenizer file tokenizers cannot read",
        ),
        pytest.param(
            ONNX_MODEL_FILE,
            HUB_REPOSITORY,
            TOKENIZER_FILE,
            ALL_MINILM_L6_V2_REVISION,
            id="a file that is no ONNX model",
        ),
        pytest.param(
            ONNX_MODEL_FILE,
            EXTERNAL_WEIGHTS_REPOSITORY,
            ONNX_MODEL_FILE,
            EXTERNAL_WEIGHTS_REVISION,
            id="an export whose weights lie in a missing file",
        ),
    ],
)
def test_a_file_its_library_cannot_load_is_refused_naming_it(
    tmp_path: Path, target: str, repository: str, relative_path: str, revision: str
):
    """tokenizers' and onnxruntime's own errors would reach the agent as no more than a
    failed tool."""
    directory = copy_of_the_model(tmp_path)
    in_its_place = hf_hub_download(repository, relative_path, revision=revision)
    shutil.copyfile(in_its_place, directory / target)

    with pytest.raises(UnusableModelError, match=target):
        OnnxEmbedder(str(directory)).load()


@pytest.mark.parametrize(
    ("input_names", "input_type", "states_shape"),
    [
        pytest.param(
            (INPUT_IDS, ATTENTION_MASK, POSITION_IDS),
            TensorProto.INT64,
            TEXTS_BY_TOKENS_BY_WIDTH,
            id="an input the embedder does not send",
        ),
        pytest.param(
            (INPUT_IDS,),
            TensorProto.INT64,
            TEXTS_BY_TOKENS_BY_WIDTH,
            id="no attention mask to take",
        ),
        pytest.param(
            (INPUT_IDS, ATTENTION_MASK),
            TensorProto.INT32,
            TEXTS_BY_TOKENS_BY_WIDTH,
            id="inputs of another type than the token ids'",
        ),
        pytest.param(
            (INPUT_IDS, ATTENTION_MASK),
            TensorProto.INT64,
            TEXTS_BY_WIDTH,
            id="a vector per text as its first output",
        ),
        pytest.param(
            (INPUT_IDS, ATTENTION_MASK),
            TensorProto.INT64,
            TEXTS_BY_TOKENS_BY_WIDTH,
            id="token states of another width than the pooling's",
        ),
    ],
)
def test_an_export_the_embedder_cannot_feed_or_read_is_refused_when_it_loads(
    tmp_path: Path, input_names: tuple[str, ...], input_type: int, states_shape: list[int]
):
    """Else it would fail at the first text, or give one text's search a wrong vector."""
    directory = copy_of_the_model(tmp_path)
    write_tiny_export(directory / ONNX_MODEL_FILE, input_names, input_type, states_shape)

    with pytest.raises(UnusableModelError, match=ONNX_MODEL_FILE):
        OnnxEmbedder(str(directory)).load()


def test_an_export_that_takes_no_token_types_embeds_as_one_given_zeros(
    embedder: OnnxEmbedder, tmp_path: Path
):
    """DistilBERT's and MPNet's exports take none."""
    directory = copy_of_the_model(tmp_path)
    drop_token_types(directory / ONNX_MODEL_FILE)

    vectors = OnnxEmbedder(str(directory)).embed_documents(TEXTS_WITH_CAPITALS)

    assert numpy.array_equal(vectors, embedder.embed_documents(TEXTS_WITH_CAPITALS))


def test_a_file_the_model_s_hub_repository_lacks_is_refused_naming_it():
    with pytest.raises(UnusableModelError, match=ABSENT_FILE):
        ModelFiles(MODEL_NAME, ALL_MINILM_L6_V2_REVISION).path(ABSENT_FILE)


def test_offline_a_file_the_hub_s_cache_lacks_is_refused_naming_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(huggingface_hub.constants, OFFLINE_SETTING, True)
    monkeypatch.setattr(huggingface_hub.constants, CACHE_SETTING, str(tmp_path))

    with pytest.raises(UnusableModelError, match=MODULES_FILE):
        ModelFiles(MODEL_NAME, ALL_MINILM_L6_V2_REVISION).path(MODULES_FILE)
