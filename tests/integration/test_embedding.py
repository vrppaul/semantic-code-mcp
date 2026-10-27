"""Tests for embeddings with the real model."""

import json
import shutil
from pathlib import Path

import huggingface_hub.constants
import numpy
import pytest
from huggingface_hub import hf_hub_download

from semantic_code_mcp.embedding import (
    ALL_MINILM_L6_V2_REVISION,
    MODULES_FILE,
    ONNX_MODEL_FILE,
    POOLING_CONFIG_FILE,
    SENTENCE_BERT_CONFIG_FILE,
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
FIRST_TOKEN_POOLING = {"pooling_mode_cls_token": True}
MEAN_AND_FIRST_TOKEN_POOLING = {"pooling_mode_mean_tokens": True, "pooling_mode_cls_token": True}
MEAN_AND_LAST_TOKEN_POOLING = {"pooling_mode_mean_tokens": True, "pooling_mode_lasttoken": True}
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
# How sentence-transformers 5.7 saves a model: its modules where that version moved them,
# the pooling by name, and the limit left to the tokenizer's config
MODULES_OF_SENTENCE_TRANSFORMERS_5_7 = [
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
POOLING_BY_NAME = {"pooling_mode": "mean"}
NO_LIMIT_OF_ITS_OWN = {"do_lower_case": False}
TOKENIZER_LIMIT = {"model_max_length": MODEL_TOKEN_LIMIT}
# transformers' "no limit", which some tokenizers' configs hold (thenlper/gte-small's),
# and the limit the transformer's positions set
NO_TOKENIZER_LIMIT = {"model_max_length": 10**30}
LIMIT_OF_THE_POSITIONS = {"max_seq_length": TRANSFORMER_POSITIONS}
# A model that says to lowercase its texts, as cased RoBERTa models do, over a tokenizer
# that keeps their case: by its normalizer, or for want of one
LOWERCASING_CONFIG = {"max_seq_length": MODEL_TOKEN_LIMIT, "do_lower_case": True}
CASE_KEEPING_NORMALIZER = {
    "type": "BertNormalizer",
    "clean_text": True,
    "handle_chinese_chars": True,
    "strip_accents": True,
    "lowercase": False,
}
NORMALIZER_FIELD = "normalizer"
TEXTS_WITH_CAPITALS = ["class DatabaseConnection: pass", "Hello World, CamelCase IDENTIFIERS"]
# An export whose weights lie in a file beside it, which a directory without that file lacks
EXTERNAL_WEIGHTS_REPOSITORY = "intfloat/multilingual-e5-large"
EXTERNAL_WEIGHTS_REVISION = "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3"
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
    ("relative_path", "content", "named"),
    [
        pytest.param(POOLING_CONFIG_FILE, FIRST_TOKEN_POOLING, "pools", id="the first token"),
        pytest.param(
            POOLING_CONFIG_FILE,
            MEAN_AND_FIRST_TOKEN_POOLING,
            "pools",
            id="the mean and the first token",
        ),
        pytest.param(
            POOLING_CONFIG_FILE,
            MEAN_AND_LAST_TOKEN_POOLING,
            "pools",
            id="the mean and the last token",
        ),
        pytest.param(MODULES_FILE, DENSE_AFTER_POOLING, "chains", id="a layer after the pooling"),
        pytest.param(
            MODULES_FILE,
            TRANSFORMER_OF_ITS_OWN_CODE,
            "chains",
            id="a transformer of the model's own code",
        ),
        pytest.param(MODULES_FILE, MODULE_OF_NO_TYPE, MODULES_FILE, id="a module of no type"),
    ],
)
def test_a_model_the_embedder_cannot_make_the_vectors_of_is_refused_when_it_loads(
    tmp_path: Path, relative_path: str, content: object, named: str
):
    """Its vectors would mean something else than its own library makes of them."""
    directory = copy_of_the_model(tmp_path)
    (directory / relative_path).write_text(json.dumps(content))

    with pytest.raises(UnusableModelError, match=named):
        OnnxEmbedder(str(directory)).load()


def test_a_model_saved_by_sentence_transformers_5_7_embeds_as_its_older_save(
    embedder: OnnxEmbedder, tmp_path: Path
):
    """It names its modules where that version moved them and its pooling instead of
    flagging it, and leaves the limit to the tokenizer."""
    directory = copy_of_the_model(tmp_path)
    (directory / MODULES_FILE).write_text(json.dumps(MODULES_OF_SENTENCE_TRANSFORMERS_5_7))
    (directory / POOLING_CONFIG_FILE).write_text(json.dumps(POOLING_BY_NAME))
    (directory / SENTENCE_BERT_CONFIG_FILE).write_text(json.dumps(NO_LIMIT_OF_ITS_OWN))
    (directory / TOKENIZER_CONFIG_FILE).write_text(json.dumps(TOKENIZER_LIMIT))
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
    "normalizer",
    [
        pytest.param(CASE_KEEPING_NORMALIZER, id="over a normalizer that keeps case"),
        pytest.param(None, id="for want of a normalizer"),
    ],
)
def test_a_model_that_says_to_lowercase_embeds_capitals_as_a_lowercasing_tokenizer_does(
    embedder: OnnxEmbedder, tmp_path: Path, normalizer: object
):
    """sentence-transformers lowercases a text first when the model's config says so."""
    directory = copy_of_the_model(tmp_path)
    (directory / SENTENCE_BERT_CONFIG_FILE).write_text(json.dumps(LOWERCASING_CONFIG))
    tokenizer = json.loads((directory / TOKENIZER_FILE).read_text())
    (directory / TOKENIZER_FILE).write_text(json.dumps(tokenizer | {NORMALIZER_FIELD: normalizer}))

    vectors = OnnxEmbedder(str(directory)).embed_documents(TEXTS_WITH_CAPITALS)

    assert numpy.array_equal(vectors, embedder.embed_documents(TEXTS_WITH_CAPITALS))


def test_a_model_directory_without_its_onnx_export_is_refused_naming_the_file(tmp_path: Path):
    """Else the agent would read only that the tool failed."""
    directory = copy_of_the_model(tmp_path)
    (directory / ONNX_MODEL_FILE).unlink()

    with pytest.raises(UnusableModelError, match=ONNX_MODEL_FILE):
        OnnxEmbedder(str(directory)).load()


@pytest.mark.parametrize(
    ("repository", "relative_path", "revision"),
    [
        pytest.param(
            HUB_REPOSITORY,
            TOKENIZER_FILE,
            ALL_MINILM_L6_V2_REVISION,
            id="a file that is no ONNX model",
        ),
        pytest.param(
            EXTERNAL_WEIGHTS_REPOSITORY,
            ONNX_MODEL_FILE,
            EXTERNAL_WEIGHTS_REVISION,
            id="an export whose weights lie in a missing file",
        ),
    ],
)
def test_an_export_onnxruntime_cannot_load_is_refused_naming_the_file(
    tmp_path: Path, repository: str, relative_path: str, revision: str
):
    """onnxruntime's own errors would reach the agent as no more than a failed tool."""
    directory = copy_of_the_model(tmp_path)
    in_its_place = hf_hub_download(repository, relative_path, revision=revision)
    shutil.copyfile(in_its_place, directory / ONNX_MODEL_FILE)

    with pytest.raises(UnusableModelError, match=ONNX_MODEL_FILE):
        OnnxEmbedder(str(directory)).load()


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
