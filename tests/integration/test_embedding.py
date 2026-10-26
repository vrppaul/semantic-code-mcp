"""Tests for embeddings with the real model."""

import json
import shutil
from pathlib import Path

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
    OnnxEmbedder,
    UnusableModelError,
)
from tests.conftest import MODEL_NAME
from tests.fakes import EMBEDDING_DIMENSION

# Vectors sentence-transformers gave the default model, before the embedder moved off it
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
    TOKENIZER_FILE,
    ONNX_MODEL_FILE,
)
# Chains the embedder does not read: poolings other than the mean alone, which
# sentence-transformers would take another way or join into a wider vector, and a layer
# after the pooling that changes the vector
FIRST_TOKEN_POOLING = {"pooling_mode_cls_token": True}
MEAN_AND_FIRST_TOKEN_POOLING = {"pooling_mode_mean_tokens": True, "pooling_mode_cls_token": True}
MEAN_AND_LAST_TOKEN_POOLING = {"pooling_mode_mean_tokens": True, "pooling_mode_lasttoken": True}
DENSE_AFTER_POOLING = [
    {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
    {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
    {"idx": 2, "name": "2", "path": "2_Dense", "type": "sentence_transformers.models.Dense"},
]
# How sentence-transformers 5 saves a model: the pooling by name, and the limit left to the
# tokenizer's config
POOLING_BY_NAME = {"pooling_mode": "mean"}
NO_LIMIT_OF_ITS_OWN = {"do_lower_case": False}
TOKENIZER_LIMIT = {"model_max_length": 256}


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


def test_a_model_saved_by_sentence_transformers_5_embeds_as_its_older_save(
    embedder: OnnxEmbedder, tmp_path: Path
):
    """It names its pooling instead of flagging it, and leaves the limit to the tokenizer."""
    directory = copy_of_the_model(tmp_path)
    (directory / POOLING_CONFIG_FILE).write_text(json.dumps(POOLING_BY_NAME))
    (directory / SENTENCE_BERT_CONFIG_FILE).write_text(json.dumps(NO_LIMIT_OF_ITS_OWN))
    (directory / TOKENIZER_CONFIG_FILE).write_text(json.dumps(TOKENIZER_LIMIT))
    texts = json.loads(REFERENCE_VECTORS.read_text())["texts"]

    vectors = OnnxEmbedder(str(directory)).embed_documents(texts)

    assert numpy.array_equal(vectors, embedder.embed_documents(texts))


def test_a_model_directory_without_its_onnx_export_is_refused_naming_the_file(tmp_path: Path):
    """Else the agent would read only that the tool failed."""
    directory = copy_of_the_model(tmp_path)
    (directory / ONNX_MODEL_FILE).unlink()

    with pytest.raises(UnusableModelError, match=ONNX_MODEL_FILE):
        OnnxEmbedder(str(directory)).load()
