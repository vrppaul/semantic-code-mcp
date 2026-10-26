"""Tests for embeddings with the real model."""

import json
import shutil
from pathlib import Path

import numpy
import pytest
from huggingface_hub import hf_hub_download

from semantic_code_mcp.embedding import (
    ALL_MINILM_L6_V2_REVISION,
    MODEL_FILES,
    POOLING_CONFIG_FILE,
    OnnxEmbedder,
    UnsupportedModelError,
)
from tests.conftest import MODEL_NAME
from tests.fakes import EMBEDDING_DIMENSION

# Vectors sentence-transformers gave the default model, before the embedder moved off it
REFERENCE_VECTORS = Path(__file__).parent.parent / "data" / "all_minilm_l6_v2_vectors.json"
# How far a component may stray from the reference: batching alone moves it by 6e-8
COMPONENT_TOLERANCE = 1e-5
HUB_REPOSITORY = f"sentence-transformers/{MODEL_NAME}"
# Poolings the embedder does not read: the first token's state, as some models take it,
# and that beside the mean, which sentence-transformers would join into one wider vector
FIRST_TOKEN_POOLING = {"word_embedding_dimension": 384, "pooling_mode_cls_token": True}
MEAN_AND_FIRST_TOKEN_POOLING = {
    "word_embedding_dimension": 384,
    "pooling_mode_mean_tokens": True,
    "pooling_mode_cls_token": True,
}


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
    "pooling",
    [
        pytest.param(FIRST_TOKEN_POOLING, id="the first token"),
        pytest.param(MEAN_AND_FIRST_TOKEN_POOLING, id="the mean and the first token"),
    ],
)
def test_a_model_that_pools_otherwise_is_refused_when_it_loads(
    tmp_path: Path, pooling: dict[str, object]
):
    """Its vectors would mean something else than its own library makes of them."""
    directory = copy_of_the_model(tmp_path)
    (directory / POOLING_CONFIG_FILE).write_text(json.dumps(pooling))

    with pytest.raises(UnsupportedModelError, match="pool"):
        OnnxEmbedder(str(directory)).load()
