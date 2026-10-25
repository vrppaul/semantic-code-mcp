"""Tests for embeddings with the real model."""

import numpy

from semantic_code_mcp.embedding import SentenceTransformerEmbedder
from tests.conftest import MODEL_NAME
from tests.fakes import EMBEDDING_DIMENSION


def cosine_similarity(first: numpy.ndarray, second: numpy.ndarray) -> float:
    return float(first @ second / (numpy.linalg.norm(first) * numpy.linalg.norm(second)))


def test_the_model_is_not_loaded_before_the_first_text():
    assert SentenceTransformerEmbedder(MODEL_NAME).is_loaded is False


def test_a_query_becomes_one_vector_of_the_model_dimension(embedder: SentenceTransformerEmbedder):
    assert embedder.embed_query("def hello(): pass").shape == (EMBEDDING_DIMENSION,)


def test_documents_become_one_row_per_text(embedder: SentenceTransformerEmbedder):
    vectors = embedder.embed_documents(["def hello(): pass", "class Foo: pass", "x = 1"])

    assert vectors.shape == (3, EMBEDDING_DIMENSION)


def test_texts_about_one_thing_are_closer_than_texts_about_another(
    embedder: SentenceTransformerEmbedder,
):
    addition = embedder.embed_query("def add(a, b): return a + b")
    summation = embedder.embed_query("def sum_numbers(x, y): return x + y")
    database = embedder.embed_query("class DatabaseConnection: pass")

    assert cosine_similarity(addition, summation) > cosine_similarity(addition, database)
