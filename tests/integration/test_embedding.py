"""Tests for embeddings with the real model."""

import math

from semantic_code_mcp.embedding import SentenceTransformerEmbedder
from tests.conftest import MODEL_NAME
from tests.fakes import EMBEDDING_DIMENSION


def cosine_similarity(first: list[float], second: list[float]) -> float:
    dot_product = sum(left * right for left, right in zip(first, second, strict=True))
    return dot_product / (math.hypot(*first) * math.hypot(*second))


def test_the_model_is_not_loaded_before_the_first_text():
    assert SentenceTransformerEmbedder(MODEL_NAME).is_loaded is False


def test_a_text_becomes_one_vector_of_the_model_dimension(embedder: SentenceTransformerEmbedder):
    assert len(embedder.embed_text("def hello(): pass")) == EMBEDDING_DIMENSION


def test_a_batch_becomes_one_vector_per_text(embedder: SentenceTransformerEmbedder):
    vectors = embedder.embed_batch(["def hello(): pass", "class Foo: pass", "x = 1"])

    assert [len(vector) for vector in vectors] == [EMBEDDING_DIMENSION] * 3


def test_an_empty_batch_becomes_no_vectors(embedder: SentenceTransformerEmbedder):
    assert embedder.embed_batch([]) == []


def test_texts_about_one_thing_are_closer_than_texts_about_another(
    embedder: SentenceTransformerEmbedder,
):
    addition = embedder.embed_text("def add(a, b): return a + b")
    summation = embedder.embed_text("def sum_numbers(x, y): return x + y")
    database = embedder.embed_text("class DatabaseConnection: pass")

    assert cosine_similarity(addition, summation) > cosine_similarity(addition, database)
