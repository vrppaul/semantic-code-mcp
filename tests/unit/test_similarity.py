"""Tests for the similarity of stored vectors to a query's."""

import numpy
import pytest
from hypothesis import given, strategies
from hypothesis.extra import numpy as numpy_strategies

from semantic_code_mcp.domain.models import HIGHEST_SIMILARITY, LOWEST_SIMILARITY
from semantic_code_mcp.storage.lancedb import cosine_similarities

WIDTH = 8
ROWS = 5
COMPONENTS = strategies.floats(min_value=-100, max_value=100, width=32)
VECTORS = numpy_strategies.arrays(numpy.float32, (ROWS, WIDTH), elements=COMPONENTS)
QUERIES = numpy_strategies.arrays(numpy.float32, WIDTH, elements=COMPONENTS)


@given(vectors=VECTORS, query=QUERIES)
def test_similarity_stays_between_opposite_and_the_same(
    vectors: numpy.ndarray, query: numpy.ndarray
):
    similarities = cosine_similarities(vectors, query)

    assert bool(
        numpy.all((similarities >= LOWEST_SIMILARITY) & (similarities <= HIGHEST_SIMILARITY))
    )


def test_a_vector_is_the_same_as_itself_and_opposite_to_its_negation_never_past_either():
    # In float32 this query's similarity to its negation computes to a hair below -1
    query = numpy.array([-0.6, -0.5, -0.2], dtype=numpy.float32)

    similarities = cosine_similarities(numpy.stack([query * 3, -query]), query)

    assert similarities.tolist() == pytest.approx([HIGHEST_SIMILARITY, LOWEST_SIMILARITY])
    assert bool(
        numpy.all((similarities >= LOWEST_SIMILARITY) & (similarities <= HIGHEST_SIMILARITY))
    )
