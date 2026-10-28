"""How close a stored vector is to a query's, which every store reports of the candidates it
finds, whichever way it found them: the product of the two scaled to length one, so a store
that holds its vectors scaled pays one product per search."""

import numpy
from numpy.typing import NDArray

from semantic_code_mcp.domain.interfaces import Vector, VectorMatrix
from semantic_code_mcp.domain.models import HIGHEST_SIMILARITY, LOWEST_SIMILARITY

# The axis of a vector matrix along one vector: rows are chunks, columns components
VECTOR_LENGTH_AXIS = 1
# The length of a vector of zeros, which has no direction to compare
NO_LENGTH = 0.0
# Each row's product with the query, by numpy's own loop on the calling thread: BLAS runs a
# product of 88k rows on every core and leaves its threads spinning after it, which loads
# the user's machine and slowed the search's next query from 38 to 73 ms
ROW_PRODUCTS = "ij,j->i"


def cosine_similarities(vectors: VectorMatrix, query: Vector) -> NDArray[numpy.float32]:
    """The cosine similarity of each row's vector to the query's."""
    return unit_similarities(unit_rows(vectors), unit_vector(query))


def unit_similarities(unit_vectors: VectorMatrix, unit_query: Vector) -> NDArray[numpy.float32]:
    """The cosine similarity of each row to the query, both scaled by `unit_rows`: their
    product, 0 where either is all zeros, held to its range, which rounding oversteps by a
    hair."""
    products = numpy.einsum(ROW_PRODUCTS, unit_vectors, unit_query)
    return numpy.clip(products, LOWEST_SIMILARITY, HIGHEST_SIMILARITY)


def unit_rows(vectors: VectorMatrix) -> VectorMatrix:
    """Each row's vector scaled to length one; a row of zeros, which points nowhere, stays
    zeros."""
    lengths = numpy.linalg.norm(vectors, axis=VECTOR_LENGTH_AXIS, keepdims=True)
    return numpy.divide(vectors, lengths, out=numpy.zeros_like(vectors), where=lengths > NO_LENGTH)


def unit_vector(vector: Vector) -> Vector:
    """A vector scaled to length one, or zeros as it was."""
    return unit_rows(vector[numpy.newaxis, :])[0]
