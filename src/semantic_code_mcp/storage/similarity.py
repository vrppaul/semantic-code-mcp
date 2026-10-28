"""How close a stored vector is to a query's, which every store reports of the candidates it
finds, whichever way it found them."""

import numpy
from numpy.typing import NDArray

from semantic_code_mcp.domain.interfaces import Vector, VectorMatrix
from semantic_code_mcp.domain.models import HIGHEST_SIMILARITY, LOWEST_SIMILARITY

# The axis of a vector matrix along one vector: rows are chunks, columns components
VECTOR_LENGTH_AXIS = 1
# The length of a vector of zeros, which has no direction to compare
NO_LENGTH = 0.0


def cosine_similarities(vectors: VectorMatrix, query: Vector) -> NDArray[numpy.float32]:
    """The cosine similarity of each row's vector to the query's: 0 where either is all
    zeros, which points nowhere, and held to its range, which rounding oversteps by a hair."""
    norms = numpy.linalg.norm(vectors, axis=VECTOR_LENGTH_AXIS) * numpy.linalg.norm(query)
    products = vectors @ query
    similarities = numpy.divide(
        products, norms, out=numpy.zeros_like(products), where=norms > NO_LENGTH
    )
    return numpy.clip(similarities, LOWEST_SIMILARITY, HIGHEST_SIMILARITY)
