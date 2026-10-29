"""Tests of how large a pool of processes is."""

import pytest

from semantic_code_mcp.pools import pool_workers


@pytest.mark.parametrize(
    ("cores", "workers"),
    [
        pytest.param(None, 0, id="cores the system cannot count"),
        pytest.param(3, 0, id="three cores"),
        pytest.param(4, 2, id="four cores"),
        pytest.param(6, 3, id="six cores"),
        pytest.param(16, 4, id="sixteen cores"),
    ],
)
def test_a_pool_takes_half_the_cores_at_most_four_and_none_where_that_is_fewer_than_two(
    cores: int | None, workers: int
):
    assert pool_workers(cores) == workers
