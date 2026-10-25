"""Durations as the services log and report them."""

import time

MILLISECONDS_PER_SECOND = 1000
MILLISECOND_DECIMALS = 1


def elapsed_milliseconds(started: float) -> float:
    """Milliseconds since a `time.perf_counter()` reading."""
    return round((time.perf_counter() - started) * MILLISECONDS_PER_SECOND, MILLISECOND_DECIMALS)
