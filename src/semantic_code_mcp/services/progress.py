"""Progress of a long request, reported to the MCP client that asked."""

from collections.abc import Awaitable, Callable

PROGRESS_TOTAL = 100.0
PROGRESS_START = 0.0

# What an MCP client's progress callback takes: progress, total, message
type ProgressCallback = Callable[[float, float, str], Awaitable[None]]


class ProgressReporter:
    """Reports progress to a client's callback, or nowhere.

    A reporter speaks in its own scale of 0 to PROGRESS_TOTAL; one made by `within`
    maps that scale into a band of its parent's, so a step that runs inside another
    (an index run inside a search) never makes the client's progress go backwards.
    """

    def __init__(
        self,
        callback: ProgressCallback | None,
        *,
        band_start: float = PROGRESS_START,
        band_end: float = PROGRESS_TOTAL,
    ) -> None:
        self._callback = callback
        self._band_start = band_start
        self._band_end = band_end

    async def report(self, progress: float, message: str) -> None:
        if self._callback is None:
            return
        share = progress / PROGRESS_TOTAL
        overall = self._band_start + share * (self._band_end - self._band_start)
        await self._callback(overall, PROGRESS_TOTAL, message)

    def within(self, start: float, end: float) -> ProgressReporter:
        """A reporter whose whole scale is the band from start to end of this one's."""
        share_of_band = (self._band_end - self._band_start) / PROGRESS_TOTAL
        return ProgressReporter(
            self._callback,
            band_start=self._band_start + start * share_of_band,
            band_end=self._band_start + end * share_of_band,
        )


def share_done(done: int, total: int) -> float:
    """How far along work of `total` items is once `done` are behind it, on the scale of a
    reporter; at its start where there is nothing to do."""
    return PROGRESS_START if total == 0 else done / total * PROGRESS_TOTAL
