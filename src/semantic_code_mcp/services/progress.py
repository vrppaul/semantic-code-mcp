"""Progress of a long request, reported to the MCP client that asked."""

from semantic_code_mcp.domain.interfaces import ProgressCallback

PROGRESS_TOTAL = 100


class ProgressReporter:
    """Reports progress, out of PROGRESS_TOTAL, to a client's callback, or nowhere."""

    def __init__(self, callback: ProgressCallback | None) -> None:
        self._callback = callback

    async def report(self, progress: float, message: str) -> None:
        if self._callback is not None:
            await self._callback(progress, PROGRESS_TOTAL, message)
