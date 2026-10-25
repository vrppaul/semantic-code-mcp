"""Profiling of requests with pyinstrument, for development (`SEMANTIC_CODE_MCP_PROFILE`)."""

from datetime import UTC, datetime
from pathlib import Path

import structlog
from mcp.server.context import CallNext, HandlerResult, ServerRequestContext

log = structlog.get_logger()

PROFILE_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S_%f"
PROFILE_SUFFIX = ".txt"
# Characters of an MCP method name that cannot stand in a file name
METHOD_SEPARATOR = "/"
FILE_NAME_SEPARATOR = "_"


class ProfilingMiddleware:
    """Profiles each request the server handles and writes the report as text."""

    def __init__(self, profiles_directory: Path) -> None:
        self._profiles_directory = profiles_directory
        self._profiles_directory.mkdir(parents=True, exist_ok=True)
        log.info("profiling_enabled", profiles_dir=str(profiles_directory))

    async def __call__(
        self, ctx: ServerRequestContext[object, object], call_next: CallNext
    ) -> HandlerResult:
        # Deferred: pyinstrument is a dev dependency, imported only when profiling is on
        from pyinstrument import Profiler  # noqa: PLC0415

        profiler = Profiler(async_mode="enabled")
        profiler.start()
        try:
            return await call_next(ctx)
        finally:
            profiler.stop()
            self._save(profiler.output_text(unicode=True, color=False), ctx.method)

    def _save(self, report: str, method: str) -> None:
        timestamp = datetime.now(tz=UTC).strftime(PROFILE_TIMESTAMP_FORMAT)
        name = method.replace(METHOD_SEPARATOR, FILE_NAME_SEPARATOR)
        path = self._profiles_directory / f"{name}{FILE_NAME_SEPARATOR}{timestamp}{PROFILE_SUFFIX}"
        path.write_text(report)
        log.debug("profile_saved", path=str(path))
