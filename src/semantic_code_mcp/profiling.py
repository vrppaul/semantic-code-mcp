"""Profiling utilities for development."""

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import ParamSpec, TypeVar

import structlog

log = structlog.get_logger()

P = ParamSpec("P")
T = TypeVar("T")

# Global flag - set by server.py based on settings
_profiling_enabled = False
_profiles_dir: Path | None = None


def configure_profiling(enabled: bool, profiles_dir: Path | None = None) -> None:
    """Configure profiling settings.

    Args:
        enabled: Whether profiling is enabled.
        profiles_dir: Directory to save profile reports. Defaults to ./profiles.
    """
    global _profiling_enabled, _profiles_dir
    _profiling_enabled = enabled
    _profiles_dir = profiles_dir or Path("profiles")

    if enabled:
        _profiles_dir.mkdir(parents=True, exist_ok=True)
        log.info("profiling_enabled", profiles_dir=str(_profiles_dir))


def profile_async(name: str) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator to profile an async function.

    When profiling is enabled, wraps the function with pyinstrument profiler
    and saves the text report to the profiles directory.

    Args:
        name: Name for the profile output file.

    Returns:
        Decorator function.
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if not _profiling_enabled:
                return await func(*args, **kwargs)

            # Import here to avoid loading pyinstrument when not profiling
            from pyinstrument import Profiler  # noqa: PLC0415

            profiler = Profiler(async_mode="enabled")
            profiler.start()

            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                profiler.stop()
                _save_profile(profiler, name)

        return wrapper

    return decorator


def _save_profile(profiler, name: str) -> None:
    """Save profiler output to a text file.

    Args:
        profiler: The pyinstrument Profiler instance.
        name: Base name for the output file.
    """
    if _profiles_dir is None:
        return

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.txt"
    filepath = _profiles_dir / filename

    # Get text output
    output = profiler.output_text(unicode=True, color=False)

    filepath.write_text(output)
    log.debug("profile_saved", path=str(filepath))
