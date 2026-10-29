"""Which watcher of directories each system has, and on which file systems it sees every change
(ADR 010, O1): inotify on Linux, FSEvents on macOS, ReadDirectoryChangesW on Windows."""

import sys
from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict

from semantic_code_mcp.domain.interfaces import DirectoryWatcher
from semantic_code_mcp.watching.directory_changes import DirectoryChanges
from semantic_code_mcp.watching.file_systems import (
    linux_sees_every_change,
    macos_sees_every_change,
    windows_sees_every_change,
)
from semantic_code_mcp.watching.fsevents import FSEvents
from semantic_code_mcp.watching.inotify import Inotify

# The systems by the names `sys.platform` gives them
LINUX_PLATFORM = "linux"
MACOS_PLATFORM = "darwin"
WINDOWS_PLATFORM = "win32"


class SystemWatching(BaseModel):
    """A system's watcher of directories, and whether it sees every change under a real
    directory, by the file systems it lies on."""

    model_config = ConfigDict(frozen=True)

    watcher: type[DirectoryWatcher]
    sees_every_change: Callable[[Path], bool]


SYSTEMS = MappingProxyType(
    {
        LINUX_PLATFORM: SystemWatching(watcher=Inotify, sees_every_change=linux_sees_every_change),
        MACOS_PLATFORM: SystemWatching(watcher=FSEvents, sees_every_change=macos_sees_every_change),
        WINDOWS_PLATFORM: SystemWatching(
            watcher=DirectoryChanges, sees_every_change=windows_sees_every_change
        ),
    }
)


def system_watcher() -> type[DirectoryWatcher] | None:
    """The watcher of directories this system has; None where it has none."""
    system = SYSTEMS.get(sys.platform)
    return None if system is None else system.watcher


def sees_every_change(directory: Path) -> bool:
    """Whether this system's watcher sees every change under a real directory; none does where
    it has none."""
    system = SYSTEMS.get(sys.platform)
    return system is not None and system.sees_every_change(directory)
