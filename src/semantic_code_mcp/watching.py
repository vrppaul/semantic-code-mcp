"""Linux's inotify, read directly, so a search's stale check reads what changed in a project's
files instead of scanning every one (ADR 010, O1).

The kernel queues an event inside the system call that changed a file, so a read of the queue
at a check holds every change made before it: no thread, no marker, no deadline. The watches
ask for changes alone, never a file opened or read, so a tool reading every file queues
nothing; a queue that overflowed says so. inotify's calls are reached through `ctypes` from the
C library the interpreter runs on: nothing to install.

inotify sees the changes this machine's kernel makes: on a network or FUSE file system, or
WSL's mount of Windows' drives, another machine's or Windows' own go unseen, so only the file
systems named in LOCAL_FILE_SYSTEMS are watched.
"""

import ctypes
import os
import re
import struct
import sys
import weakref
from enum import IntFlag, StrEnum, auto
from pathlib import Path

from pydantic import BaseModel, ConfigDict

LINUX_PLATFORM = "linux"
# The C library the interpreter runs on, which holds inotify's calls: `dlopen(NULL)`
RUNNING_PROGRAM = None
# The header of each event inotify hands a read, `struct inotify_event`: the watch, the mask,
# the cookie pairing a move's two halves, and the length of the name after it, padded with NUL
EVENT_HEADER = struct.Struct("iIII")
NAME_PADDING = b"\0"
# What one read takes from the queue: many events, and more than the largest one, a header and a
# name of NAME_MAX bytes
READ_SIZE = 65_536
# The text of the paths in the mounts' table escapes a blank, a tab, a newline and a backslash
# as a backslash and three octal digits
MOUNT_ESCAPE = re.compile(r"\\([0-7]{3})")
OCTAL = 8
# A line of /proc/self/mountinfo: its mount point is the fifth field, and its file system's type
# follows the field that ends the optional ones
MOUNT_POINT_FIELD = 4
OPTIONAL_FIELDS_END = "-"
MOUNT_INFO = Path("/proc/self/mountinfo")
FIELD_SEPARATOR = " "
PATH_SEPARATOR = "/"
# File systems whose every change this kernel makes, so inotify sees it: those on a local disk,
# in memory, or an overlay of them
LOCAL_FILE_SYSTEMS = frozenset(
    {
        "ext2",
        "ext3",
        "ext4",
        "xfs",
        "btrfs",
        "f2fs",
        "zfs",
        "bcachefs",
        "jfs",
        "reiserfs",
        "nilfs2",
        "tmpfs",
        "ramfs",
        "overlay",
    }
)


class InotifyFlag(IntFlag):
    """The bits of an inotify mask this module asks for or reads, as <sys/inotify.h> gives them."""

    modified = 0x0000_0002  # IN_MODIFY
    attributes_changed = 0x0000_0004  # IN_ATTRIB
    closed_after_writing = 0x0000_0008  # IN_CLOSE_WRITE
    moved_from = 0x0000_0040  # IN_MOVED_FROM
    moved_to = 0x0000_0080  # IN_MOVED_TO
    created = 0x0000_0100  # IN_CREATE
    deleted = 0x0000_0200  # IN_DELETE
    deleted_itself = 0x0000_0400  # IN_DELETE_SELF
    moved_itself = 0x0000_0800  # IN_MOVE_SELF
    unmounted = 0x0000_2000  # IN_UNMOUNT
    queue_overflowed = 0x0000_4000  # IN_Q_OVERFLOW
    watch_removed = 0x0000_8000  # IN_IGNORED
    only_directory = 0x0100_0000  # IN_ONLYDIR
    not_following_links = 0x0200_0000  # IN_DONT_FOLLOW
    is_directory = 0x4000_0000  # IN_ISDIR


# What a watch asks for: every change of an entry or of the directory itself, never a file
# opened or read; and the directory itself, not a symlink to one
WATCHED_MASK = (
    InotifyFlag.modified
    | InotifyFlag.attributes_changed
    | InotifyFlag.closed_after_writing
    | InotifyFlag.moved_from
    | InotifyFlag.moved_to
    | InotifyFlag.created
    | InotifyFlag.deleted
    | InotifyFlag.deleted_itself
    | InotifyFlag.moved_itself
    | InotifyFlag.only_directory
    | InotifyFlag.not_following_links
)
ENTRY_ADDED = InotifyFlag.created | InotifyFlag.moved_to
ENTRY_REMOVED = InotifyFlag.deleted | InotifyFlag.moved_from
DIRECTORY_GONE = (
    InotifyFlag.deleted_itself
    | InotifyFlag.moved_itself
    | InotifyFlag.unmounted
    | InotifyFlag.watch_removed
)


class Change(StrEnum):
    """What an event says happened in a watched directory."""

    # An entry made, or moved in
    entry_added = auto()
    # An entry deleted, or moved out
    entry_removed = auto()
    # An entry written, closed after writing, or its attributes changed; or the directory's own
    entry_changed = auto()
    # The watched directory itself deleted, moved, unmounted, or its watch removed
    directory_gone = auto()
    # Events were lost: the queue was full
    queue_overflowed = auto()


class WatchEvent(BaseModel):
    """One event of an inotify instance's queue."""

    model_config = ConfigDict(frozen=True)

    # The descriptor of the watch it came from; -1 for a queue that overflowed, which names none
    watch: int
    change: Change
    # The entry's name in the watched directory, as the file system spells it; empty for an
    # event of the directory itself
    name: str
    is_directory: bool


def watching_supported() -> bool:
    """Whether inotify is here to watch with: on Linux alone."""
    return sys.platform == LINUX_PLATFORM


def change_of(mask: int) -> Change:
    """What an event's mask says happened."""
    if mask & InotifyFlag.queue_overflowed:
        return Change.queue_overflowed
    if mask & ENTRY_ADDED:
        return Change.entry_added
    if mask & ENTRY_REMOVED:
        return Change.entry_removed
    if mask & DIRECTORY_GONE:
        return Change.directory_gone
    return Change.entry_changed


def parsed_events(buffer: bytes) -> list[WatchEvent]:
    """The events one read of an inotify descriptor handed back, in their order."""
    events: list[WatchEvent] = []
    offset = 0
    while offset < len(buffer):
        watch, mask, _cookie, name_length = EVENT_HEADER.unpack_from(buffer, offset)
        name_start = offset + EVENT_HEADER.size
        name = buffer[name_start : name_start + name_length].rstrip(NAME_PADDING)
        events.append(
            WatchEvent(
                watch=watch,
                change=change_of(mask),
                name=os.fsdecode(name),
                is_directory=bool(mask & InotifyFlag.is_directory),
            )
        )
        offset = name_start + name_length
    return events


def last_error(path: str) -> OSError:
    """The error a call of the C library just set, as Python raises it for the same number:
    FileNotFoundError, PermissionError and their kin."""
    error_number = ctypes.get_errno()
    return OSError(error_number, os.strerror(error_number), path)


class Inotify:
    """One inotify instance: watches directories, each on its own, and reads what changed in them
    since the last read. Not safe for two threads at once; its descriptor is closed with it, or
    when it is collected."""

    def __init__(self) -> None:
        """Raises the OSError the system gives where no instance can be made, as at the user's
        limit of instances."""
        library = ctypes.CDLL(RUNNING_PROGRAM, use_errno=True)
        self._add_watch = library.inotify_add_watch
        self._add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        self._add_watch.restype = ctypes.c_int
        self._remove_watch = library.inotify_rm_watch
        self._remove_watch.argtypes = [ctypes.c_int, ctypes.c_int]
        self._remove_watch.restype = ctypes.c_int
        # A read that finds nothing returns at once; a process started meanwhile does not inherit it
        descriptor = library.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if descriptor < 0:
            raise last_error(path="")
        self._descriptor = descriptor
        self._close_descriptor = weakref.finalize(self, os.close, descriptor)

    def watch(self, directory: str) -> int:
        """Watch a directory, never a symlink to one, for changes of its entries and of itself;
        the watch's descriptor, the same for a directory watched already. Raises the OSError the
        system gives: FileNotFoundError or NotADirectoryError where it is gone or no directory,
        PermissionError where it may not be read, an OSError of ENOSPC at the user's limit of
        watches."""
        watch = self._add_watch(self._descriptor, os.fsencode(directory), WATCHED_MASK)
        if watch < 0:
            raise last_error(directory)
        return watch

    def unwatch(self, watch: int) -> None:
        """Stop a watch; one the kernel removed already, as a deleted directory's, is left as
        it is."""
        self._remove_watch(self._descriptor, watch)

    def events(self) -> list[WatchEvent]:
        """Every event queued since the last read, in their order."""
        events: list[WatchEvent] = []
        while True:
            try:
                buffer = os.read(self._descriptor, READ_SIZE)
            except BlockingIOError:
                return events
            events += parsed_events(buffer)

    def close(self) -> None:
        """Close the descriptor, which removes every watch."""
        self._close_descriptor()


def unescaped(mount_text: str) -> str:
    """A path of the mounts' table as the file system spells it."""
    return MOUNT_ESCAPE.sub(lambda escape: chr(int(escape.group(1), OCTAL)), mount_text)


def lies_within(path: str, mount_point: str) -> bool:
    """Whether a path lies at a mount point or below it."""
    return path == mount_point or path.startswith(
        mount_point.rstrip(PATH_SEPARATOR) + PATH_SEPARATOR
    )


def file_system_type(path: str, mount_info: str) -> str | None:
    """The type of the file system a real, absolute path lies on, by the mounts' table of
    /proc/self/mountinfo: the deepest mount point holding it, the last one mounted there where
    several are; None where none holds it."""
    deepest_point: str | None = None
    found_type: str | None = None
    for line in mount_info.splitlines():
        fields = line.split(FIELD_SEPARATOR)
        mount_point = unescaped(fields[MOUNT_POINT_FIELD])
        if lies_within(path, mount_point) and (
            deepest_point is None or len(mount_point) >= len(deepest_point)
        ):
            deepest_point = mount_point
            found_type = fields[fields.index(OPTIONAL_FIELDS_END) + 1]
    return found_type


def sees_every_change(directory: Path) -> bool:
    """Whether inotify sees every change under a real directory: it lies on a local file
    system."""
    try:
        mount_info = os.fsdecode(MOUNT_INFO.read_bytes())
    except OSError:
        return False
    return file_system_type(str(directory), mount_info) in LOCAL_FILE_SYSTEMS
