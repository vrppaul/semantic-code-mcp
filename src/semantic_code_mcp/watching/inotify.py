"""Linux's inotify, read directly, so a search's stale check reads what changed in a project's
files instead of scanning every one (ADR 010, O1).

The kernel queues an event inside the system call that changed a file, so a read of the queue
at a check holds every change made before it: no thread, no marker, no deadline. The watches
ask for changes alone, never a file opened or read, so a tool reading every file queues
nothing; a queue that overflowed says so. inotify's calls are reached through `ctypes` from the
C library the interpreter runs on: nothing to install.
"""

import ctypes
import os
import struct
import weakref
from enum import IntFlag
from typing import override

from semantic_code_mcp.domain.interfaces import DirectoryWatcher
from semantic_code_mcp.domain.models import WatchChange, WatchEvent

# The C library the interpreter runs on, which holds inotify's calls: `dlopen(NULL)`
RUNNING_PROGRAM = None
# The header of each event inotify hands a read, `struct inotify_event`: the watch, the mask,
# the cookie pairing a move's two halves, and the length of the name after it, padded with NUL
EVENT_HEADER = struct.Struct("iIII")
NAME_PADDING = b"\0"
# What one read takes from the queue: many events, and more than the largest one, a header and a
# name of NAME_MAX bytes
READ_SIZE = 65_536


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


def change_of(mask: int) -> WatchChange:
    """What an event's mask says happened."""
    if mask & InotifyFlag.queue_overflowed:
        return WatchChange.queue_overflowed
    if mask & ENTRY_ADDED:
        return WatchChange.entry_added
    if mask & ENTRY_REMOVED:
        return WatchChange.entry_removed
    if mask & DIRECTORY_GONE:
        return WatchChange.directory_gone
    return WatchChange.entry_changed


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


class Inotify(DirectoryWatcher):
    """One inotify instance: watches directories, each on its own, and reads what changed in them
    since the last read. Its descriptor is closed with it, or when it is collected."""

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

    @override
    def watch(self, directory: str) -> int:
        """The watch's descriptor; at the user's limit of watches an OSError of ENOSPC."""
        watch = self._add_watch(self._descriptor, os.fsencode(directory), WATCHED_MASK)
        if watch < 0:
            raise last_error(directory)
        return watch

    @override
    def unwatch(self, watch: int) -> None:
        self._remove_watch(self._descriptor, watch)

    @override
    def events(self) -> list[WatchEvent]:
        events: list[WatchEvent] = []
        while True:
            try:
                buffer = os.read(self._descriptor, READ_SIZE)
            except BlockingIOError:
                return events
            events += parsed_events(buffer)

    @override
    def close(self) -> None:
        """Close the descriptor, which removes every watch."""
        self._close_descriptor()
