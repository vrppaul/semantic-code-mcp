"""Windows' ReadDirectoryChangesW, read directly, so a search's stale check reads what changed in a
project's files instead of scanning every one (ADR 010, O1).

A tree is a handle of its directory opened for overlapped reads, one read of its changes always
asked. The file system tells a change inside the call that made it: it completes the read
asked, or keeps the change for the next. So a look that takes each read completed and asks
again, until one is left waiting, holds every change made before it: no marker, no deadline.
The read asks for names, sizes and times written, never a file opened or read; a buffer the
changes overflowed says so. The calls are reached through `ctypes` from kernel32: nothing to
install.

The system tells the changes of the whole tree, the directories a scan never reads among them,
and keeps between two reads only what one buffer holds: a build's burst under `target/`
overflowed it between two looks, and the look after started the following again, 2.2 scans on
CI's runner. So a thread of the tree's own takes each read as it ends and keeps its changes for
the next look, which takes what the thread kept and what has ended since, under one lock; past
MOST_CHANGES_KEPT the tree keeps that changes were lost instead, as an install's flood is no
cheaper to follow than to scan again.

A read asked by a thread that ends is cancelled: its changes since are lost, which ends the
following as an overflow does.
"""

import ctypes
import ctypes.wintypes
import os
import struct
import threading
import weakref
from enum import IntEnum
from typing import override

from semantic_code_mcp.watching.trees import CHANGES_LOST, Tree, TreeAction, TreeChange, TreeWatcher

WINDOWS_KERNEL = "kernel32"
# A handle that lists a directory, whose files others may read, write, delete and rename while it
# is watched; a directory opens only for backup's semantics, here for overlapped reads
LIST_DIRECTORY = 0x0001  # FILE_LIST_DIRECTORY
SHARE_EVERYTHING = 0x0001 | 0x0002 | 0x0004  # FILE_SHARE_READ, FILE_SHARE_WRITE, FILE_SHARE_DELETE
OPEN_EXISTING = 3
BACKUP_SEMANTICS = 0x0200_0000  # FILE_FLAG_BACKUP_SEMANTICS
OVERLAPPED_READS = 0x4000_0000  # FILE_FLAG_OVERLAPPED
INVALID_HANDLE = ctypes.c_void_p(-1).value
# What a read tells of the directory and every one below it: names made, removed or renamed,
# of files and of directories; sizes; times written
WATCH_SUBTREE = True
FILE_NAMES = 0x0001  # FILE_NOTIFY_CHANGE_FILE_NAME
DIRECTORY_NAMES = 0x0002  # FILE_NOTIFY_CHANGE_DIR_NAME
SIZES = 0x0008  # FILE_NOTIFY_CHANGE_SIZE
TIMES_WRITTEN = 0x0010  # FILE_NOTIFY_CHANGE_LAST_WRITE
TOLD = FILE_NAMES | DIRECTORY_NAMES | SIZES | TIMES_WRITTEN
# What one read takes, and what the system keeps between two reads: about 20,000 records of short
# names, so a test run or a commit between two looks rarely overflows it
BUFFER_SIZE = 1 << 20
# The event a read's end is told by: reset by hand, as each read asked resets it, and unset;
# so is the event that stops a tree's thread
MANUAL_RESET = True
NOT_SIGNALLED = False
# What a tree's thread waits for: either of its two events, as long as it takes, and which of
# them came, its read's first
WAIT_FOR_EITHER = False
NO_END = 0xFFFF_FFFF  # INFINITE
READ_ENDED = 0  # WAIT_OBJECT_0
# The changes a tree keeps between two looks before it keeps that they were lost instead:
# about 17,000 files made, each told about three times; 100,000 changes of paths of 71
# characters held 62 MB, measured
MOST_CHANGES_KEPT = 50_000
# A look takes what is done and never waits; a handle closed waits for its read cancelled
DO_NOT_WAIT = False
WAIT = True
READ_NOT_DONE = 996  # ERROR_IO_INCOMPLETE
# `FILE_NOTIFY_INFORMATION`: the offset of the next record, 0 for the last, the action, and the
# length in bytes of the name after it, in UTF-16, from the directory watched
RECORD_HEADER = struct.Struct("<III")
LAST_RECORD = 0
NAME_ENCODING = "utf-16-le"
# A name Windows holds may pair no surrogates, which Python's own names keep as they are
NAME_ERRORS = "surrogatepass"


class FileAction(IntEnum):
    """What a record says happened, as <winnt.h> names it."""

    added = 1  # FILE_ACTION_ADDED
    removed = 2  # FILE_ACTION_REMOVED
    modified = 3  # FILE_ACTION_MODIFIED
    renamed_from = 4  # FILE_ACTION_RENAMED_OLD_NAME
    renamed_to = 5  # FILE_ACTION_RENAMED_NEW_NAME


TREE_ACTIONS = {
    FileAction.added: TreeAction.made_or_removed,
    FileAction.removed: TreeAction.made_or_removed,
    FileAction.modified: TreeAction.written,
    FileAction.renamed_from: TreeAction.made_or_removed,
    FileAction.renamed_to: TreeAction.made_or_removed,
}


class Overlapped(ctypes.Structure):
    """`OVERLAPPED`, <minwinbase.h>: a read's status, kept by the system, and its event."""

    _fields_ = (
        ("status", ctypes.c_size_t),
        ("transferred", ctypes.c_size_t),
        ("offset", ctypes.wintypes.DWORD),
        ("offset_high", ctypes.wintypes.DWORD),
        ("event", ctypes.wintypes.HANDLE),
    )


class Kernel:
    """The calls of kernel32 a tree makes, each with its signature."""

    def __init__(self) -> None:
        library = ctypes.WinDLL(WINDOWS_KERNEL, use_last_error=True)
        self.create_file = library.CreateFileW
        self.create_file.argtypes = [
            ctypes.wintypes.LPCWSTR,
            ctypes.wintypes.DWORD,
            ctypes.wintypes.DWORD,
            ctypes.wintypes.LPVOID,
            ctypes.wintypes.DWORD,
            ctypes.wintypes.DWORD,
            ctypes.wintypes.HANDLE,
        ]
        self.create_file.restype = ctypes.wintypes.HANDLE
        self.create_event = library.CreateEventW
        self.create_event.argtypes = [
            ctypes.wintypes.LPVOID,
            ctypes.wintypes.BOOL,
            ctypes.wintypes.BOOL,
            ctypes.wintypes.LPCWSTR,
        ]
        self.create_event.restype = ctypes.wintypes.HANDLE
        self.read_changes = library.ReadDirectoryChangesW
        self.read_changes.argtypes = [
            ctypes.wintypes.HANDLE,
            ctypes.wintypes.LPVOID,
            ctypes.wintypes.DWORD,
            ctypes.wintypes.BOOL,
            ctypes.wintypes.DWORD,
            ctypes.wintypes.LPDWORD,
            ctypes.POINTER(Overlapped),
            ctypes.wintypes.LPVOID,
        ]
        self.read_changes.restype = ctypes.wintypes.BOOL
        self.read_result = library.GetOverlappedResult
        self.read_result.argtypes = [
            ctypes.wintypes.HANDLE,
            ctypes.POINTER(Overlapped),
            ctypes.wintypes.LPDWORD,
            ctypes.wintypes.BOOL,
        ]
        self.read_result.restype = ctypes.wintypes.BOOL
        self.cancel = library.CancelIoEx
        self.cancel.argtypes = [ctypes.wintypes.HANDLE, ctypes.POINTER(Overlapped)]
        self.cancel.restype = ctypes.wintypes.BOOL
        self.close_handle = library.CloseHandle
        self.close_handle.argtypes = [ctypes.wintypes.HANDLE]
        self.close_handle.restype = ctypes.wintypes.BOOL
        self.set_event = library.SetEvent
        self.reset_event = library.ResetEvent
        for call in (self.set_event, self.reset_event):
            call.argtypes = [ctypes.wintypes.HANDLE]
            call.restype = ctypes.wintypes.BOOL
        self.wait_for_any = library.WaitForMultipleObjects
        self.wait_for_any.argtypes = [
            ctypes.wintypes.DWORD,
            ctypes.POINTER(ctypes.wintypes.HANDLE),
            ctypes.wintypes.BOOL,
            ctypes.wintypes.DWORD,
        ]
        self.wait_for_any.restype = ctypes.wintypes.DWORD


def parsed_changes(records: bytes, directory: str) -> list[TreeChange]:
    """The changes the records of one read tell, in their order."""
    changes: list[TreeChange] = []
    offset = 0
    while True:
        next_offset, action, name_length = RECORD_HEADER.unpack_from(records, offset)
        name_start = offset + RECORD_HEADER.size
        name = records[name_start : name_start + name_length].decode(NAME_ENCODING, NAME_ERRORS)
        changes.append(
            TreeChange(path=f"{directory}{os.sep}{name}", action=TREE_ACTIONS[FileAction(action)])
        )
        if next_offset == LAST_RECORD:
            return changes
        offset += next_offset


class Read:
    """What the system writes into while a read of a directory's changes waits: the directory's
    handle, the read's status and event, its buffer, what it filled, and whether one waits."""

    def __init__(self, handle: int, event: int) -> None:
        self.handle = handle
        self.overlapped = Overlapped(event=event)
        self.buffer = ctypes.create_string_buffer(BUFFER_SIZE)
        self.transferred = ctypes.wintypes.DWORD()
        # Set while the system holds a read asked: one it refused leaves its status pending
        # with no end to come, which a wait for it would wait on for good
        self.pending = False


def within_what_is_kept(changes: list[TreeChange]) -> list[TreeChange]:
    """The changes, or that changes were lost where they are more than a tree keeps."""
    return [CHANGES_LOST] if len(changes) > MOST_CHANGES_KEPT else changes


class Reading:
    """A tree's reads of its directory's changes, each taken as it ends by the tree's thread or
    at a look, under one lock, and the changes the thread keeps for the next look."""

    def __init__(self, kernel: Kernel, read: Read, stop: int, directory: str) -> None:
        self._kernel = kernel
        self.read = read
        # Set as the tree closes, which ends its thread
        self.stop = stop
        self._directory = directory
        self._lock = threading.Lock()
        self._kept: list[TreeChange] = []
        # Set once the system refuses a read, as for a directory deleted: nothing is told since
        self._broken = False

    def ask(self) -> bool:
        """Ask for the next read of the changes; whether the system took it. Its event is
        unset first, so one set tells this read's end."""
        self._kernel.reset_event(self.read.overlapped.event)
        self.read.pending = bool(
            self._kernel.read_changes(
                self.read.handle,
                self.read.buffer,
                BUFFER_SIZE,
                WATCH_SUBTREE,
                TOLD,
                None,
                ctypes.byref(self.read.overlapped),
                None,
            )
        )
        return self.read.pending

    def run(self) -> None:
        """The tree's thread: keep the changes of each read as it ends, until the tree closes or
        the system refuses a read."""
        events = (ctypes.wintypes.HANDLE * 2)(self.read.overlapped.event, self.stop)
        while self._kernel.wait_for_any(len(events), events, WAIT_FOR_EITHER, NO_END) == READ_ENDED:
            with self._lock:
                self._kept = within_what_is_kept([*self._kept, *self._ended()])
                if self._broken:
                    return

    def taken(self) -> list[TreeChange]:
        """What the thread kept and what has ended since, every change made before the call."""
        with self._lock:
            changes = within_what_is_kept([*self._kept, *self._ended()])
            self._kept = []
        return changes

    def _ended(self) -> list[TreeChange]:
        """The changes of each read ended, the next asked after each, until one waits."""
        changes: list[TreeChange] = []
        while not self._broken:
            read = self.read
            done = self._kernel.read_result(
                read.handle,
                ctypes.byref(read.overlapped),
                ctypes.byref(read.transferred),
                DO_NOT_WAIT,
            )
            if not done and ctypes.get_last_error() == READ_NOT_DONE:
                return changes
            read.pending = False
            if done and read.transferred.value:
                records = ctypes.string_at(read.buffer, read.transferred.value)
                changes += parsed_changes(records, self._directory)
            else:
                # Nothing told: more changed than the buffer holds, which the read ends with no
                # record or with ERROR_NOTIFY_ENUM_DIR, or it failed
                changes.append(CHANGES_LOST)
            self._broken = not self.ask()
        return [*changes, CHANGES_LOST]


def released(kernel: Kernel, read: Read) -> None:
    """Cancel a tree's read and close its handles, once the system writes no more into what the
    read holds: a read still pending is cancelled and waited for."""
    if read.pending:
        kernel.cancel(read.handle, ctypes.byref(read.overlapped))
        kernel.read_result(
            read.handle, ctypes.byref(read.overlapped), ctypes.byref(read.transferred), WAIT
        )
    kernel.close_handle(read.overlapped.event)
    kernel.close_handle(read.handle)


def closed(kernel: Kernel, reading: Reading, thread: threading.Thread) -> None:
    """End a tree's thread, then release its read and its handles."""
    kernel.set_event(reading.stop)
    thread.join()
    released(kernel, reading.read)
    kernel.close_handle(reading.stop)


class DirectoryChangesTree(Tree):
    """A directory and every one below it, watched by ReadDirectoryChangesW."""

    def __init__(self, directory: str, kernel: Kernel) -> None:
        super().__init__(directory)
        handle = kernel.create_file(
            directory,
            LIST_DIRECTORY,
            SHARE_EVERYTHING,
            None,
            OPEN_EXISTING,
            BACKUP_SEMANTICS | OVERLAPPED_READS,
            None,
        )
        if handle == INVALID_HANDLE:
            raise ctypes.WinError(ctypes.get_last_error())
        read_ended = kernel.create_event(None, MANUAL_RESET, NOT_SIGNALLED, None)
        stop = kernel.create_event(None, MANUAL_RESET, NOT_SIGNALLED, None)
        if not read_ended or not stop:
            error = ctypes.get_last_error()
            for opened in (read_ended, stop, handle):
                kernel.close_handle(opened)
            raise ctypes.WinError(error)
        read = Read(handle, read_ended)
        self._reading = Reading(kernel, read, stop, directory)
        if not self._reading.ask():
            error = ctypes.get_last_error()
            released(kernel, read)
            kernel.close_handle(stop)
            raise ctypes.WinError(error)
        thread = threading.Thread(target=self._reading.run, daemon=True)
        thread.start()
        self._close = weakref.finalize(self, closed, kernel, self._reading, thread)

    @override
    def changes(self) -> list[TreeChange]:
        return self._reading.taken()

    @override
    def close(self) -> None:
        self._close()


class DirectoryChanges(TreeWatcher):
    """Directories watched within trees ReadDirectoryChangesW tells the changes of."""

    def __init__(self) -> None:
        super().__init__()
        self._kernel = Kernel()

    @override
    def open_tree(self, directory: str) -> Tree:
        return DirectoryChangesTree(directory, self._kernel)
