"""macOS's FSEvents, read directly, so a search's stale check reads what changed in a project's
files instead of scanning every one (ADR 010, O1).

A tree is one stream of its directory's events, file by file, delivered on a queue of its own
as they come. A look first flushes the stream: `FSEventStreamFlushSync` returns once every event
that happened before it has been delivered, so a look holds every change made before it, with
no marker and no deadline. The kernel tells changes alone, never a file opened or read; a
stream that dropped events, or whose directory moved, says so. The calls are reached through
`ctypes` from CoreServices and CoreFoundation: nothing to install.
"""

import ctypes
import errno
import os
import threading
import weakref
from enum import IntFlag
from typing import override

from semantic_code_mcp.watching.trees import Tree, TreeAction, TreeChange, TreeWatcher

CORE_SERVICES = "/System/Library/Frameworks/CoreServices.framework/CoreServices"
CORE_FOUNDATION = "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
# The C library the interpreter runs on, which holds the dispatch queues' calls: `dlopen(NULL)`
RUNNING_PROGRAM = None
# The callbacks that retain and release the paths an array of CoreFoundation holds
ARRAY_CALLBACKS = "kCFTypeArrayCallBacks"
NO_ALLOCATOR = None
NO_CONTEXT = None
# A queue's events start now, never from the history FSEvents keeps on disk
SINCE_NOW = 0xFFFF_FFFF_FFFF_FFFF  # kFSEventStreamEventIdSinceNow
# How long FSEvents gathers events before it delivers them: git's own watcher of macOS saw events
# dropped at 0.1 s and none at 0.001
LATENCY_SECONDS = 0.001
# The queue the events are delivered on, by its name; one event at a time, as a serial queue runs
QUEUE_NAME = b"semantic-code-mcp.fsevents"
SERIAL_QUEUE = None
REFUSED_MESSAGE = "FSEvents refused to watch the directory"


class StreamFlag(IntFlag):
    """How a stream is made, as <CoreServices/FSEvents.h> names it."""

    # The first event after a quiet spell is delivered at once
    no_defer = 0x0000_0002  # kFSEventStreamCreateFlagNoDefer
    # The directory watched moving, or a directory above it, is told
    watch_root = 0x0000_0004  # kFSEventStreamCreateFlagWatchRoot
    # An event for each file, not only for the directory it lies in
    file_events = 0x0000_0010  # kFSEventStreamCreateFlagFileEvents


STREAM_FLAGS = StreamFlag.no_defer | StreamFlag.watch_root | StreamFlag.file_events


class EventFlag(IntFlag):
    """What an event's flags say, as <CoreServices/FSEvents.h> names them."""

    must_scan_below = 0x0000_0001  # kFSEventStreamEventFlagMustScanSubDirs
    dropped_by_the_service = 0x0000_0002  # kFSEventStreamEventFlagUserDropped
    dropped_by_the_kernel = 0x0000_0004  # kFSEventStreamEventFlagKernelDropped
    identifiers_wrapped = 0x0000_0008  # kFSEventStreamEventFlagEventIdsWrapped
    root_changed = 0x0000_0020  # kFSEventStreamEventFlagRootChanged
    mounted = 0x0000_0040  # kFSEventStreamEventFlagMount
    unmounted = 0x0000_0080  # kFSEventStreamEventFlagUnmount
    created = 0x0000_0100  # kFSEventStreamEventFlagItemCreated
    removed = 0x0000_0200  # kFSEventStreamEventFlagItemRemoved
    inode_changed = 0x0000_0400  # kFSEventStreamEventFlagItemInodeMetaMod
    renamed = 0x0000_0800  # kFSEventStreamEventFlagItemRenamed
    owner_changed = 0x0000_4000  # kFSEventStreamEventFlagItemChangeOwner
    cloned = 0x0040_0000  # kFSEventStreamEventFlagItemCloned


# Events lost, or what no event of a path tells: the directory watched moved, a file system
# mounted or unmounted below it
CHANGES_LOST = (
    EventFlag.must_scan_below
    | EventFlag.dropped_by_the_service
    | EventFlag.dropped_by_the_kernel
    | EventFlag.identifiers_wrapped
    | EventFlag.root_changed
    | EventFlag.mounted
    | EventFlag.unmounted
)
MADE_OR_REMOVED = EventFlag.created | EventFlag.removed | EventFlag.renamed | EventFlag.cloned
ATTRIBUTES_CHANGED = EventFlag.inode_changed | EventFlag.owner_changed

# `FSEventStreamCallback`: the stream, its context, the count of events, their paths in UTF-8,
# their flags and their identifiers
EVENTS_DELIVERED = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.POINTER(ctypes.c_uint64),
)
# `dispatch_function_t`: work a queue runs, handed its context
QUEUE_WORK = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


def action_of(flags: int) -> TreeAction:
    """What an event's flags, folded together for one path, say happened there."""
    if flags & CHANGES_LOST:
        return TreeAction.lost
    if flags & MADE_OR_REMOVED:
        return TreeAction.made_or_removed
    if flags & ATTRIBUTES_CHANGED:
        return TreeAction.attributes_changed
    return TreeAction.written


def nothing_to_run(_context: int | None) -> None:
    """Work that does nothing: run on a queue, it returns once all queued before it has run."""


# A queue's work of nothing, made once: a callback ctypes frees with its object
NOTHING_TO_RUN = QUEUE_WORK(nothing_to_run)


class Frameworks:
    """The calls of CoreServices, CoreFoundation and the dispatch queues a tree makes, each with
    its signature."""

    def __init__(self) -> None:
        services = ctypes.CDLL(CORE_SERVICES)
        foundation = ctypes.CDLL(CORE_FOUNDATION)
        system = ctypes.CDLL(RUNNING_PROGRAM)
        self.string_of_path = foundation.CFStringCreateWithFileSystemRepresentation
        self.string_of_path.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.string_of_path.restype = ctypes.c_void_p
        self.array_of = foundation.CFArrayCreate
        self.array_of.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_long,
            ctypes.c_void_p,
        ]
        self.array_of.restype = ctypes.c_void_p
        self.array_callbacks = ctypes.addressof(ctypes.c_void_p.in_dll(foundation, ARRAY_CALLBACKS))
        self.release = foundation.CFRelease
        self.release.argtypes = [ctypes.c_void_p]
        self.release.restype = None
        self.create_stream = services.FSEventStreamCreate
        self.create_stream.argtypes = [
            ctypes.c_void_p,
            EVENTS_DELIVERED,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_double,
            ctypes.c_uint32,
        ]
        self.create_stream.restype = ctypes.c_void_p
        self.deliver_on = services.FSEventStreamSetDispatchQueue
        self.start = services.FSEventStreamStart
        self.start.restype = ctypes.c_bool
        self.flush = services.FSEventStreamFlushSync
        self.stop = services.FSEventStreamStop
        self.invalidate = services.FSEventStreamInvalidate
        self.release_stream = services.FSEventStreamRelease
        for call in (self.start, self.flush, self.stop, self.invalidate, self.release_stream):
            call.argtypes = [ctypes.c_void_p]
        for call in (self.flush, self.stop, self.invalidate, self.release_stream):
            call.restype = None
        self.deliver_on.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.deliver_on.restype = None
        self.create_queue = system.dispatch_queue_create
        self.create_queue.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
        self.create_queue.restype = ctypes.c_void_p
        self.run_on = system.dispatch_sync_f
        self.run_on.argtypes = [ctypes.c_void_p, ctypes.c_void_p, QUEUE_WORK]
        self.run_on.restype = None
        self.release_queue = system.dispatch_release
        self.release_queue.argtypes = [ctypes.c_void_p]
        self.release_queue.restype = None


class Delivered:
    """The changes a stream delivered on its queue, kept until a look takes them: the queue's
    thread adds them, a look's takes them."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._changes: list[TreeChange] = []

    def deliver(
        self,
        _stream: int | None,
        _context: int | None,
        count: int,
        paths: ctypes._Pointer[ctypes.c_char_p],
        flags: ctypes._Pointer[ctypes.c_uint32],
        _identifiers: ctypes._Pointer[ctypes.c_uint64],
    ) -> None:
        """What the stream calls back with, on its queue's thread."""
        delivered = [
            TreeChange(path=os.fsdecode(paths[event]), action=action_of(flags[event]))
            for event in range(count)
        ]
        with self._lock:
            self._changes += delivered

    def taken(self) -> list[TreeChange]:
        with self._lock:
            changes = self._changes
            self._changes = []
        return changes


class Stream:
    """A stream of FSEvents, the queue it delivers on, and the callback it delivers to, which
    lives as long as the stream may call it."""

    def __init__(self, pointer: int, queue: int, callback: ctypes._CFunctionType) -> None:
        self.pointer = pointer
        self.queue = queue
        self.callback = callback


def released(frameworks: Frameworks, stream: Stream) -> None:
    """Stop a stream and let go of it and of its queue, once nothing it delivered still runs
    there: work of nothing run on the queue returns after all queued before it."""
    frameworks.stop(stream.pointer)
    frameworks.invalidate(stream.pointer)
    frameworks.release_stream(stream.pointer)
    frameworks.run_on(stream.queue, None, NOTHING_TO_RUN)
    frameworks.release_queue(stream.queue)


class FSEventsTree(Tree):
    """A directory and every one below it, watched by a stream of FSEvents."""

    def __init__(self, directory: str, frameworks: Frameworks) -> None:
        super().__init__(directory)
        self._frameworks = frameworks
        self._delivered = Delivered()
        callback = EVENTS_DELIVERED(self._delivered.deliver)
        path = frameworks.string_of_path(NO_ALLOCATOR, os.fsencode(directory))
        paths = (ctypes.c_void_p * 1)(path)
        array = frameworks.array_of(NO_ALLOCATOR, paths, len(paths), frameworks.array_callbacks)
        frameworks.release(path)
        pointer = frameworks.create_stream(
            NO_ALLOCATOR, callback, NO_CONTEXT, array, SINCE_NOW, LATENCY_SECONDS, STREAM_FLAGS
        )
        frameworks.release(array)
        if not pointer:
            raise OSError(errno.EIO, REFUSED_MESSAGE, directory)
        queue = frameworks.create_queue(QUEUE_NAME, SERIAL_QUEUE)
        frameworks.deliver_on(pointer, queue)
        self._stream = Stream(pointer, queue, callback)
        self._release = weakref.finalize(self, released, frameworks, self._stream)
        if not frameworks.start(pointer):
            self._release()
            raise OSError(errno.EIO, REFUSED_MESSAGE, directory)

    @override
    def changes(self) -> list[TreeChange]:
        self._frameworks.flush(self._stream.pointer)
        return self._delivered.taken()

    @override
    def close(self) -> None:
        self._release()


class FSEvents(TreeWatcher):
    """Directories watched within trees FSEvents tells the changes of."""

    def __init__(self) -> None:
        super().__init__()
        self._frameworks = Frameworks()

    @override
    def open_tree(self, directory: str) -> Tree:
        return FSEventsTree(directory, self._frameworks)
