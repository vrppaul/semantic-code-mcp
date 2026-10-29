"""Which file systems a system's watcher sees every change on (ADR 010, O1).

A watcher sees the changes this machine's kernel makes: on a network or FUSE file system, or
WSL's mount of Windows' drives, another machine's or Windows' own go unseen, so a project is
followed only on a file system of a local disk or of memory. Linux reads its mounts' table,
which also names each file system mounted below the project; macOS and Windows ask the system
the type of the one the project lies on, and a file system mounted below it is the check
against a scan's to find.
"""

import ctypes
import ctypes.wintypes
import os
import platform
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict

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
# Linux's file systems whose every change this kernel makes, so inotify sees it: those on a
# local disk, in memory, or an overlay of them
LINUX_LOCAL_FILE_SYSTEMS = frozenset(
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

# macOS's file systems of a local disk, whose changes FSEvents hears from the kernel
MACOS_LOCAL_FILE_SYSTEMS = frozenset({"apfs", "hfs"})
# The C library the interpreter runs on, which holds `statfs`: `dlopen(NULL)`
RUNNING_PROGRAM = None
# The `statfs` that fills `struct statfs` as below, by the processor: Intel's variant of 64-bit
# inodes, the only one arm64 has
STATFS_SYMBOLS = {"x86_64": "statfs$INODE64"}
DEFAULT_STATFS_SYMBOL = "statfs"
# The lengths of a file system's type name and of a path in `struct statfs`
FILE_SYSTEM_TYPE_LENGTH = 16
PATH_LENGTH = 1024
RESERVED_WORDS = 7

# Windows' file systems of a local disk, which tell ReadDirectoryChangesW each change, and the
# kinds of drive they lie on, as GetDriveTypeW names them
WINDOWS_LOCAL_FILE_SYSTEMS = frozenset({"NTFS", "ReFS", "FAT", "FAT32", "exFAT"})
REMOVABLE_DRIVE = 2  # DRIVE_REMOVABLE
FIXED_DRIVE = 3  # DRIVE_FIXED
DRIVE_IN_MEMORY = 6  # DRIVE_RAMDISK
WINDOWS_LOCAL_DRIVES = frozenset({REMOVABLE_DRIVE, FIXED_DRIVE, DRIVE_IN_MEMORY})
# The library of Windows' calls of files and volumes
WINDOWS_KERNEL = "kernel32"
# The longest path Windows takes, in UTF-16 units; and a file system's name, MAX_PATH + 1
LONGEST_WINDOWS_PATH = 32_768
FILE_SYSTEM_NAME_LENGTH = 261


def unescaped(mount_text: str) -> str:
    """A path of the mounts' table as the file system spells it."""
    return MOUNT_ESCAPE.sub(lambda escape: chr(int(escape.group(1), OCTAL)), mount_text)


def lies_within(path: str, mount_point: str) -> bool:
    """Whether a path lies at a mount point or below it."""
    return path == mount_point or path.startswith(
        mount_point.rstrip(PATH_SEPARATOR) + PATH_SEPARATOR
    )


class Mount(BaseModel):
    """A file system mounted: where, and its type."""

    model_config = ConfigDict(frozen=True)

    point: str
    file_system: str


def mounts_of(mount_info: str) -> list[Mount]:
    """The mounts of /proc/self/mountinfo's table, in its order."""
    mounts: list[Mount] = []
    for line in mount_info.splitlines():
        fields = line.split(FIELD_SEPARATOR)
        mounts.append(
            Mount(
                point=unescaped(fields[MOUNT_POINT_FIELD]),
                file_system=fields[fields.index(OPTIONAL_FIELDS_END) + 1],
            )
        )
    return mounts


def file_system_types(path: str, mount_info: str) -> list[str]:
    """The types of the file systems at and below a real, absolute path, by the mounts' table:
    the one it lies on, the deepest mount point holding it and the last one mounted there where
    several are, then each mounted below it, in the table's order."""
    mounts = mounts_of(mount_info)
    holding = [mount for mount in mounts if lies_within(path, mount.point)]
    # The last of the deepest, as `max` keeps the first of equals it meets
    deepest = max(reversed(holding), key=lambda mount: len(mount.point), default=None)
    below = [
        mount.file_system
        for mount in mounts
        if mount.point != path and lies_within(mount.point, path)
    ]
    return ([deepest.file_system] if deepest is not None else []) + below


def linux_sees_every_change(directory: Path) -> bool:
    """Whether inotify sees every change under a real directory: it lies on a local file
    system, and so does each file system mounted below it."""
    try:
        mount_info = os.fsdecode(MOUNT_INFO.read_bytes())
    except OSError:
        return False
    types = file_system_types(str(directory), mount_info)
    return bool(types) and all(file_system in LINUX_LOCAL_FILE_SYSTEMS for file_system in types)


class FileSystemStatus(ctypes.Structure):
    """macOS's `struct statfs` of 64-bit inodes, <sys/mount.h>: of it only the type is read."""

    _fields_ = (
        ("block_size", ctypes.c_uint32),
        ("transfer_size", ctypes.c_int32),
        ("blocks", ctypes.c_uint64),
        ("free_blocks", ctypes.c_uint64),
        ("available_blocks", ctypes.c_uint64),
        ("files", ctypes.c_uint64),
        ("free_files", ctypes.c_uint64),
        ("identifier", ctypes.c_int32 * 2),
        ("owner", ctypes.c_uint32),
        ("type_number", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("subtype", ctypes.c_uint32),
        ("type_name", ctypes.c_char * FILE_SYSTEM_TYPE_LENGTH),
        ("mounted_on", ctypes.c_char * PATH_LENGTH),
        ("mounted_from", ctypes.c_char * PATH_LENGTH),
        ("more_flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * RESERVED_WORDS),
    )


def macos_sees_every_change(directory: Path) -> bool:
    """Whether FSEvents sees every change under a directory: it lies on a local disk's file
    system, as `statfs` names its type."""
    library = ctypes.CDLL(RUNNING_PROGRAM, use_errno=True)
    status_of = library[STATFS_SYMBOLS.get(platform.machine(), DEFAULT_STATFS_SYMBOL)]
    status_of.argtypes = [ctypes.c_char_p, ctypes.POINTER(FileSystemStatus)]
    status_of.restype = ctypes.c_int
    status = FileSystemStatus()
    if status_of(os.fsencode(directory), ctypes.byref(status)) != 0:
        return False
    return os.fsdecode(status.type_name) in MACOS_LOCAL_FILE_SYSTEMS


def windows_sees_every_change(directory: Path) -> bool:
    """Whether ReadDirectoryChangesW sees every change under a directory: the volume it lies on
    is a local drive's, and its file system one that tells each change."""
    kernel = ctypes.WinDLL(WINDOWS_KERNEL, use_last_error=True)
    volume_path_of = kernel.GetVolumePathNameW
    volume_path_of.argtypes = [
        ctypes.wintypes.LPCWSTR,
        ctypes.wintypes.LPWSTR,
        ctypes.wintypes.DWORD,
    ]
    volume_path_of.restype = ctypes.wintypes.BOOL
    drive_type_of = kernel.GetDriveTypeW
    drive_type_of.argtypes = [ctypes.wintypes.LPCWSTR]
    drive_type_of.restype = ctypes.wintypes.UINT
    volume_information_of = kernel.GetVolumeInformationW
    volume_information_of.argtypes = [
        ctypes.wintypes.LPCWSTR,
        ctypes.wintypes.LPWSTR,
        ctypes.wintypes.DWORD,
        ctypes.wintypes.LPDWORD,
        ctypes.wintypes.LPDWORD,
        ctypes.wintypes.LPDWORD,
        ctypes.wintypes.LPWSTR,
        ctypes.wintypes.DWORD,
    ]
    volume_information_of.restype = ctypes.wintypes.BOOL
    volume = ctypes.create_unicode_buffer(LONGEST_WINDOWS_PATH)
    if not volume_path_of(str(directory), volume, LONGEST_WINDOWS_PATH):
        return False
    if drive_type_of(volume.value) not in WINDOWS_LOCAL_DRIVES:
        return False
    file_system = ctypes.create_unicode_buffer(FILE_SYSTEM_NAME_LENGTH)
    # Neither the volume's name nor its serial number, its names' length or its flags is asked
    found = volume_information_of(
        volume.value, None, 0, None, None, None, file_system, FILE_SYSTEM_NAME_LENGTH
    )
    return bool(found) and file_system.value in WINDOWS_LOCAL_FILE_SYSTEMS
