"""Tests of which file system a path lies on, read from the mounts' table: inotify sees every
change only on a local one."""

import pytest

from semantic_code_mcp.watching import file_system_type

# A table as /proc/self/mountinfo gives it: the root on a disk, network and FUSE shares below
# the home directory, one spelled with a blank, WSL's mount of Windows' drive, and two file
# systems mounted at one point, the overlay last
MOUNT_INFO = """\
22 1 8:2 / / rw,relatime shared:1 - ext4 /dev/sda2 rw,errors=remount-ro
40 22 0:35 / /home/user/shares rw,relatime shared:20 master:3 - nfs4 server:/export rw
41 40 0:36 / /home/user/shares/my\\040code rw,relatime shared:21 - fuse.sshfs host: rw
50 22 0:40 / /mnt/c rw,noatime - 9p drvfs rw
51 22 0:41 / /tmp rw shared:30 - tmpfs tmpfs rw
52 51 0:42 / /tmp rw shared:31 - overlay overlay rw
"""


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        pytest.param("/home/user/project", "ext4", id="on the root's disk"),
        pytest.param("/home/user/shares/repository", "nfs4", id="on a network share"),
        pytest.param(
            "/home/user/shares/my code/repository", "fuse.sshfs", id="spelled with a blank"
        ),
        pytest.param("/mnt/c/Users/user/project", "9p", id="on Windows' drive under WSL"),
        pytest.param("/tmp/project", "overlay", id="where two are mounted, the last"),
        pytest.param("/home/user/sharesmore/project", "ext4", id="a name a mount point begins"),
        pytest.param("/home/user/shares", "nfs4", id="at a mount point itself"),
    ],
)
def test_a_path_lies_on_the_deepest_file_system_mounted_over_it(path: str, expected: str):
    assert file_system_type(path, MOUNT_INFO) == expected
