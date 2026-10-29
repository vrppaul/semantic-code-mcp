"""Tests of what the speed harness reads of the server's processes."""

import subprocess
import sys
from pathlib import Path

import pytest
from evals.speed import SERVER_COMMAND_MARK, ServerMemory

# A stand-in server that starts a process holding memory, as a first index's pool does
SERVER_WITH_A_CHILD = Path(__file__).parents[1] / "data" / "server_with_a_child.py"
# What the stand-in's child holds: more than a Python process holds of its own
HELD_MEGABYTES = 64
NEXT_LINE = "\n"


def tell(server: subprocess.Popen[str]) -> None:
    """Have the stand-in server take its next step, and wait until it has."""
    server.stdin.write(NEXT_LINE)
    server.stdin.flush()
    server.stdout.readline()


@pytest.mark.skipif(sys.platform != "linux", reason="memory is read from Linux's /proc")
def test_a_server_s_peak_memory_counts_what_a_process_it_started_held_before_it_ended():
    """A first index's pool lives for seconds, its processes' memory the server's while it
    does; what the server holds after is its own and that of the processes left."""
    server = subprocess.Popen(  # nosec B603
        [
            sys.executable,
            str(SERVER_WITH_A_CHILD),
            SERVER_COMMAND_MARK.decode(),
            str(HELD_MEGABYTES),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    memory = ServerMemory()
    server.stdout.readline()
    memory.sample()
    tell(server)
    memory.sample()
    tell(server)

    peak = memory.peak_megabytes()
    resident = memory.resident_megabytes()
    server.stdin.close()
    server.wait()

    assert (peak >= HELD_MEGABYTES, resident >= HELD_MEGABYTES) == (True, False)
