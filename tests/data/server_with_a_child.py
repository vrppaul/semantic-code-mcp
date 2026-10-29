"""A server as the speed harness reads its memory, with a process of its own beside it, as a
first index's pool is: it says `ready`; at the next line it reads it starts `memory_holder.py`
holding the megabytes given and says `holding`; at the line after it ends that child and says
`ended`; it stops once its input closes. Its command line holds the mark the harness knows a
server by, as its first argument.

    python server_with_a_child.py <mark> <megabytes>
"""

import subprocess  # nosec B404
import sys
from pathlib import Path

READY = "ready\n"
HOLDING = "holding\n"
ENDED = "ended\n"
HOLDER = Path(__file__).with_name("memory_holder.py")


def say(line: str) -> None:
    sys.stdout.write(line)
    sys.stdout.flush()


say(READY)
sys.stdin.readline()
holder = subprocess.Popen(  # nosec B603
    [sys.executable, str(HOLDER), sys.argv[2]],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
)
holder.stdout.readline()
say(HOLDING)
sys.stdin.readline()
holder.stdin.close()
holder.wait()
say(ENDED)
sys.stdin.read()
