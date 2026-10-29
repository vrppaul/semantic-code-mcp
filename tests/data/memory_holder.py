"""A process that holds the megabytes given, says `ready`, and ends once its input closes:
the child of `server_with_a_child.py`.

    python memory_holder.py <megabytes>
"""

import sys

READY = "ready\n"
BYTES_PER_MEGABYTE = 1024 * 1024
# Bytes written, so every page held is in memory
HELD_BYTE = b"x"

held = HELD_BYTE * (int(sys.argv[1]) * BYTES_PER_MEGABYTE)
sys.stdout.write(READY)
sys.stdout.flush()
sys.stdin.read()
