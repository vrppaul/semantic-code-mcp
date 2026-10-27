"""Refuse a lock whose runtime packages a fresh install would not take.

`uvx semantic-code-mcp` ignores uv.lock: it resolves the newest versions within the bounds
of pyproject.toml, and only the runtime ones. The suite and the harness run on the lock, so
a release ships only a lock whose runtime packages are those such a resolve takes (ADR 006,
D14). Both sides leave the dev group out: no user installs it, so a new release of a dev
tool refuses nothing. A dev package that holds a runtime one back in the lock shows too, as
the lock's version then differs from the fresh resolve's.

Run in the repository's root, before tagging a release, weekly in CI after the lock is
moved, and in the release's own workflow, with any Python 3.12 or later:

    uv run --no-project python scripts/check_fresh_lock.py

Asks uv for the lock's runtime packages and for a fresh universal resolve of
pyproject.toml (from the package index), prints each requirement only one side holds, and
exits 0 when they match, 1 when they differ, or 2 when uv fails.
"""

import subprocess  # nosec B404
import sys

# The lock's runtime packages, as requirement lines with their markers
LOCKED_COMMAND = (
    "uv",
    "export",
    "--frozen",
    "--no-dev",
    "--no-hashes",
    "--no-emit-project",
    "--no-header",
    "--no-annotate",
)
# What a fresh install of the package resolves, for every platform, in the same form
FRESH_COMMAND = (
    "uv",
    "pip",
    "compile",
    "pyproject.toml",
    "--universal",
    "--upgrade",
    "--no-header",
    "--no-annotate",
    "--quiet",
)
LOCK_IS_FRESH = 0
LOCK_IS_STALE = 1
UV_FAILED = 2


def requirements(command: tuple[str, ...]) -> frozenset[str] | None:
    """The requirement lines a uv command prints, or None after printing why it failed."""
    shown = " ".join(command)
    try:
        # A command of this module's own, uv found on the PATH as the workflows install it
        completed = subprocess.run(  # nosec B603, B607
            command, capture_output=True, text=True, check=True
        )
    except OSError as error:
        print(f"{shown} did not start: {error}", file=sys.stderr)
        return None
    except subprocess.CalledProcessError as error:
        print(f"{shown} failed:\n{error.stderr}", file=sys.stderr)
        return None
    return frozenset(line for line in completed.stdout.splitlines() if line.strip())


def main() -> int:
    locked = requirements(LOCKED_COMMAND)
    fresh = requirements(FRESH_COMMAND)
    if locked is None or fresh is None:
        return UV_FAILED
    if locked == fresh:
        return LOCK_IS_FRESH
    for requirement in sorted(locked - fresh):
        print(f"locked, not taken fresh: {requirement}")
    for requirement in sorted(fresh - locked):
        print(f"taken fresh, not locked: {requirement}")
    print("Move the lock (`uv lock --upgrade`) and run the suite and both evals on it.")
    return LOCK_IS_STALE


if __name__ == "__main__":
    sys.exit(main())
