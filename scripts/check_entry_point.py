"""Refuse an entry point whose module imports the server when it is imported.

The console script that `uvx` runs imports `semantic_code_mcp.__main__` at its top, and a pool's
processes import the console script once more: Linux's forkserver once a session, each process
macOS and Windows spawn (ADR 010, O4). An entry module that imported `cli` there made the
forkserver hold 94 MB for the rest of the session, and would make each spawned worker import
`mcp` and the whole server. This imports the entry module alone, in a fresh interpreter, and
names every module of the package or of `mcp` that came with it:

    uv run python scripts/check_entry_point.py

Prints each module the entry module brought in, and exits 0 when it brought in none, or 1 when
it did.
"""

import importlib
import sys

PACKAGE = "semantic_code_mcp"
ENTRY_MODULE = "semantic_code_mcp.__main__"
# What importing the entry module may load of the package: the package and the module itself
LOADED_WITH_IT = frozenset({PACKAGE, ENTRY_MODULE})
# The packages whose modules it must not load, the package's own and the MCP SDK's
HEAVY_PACKAGES = (PACKAGE, "mcp")
MODULE_NAME_SEPARATOR = "."
ENTRY_POINT_LIGHT = 0
ENTRY_POINT_IMPORTS_THE_SERVER = 1


def brought_in() -> list[str]:
    """The modules of the heavy packages that importing the entry module loaded, beyond itself
    and its package."""
    importlib.import_module(ENTRY_MODULE)
    return sorted(
        name
        for name in sys.modules
        if name not in LOADED_WITH_IT and name.split(MODULE_NAME_SEPARATOR)[0] in HEAVY_PACKAGES
    )


def main() -> int:
    modules = brought_in()
    for module in modules:
        print(f"{ENTRY_MODULE} imports {module} as it is imported")
    return ENTRY_POINT_IMPORTS_THE_SERVER if modules else ENTRY_POINT_LIGHT


if __name__ == "__main__":
    sys.exit(main())
