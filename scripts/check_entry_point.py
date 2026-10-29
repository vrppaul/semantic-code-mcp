"""Refuse a console script whose module imports the server when it is imported.

The console script that `uvx` runs imports the module `pyproject.toml`'s `[project.scripts]`
names at its top, and a pool's processes import the console script once more: Linux's
forkserver once a session, each process macOS and Windows spawn (ADR 010, O4). A script that
named `semantic_code_mcp.cli` made the forkserver hold 94 MB for the rest of the session, and
would make each spawned worker import `mcp` and the whole server. This imports each script's
module alone, in a fresh interpreter, and names every module of the package or of `mcp` that
came with it:

    uv run python scripts/check_entry_point.py

Prints each module a script's module brought in, and exits 0 when they brought in none, or 1
when one did.
"""

import importlib
import sys
import tomllib
from pathlib import Path

PACKAGE = "semantic_code_mcp"
PYPROJECT = Path("pyproject.toml")
# Where a console script names its module and function: `module:function`
SCRIPT_TARGET_SEPARATOR = ":"
# The packages whose modules it must not load, the package's own and the MCP SDK's
HEAVY_PACKAGES = (PACKAGE, "mcp")
MODULE_NAME_SEPARATOR = "."
ENTRY_POINT_LIGHT = 0
ENTRY_POINT_IMPORTS_THE_SERVER = 1


def script_modules() -> list[str]:
    """The module each console script of the package names."""
    scripts = tomllib.loads(PYPROJECT.read_text())["project"]["scripts"]
    return sorted({target.split(SCRIPT_TARGET_SEPARATOR)[0] for target in scripts.values()})


def brought_in(modules: list[str]) -> list[str]:
    """The modules of the heavy packages that importing the scripts' modules loaded, beyond
    those modules and the package."""
    for module in modules:
        importlib.import_module(module)
    loaded_with_them = {PACKAGE, *modules}
    return sorted(
        name
        for name in sys.modules
        if name not in loaded_with_them and name.split(MODULE_NAME_SEPARATOR)[0] in HEAVY_PACKAGES
    )


def main() -> int:
    modules = script_modules()
    heavy = brought_in(modules)
    named = ", ".join(modules)
    for module in heavy:
        print(f"a console script's module ({named}) imports {module} as it is imported")
    return ENTRY_POINT_IMPORTS_THE_SERVER if heavy else ENTRY_POINT_LIGHT


if __name__ == "__main__":
    sys.exit(main())
