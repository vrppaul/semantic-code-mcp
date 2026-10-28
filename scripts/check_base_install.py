"""Refuse a package whose modules do not import without its extras' libraries.

The dev group installs every extra, so the suite and every other gate run with LanceDB's
and the model's libraries present, and none of them fails where a user installed the
package alone. A module that imports an extra's library, or a module an extra holds, at its
top would stop every server started without that extra. This imports every module of the
package but those an extra holds, in an environment of the package and its runtime
dependencies alone, which uv makes apart from the project's own:

    uv run --isolated --no-dev python scripts/check_base_install.py

Prints each module that fails to import, with why, and exits 0 when every one imports, or 1
when one does not.
"""

import importlib
import pkgutil
import sys

import semantic_code_mcp

# The modules an extra holds, each imported only when its store is chosen or its model
# loads, as each needs the extra's libraries
EXTRA_MODULES = frozenset(
    {"semantic_code_mcp.storage.lancedb", "semantic_code_mcp.embedding.onnx_model"}
)
MODULE_NAME_SEPARATOR = "."
EVERY_MODULE_IMPORTS = 0
A_MODULE_FAILS = 1


def failed_imports() -> list[str]:
    """Each module of the package but an extra's that does not import, with why."""
    prefix = semantic_code_mcp.__name__ + MODULE_NAME_SEPARATOR
    failures: list[str] = []
    for module in pkgutil.walk_packages(semantic_code_mcp.__path__, prefix):
        if module.name in EXTRA_MODULES:
            continue
        try:
            importlib.import_module(module.name)
        except ImportError as error:
            failures.append(f"{module.name}: {error}")
    return failures


def main() -> int:
    failures = failed_imports()
    for failure in failures:
        print(failure)
    return A_MODULE_FAILS if failures else EVERY_MODULE_IMPORTS


if __name__ == "__main__":
    sys.exit(main())
