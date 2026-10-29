"""The server's entry point, for `python -m semantic_code_mcp` and the console script alike.

A pool's processes import the main module once more (ADR 010, O4): Linux's forkserver once a
session, a process macOS or Windows spawns each time. The console script's main module imports
this one, so it imports the server only when it runs: importing it here made the forkserver
take 0.6 s and hold 94 MB for the rest of the session, where it holds 19.
"""


def main() -> None:
    """Run the server over stdio until the client hangs up."""
    # Imported to run, never on import: see the module's docstring
    from semantic_code_mcp import cli  # noqa: PLC0415

    cli.main()


if __name__ == "__main__":
    main()
