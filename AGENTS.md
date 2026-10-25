# Semantic Code MCP

A local MCP server that gives coding agents semantic code search: tree-sitter splits a project into chunks, sentence-transformers embeds them, LanceDB stores them, and `search_code` finds code by meaning.

## Commands

```bash
uv sync                               # install dependencies
uv run python -m semantic_code_mcp    # run the server over stdio
uv run pytest                         # the tests
uv run pre-commit run --all-files     # every gate a commit runs, over every file
uv run python -m evals quality        # search quality on the pinned corpora; `speed` for speed
```

- The gates are one list, `.pre-commit-config.yaml`, each with its command; CI runs the same list. Leave a hook out only on purpose, by naming it in `SKIP`.
- Everything runs through `uv`. Add a dependency with `uv add` (dev: `uv add --dev`): never pip, an activated virtualenv, or a hand edit of a dependency list.

## Stack

Python 3.14 · uv · mcp (FastMCP, pinned below 2 until the migration) · sentence-transformers (all-MiniLM-L6-v2, 384 dimensions) · LanceDB · tree-sitter (Python, Rust, Markdown) · structlog · pydantic and pydantic-settings.

## How we work

- **The plan is ADR 006.** `docs/decisions/006-rebuild-on-mcp-2.md` holds the stages; work follows them and the ADR changes when the plan does.
- **Small chunks.** One logical piece at a time, committed on its own, then reviewed by an independent reviewer before the next.
- **Tests hold behaviour.** A behaviour someone can name gets a test first, seen failing, at the cheapest level: `.claude/rules/testing.md`.
- **Review before presenting.** Read every changed line: the right layer? redundant work? every line needed?
- **Speed and search quality are measured**, never judged by feel; a change to either says its numbers.
- **Ask first** before adding a dependency, changing the storage schema, a domain model or an MCP tool's signature.
- Never commit secrets or credentials.
- A commit follows Conventional Commits with a required scope, `type(scope): description`.
- Planned work is in `TODO.md`, where an item done is removed; how it is built, in `docs/decisions/`; what changed, in `CHANGELOG.md`.

## Architecture

- `src/semantic_code_mcp/server.py` holds the MCP tools; `src/semantic_code_mcp/container.py` wires them to `src/semantic_code_mcp/services/`.
- Indexing: `src/semantic_code_mcp/discovery.py` lists the files, `src/semantic_code_mcp/chunkers/` split them, `src/semantic_code_mcp/embedder.py` embeds the chunks, `src/semantic_code_mcp/storage/` keeps vectors and the mtime cache.
- Search re-indexes stale files, embeds the query, merges vector and full-text hits, filters and ranks them.
- The target layering, models and storage are ADR 006, D2–D4.

## Rules for all code

- A name says what it holds, in full words: `first_row`, `project_root`, never `r`, `t0`, `idx`.
- No function inside another: a helper is a function or method of the module, with a name and parameters.
- No literal with a meaning in an expression: a number, a string key, a timeout, a limit is a named constant or data, with a comment saying what it is.
- No tuple where a type belongs: a named pydantic model, never `list[tuple[X, Y]]`.
- No alias or compatibility shim: rename a thing and fix every user.
- No new module-level mutable state; what is shared is built in one place and passed in.
- A parameter every caller must give has no default.
- Exceptions are specific types, never a bare `Exception`; log with structlog, never `print()`.
- What is not obvious gets a one-line comment; a module's docstring says why it exists.

## Only better

Every change leaves the repository better than it found it, in every part it touches. A change that only works is not done.

- **Construction first.** Settle who owns each state, which layer it lives in and the contract before writing code.
- **Simplest first.** A new state, table, gate or module says why nothing simpler does.
- **Root cause first.** Before a fix, say why the fault can happen and change that; a guard hiding a wrong value needs a reason.
- **Show it with facts.** Say what a reader can check; say as plainly what did not get better.
- **The verdict is somebody else's.** A chunk is done when an independent reviewer has read its whole diff.
- **A regression found becomes a gate** where a tool can hold it.

## Rules by area

Before you change or review code in an area, read its file whole; a brief to another agent names the files.

- `.claude/rules/python.md` — any Python file: types, enums, logging, errors, modules.
- `.claude/rules/mcp-server.md` — the MCP tools in `src/semantic_code_mcp/server.py`.
- `.claude/rules/testing.md` — any test under `tests/`.
- `.claude/rules/scripts.md` — the gates of our own under `scripts/`.
- `.claude/rules/documentation.md` — `TODO.md`, `CHANGELOG.md` and the decisions under `docs/decisions/`.
- `.claude/rules/publishing.md` — a release.
- `.claude/rules/instructions.md` — this file and the rules of each area.
