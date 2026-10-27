# Semantic Code MCP

A local MCP server that gives coding agents semantic code search: tree-sitter splits a project into chunks, onnxruntime embeds them with a model's ONNX export, LanceDB stores them, and `search_code` finds code by meaning.

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

Python 3.14 · uv · mcp 2.x (MCPServer) · onnxruntime and tokenizers (all-MiniLM-L6-v2's ONNX export, 384 dimensions) · LanceDB · tree-sitter (Python, Rust, Markdown) · structlog · pydantic and pydantic-settings.

## How we work

- **The plan is ADR 006.** `docs/decisions/006-rebuild-on-mcp-2.md` holds the stages; work follows them and the ADR changes when the plan does.
- **Chunks of work.** A chunk is a large block of the plan, a group of an ADR's steps, in many commits; an independent reviewer (`.claude/agents/search-reviewer.md`) reads its whole range once, before the next starts.
- **Product first.** Effort goes to what users get, and to what measures or gates it: `evals/` and the gates pre-commit or CI runs.
- A helper for the author's own work (a probe, a runner, a check run by hand) is throwaway code in the session's scratch directory: no tests, no review, never committed; where git, pytest or shell does the job, none is written.
- **Tests hold behaviour.** A behaviour someone can name gets a test first, seen failing, at the cheapest level: `.claude/rules/testing.md`.
- **Review before presenting.** Read every changed line: the right layer? redundant work? every line needed?
- **Speed and search quality are measured**, never judged by feel; a change to either says its numbers.
- **Ask first** before adding a dependency, changing the storage schema, a domain model or an MCP tool's signature.
- Never commit secrets or credentials.
- A commit follows Conventional Commits with a required scope, `type(scope): description`.
- Planned work is in `TODO.md`, where an item done is removed; how it is built, in `docs/decisions/`; what changed, in `CHANGELOG.md`.

## Architecture

All under `src/semantic_code_mcp/`, each layer building only on those after it:

- cli → app → server → services → the parts → domain. `uv run lint-imports` refuses an import against them; `pyproject.toml` holds the contracts and their reasons.
- The parts: discovery lists the files, the chunkers split them, embedding makes the vectors, storage keeps them with a record of each file and a manifest of what built the index.
- `mcp` stays in the server layer, the app and the cli; each library stays in the part that wraps it.
- The app builds the server; its lifespan holds the one `ProjectRegistry` every request shares. Nothing is module-level state.
- Interfaces are abstract classes in `src/semantic_code_mcp/domain/interfaces.py`; tests use the fakes in `tests/fakes.py`.

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
- `.claude/rules/mcp-server.md` — the MCP tools and their answers under `src/semantic_code_mcp/server/`.
- `.claude/rules/testing.md` — any test under `tests/`.
- `.claude/rules/scripts.md` — the gates of our own under `scripts/`.
- `.claude/rules/documentation.md` — `TODO.md`, `CHANGELOG.md` and the decisions under `docs/decisions/`.
- `.claude/rules/publishing.md` — a release.
- `.claude/rules/instructions.md` — this file and the rules of each area.
