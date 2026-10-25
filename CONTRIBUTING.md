# Contributing

## Setup

```bash
uv sync
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
```

Everything runs through `uv`. The first test run downloads the embedding model (about 90 MB) from Hugging Face.

## Development

```bash
uv run pytest                        # the tests
uv run pre-commit run --all-files    # every gate a commit runs
uv run python -m semantic_code_mcp   # the server over stdio
```

The rules every change is held to are in `AGENTS.md`, and those of each area under `.claude/rules/`. The plan the project follows is `docs/decisions/006-rebuild-on-mcp-2.md`.

## Conventions

- Commits follow Conventional Commits with a required scope, `type(scope): description`; the `commit-msg` hook checks them.
- A behaviour gets its test first, seen failing (`.claude/rules/testing.md`).
- What changed goes in `CHANGELOG.md` under `[Unreleased]`; what is planned, in `TODO.md`; how an epic is built, in `docs/decisions/`.

## Architecture

- `src/semantic_code_mcp/server.py` — the three MCP tools.
- `src/semantic_code_mcp/container.py` — builds and shares the model, the stores and the services.
- `src/semantic_code_mcp/services/` — indexing (scan, detect changes, chunk, embed) and search (re-index stale files, embed the query, rank).
- `src/semantic_code_mcp/discovery.py` — which files get indexed.
- `src/semantic_code_mcp/chunkers/` — a tree-sitter chunker per language behind one dispatcher.
- `src/semantic_code_mcp/storage/` — LanceDB and the mtime cache.

ADR 006 (D2–D4) describes the structure this is moving to.

## Releasing

1. Move the changelog's `[Unreleased]` entries under `[x.y.z] - date`, in one commit on `master`, and push it.
2. Tag it `vx.y.z` and push the tag. `hatch-vcs` takes the version from the tag; nothing else holds one.
3. `.github/workflows/publish.yml` runs the whole of `.github/workflows/ci.yml` on the tagged commit, then waits on the `pypi` environment.
4. Approve the deployment on GitHub. Only then does the job build, publish to PyPI through trusted publishing, and create the GitHub Release.
