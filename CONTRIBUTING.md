# Contributing

## Setup

```bash
uv sync
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
```

Everything runs through `uv`. The first test run downloads the embedding model (about 90 MB) from Hugging Face: the tests hold the search by meaning too, which a server's setting turns on.

## Development

```bash
uv run pytest                        # the tests
uv run pre-commit run --all-files    # every gate a commit runs
uv run python -m semantic_code_mcp   # the server over stdio
uv run python -m evals quality       # search quality on the pinned corpora, against evals/baseline/
uv run python -m evals speed         # indexing and search speed, memory, index growth
uv run python -m evals agents --corpus-file <file>   # agents' sessions by arm; spends a Claude subscription
```

A change to search, chunking or speed runs the matching `evals` command before and after and says both numbers in its commit.

The rules every change is held to are in `AGENTS.md`, and those of each area under `.claude/rules/`. The plan the project follows is `docs/decisions/006-rebuild-on-mcp-2.md`.

## Conventions

- Commits follow Conventional Commits with a required scope, `type(scope): description`; the `commit-msg` hook checks them.
- A behaviour gets its test first, seen failing (`.claude/rules/testing.md`).
- What changed goes in `CHANGELOG.md` under `[Unreleased]`; what is planned, in `TODO.md`; how an epic is built, in `docs/decisions/`.

## Architecture

`src/semantic_code_mcp/` is layered, each part building only on those after it, which import-linter holds (`pyproject.toml`):

- `cli.py`, `app.py` — the entry point and the composition root: settings, logging, the server, its lifespan, and the store and chunkers every project gets.
- `server/` — the three MCP tools, their answers and presenters, the profiling middleware.
- `services/` — indexing (scan, plan against the stored file records, chunk, then one write of the words), search (re-index stale files, rank by words; by meaning too, embed the chunks without vectors and the query, and fuse), the vectors, and the registry of projects.
- `discovery.py`, `chunkers/`, `embedding/`, `storage/` — the parts: which files, how they split, their vectors (the model loaded from the `meaning` extra), where they are kept.
- `domain/` — the models and the interfaces the parts implement.

ADR 006 (D2–D4) describes where this is still moving.

## Releasing

1. Move the changelog's `[Unreleased]` entries under `[x.y.z] - date`, in one commit on `master`, and push it.
2. Tag it `vx.y.z` and push the tag. `hatch-vcs` takes the version from the tag; nothing else holds one.
3. `.github/workflows/publish.yml` runs the whole of `.github/workflows/ci.yml` on the tagged commit, then enters the `pypi` environment.
4. With a required reviewer set on that environment (Settings → Environments → pypi; not set as of 2026-10-25), approve the deployment on GitHub. Then the job builds, publishes to PyPI through trusted publishing, and creates the GitHub Release.
