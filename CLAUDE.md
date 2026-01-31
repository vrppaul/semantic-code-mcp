# Semantic Code MCP Server

Local MCP server providing semantic code search for Claude Code. Replaces iterative grep/glob with embedding-based vector search, so agents find code by meaning rather than exact text matching.

## Quick Reference

```bash
uv sync                               # Install dependencies
uv run python -m semantic_code_mcp    # Run server
uv run pytest                         # Run tests
uv run ruff check src/                # Lint
uv run ruff format src/               # Format
```

## Tech Stack

- **Python 3.14** — use modern syntax (type parameter syntax, match, walrus)
- **uv** — package and venv management (NEVER use pip directly)
- **FastMCP** — MCP server framework
- **sentence-transformers** — local embeddings (all-MiniLM-L6-v2, 384d)
- **LanceDB** — embedded vector database
- **tree-sitter** — AST-based code chunking
- **structlog** — structured logging
- **pydantic + pydantic-settings** — data models and configuration

## Boundaries

**Always:**
- Use `uv run` to execute anything, `uv add` to add dependencies
- Write tests before implementation (TDD)
- Run tests (`uv run pytest`) and linter (`uv run ruff check src/`) after changes
- Use structlog for logging, never print()
- Use type hints on public functions

**Ask first:**
- Changing data models or storage schema
- Adding new dependencies
- Modifying MCP tool signatures

**Never:**
- Use pip or activate venv manually
- Put code in `__init__.py` files
- Use generic Exception for error handling
- Commit secrets or credentials

## Commit Messages

Conventional Commits with **required scope** (`--strict --force-scope`).
Format: `type(scope): description`

Types: `feat fix docs style refactor perf test build ci chore`

## Key Architecture

The server lazily loads the embedding model on first query (with MCP progress notifications). Code is chunked via tree-sitter AST parsing, embedded in batches, and stored in LanceDB. Search embeds the query and performs vector similarity lookup with optional full-text hybrid search. Context-specific coding rules live in `.claude/rules/` and activate based on file glob patterns. Architecture decisions are in `docs/decisions/`. Project planning flows through TODO.md (epics) → decisions/ (how) → CHANGELOG.md (done).

## Maintaining This File

- Keep under 80 lines. Every line must prevent a concrete mistake.
- Only universally-applicable content — domain-specific rules go in `.claude/rules/`.
- When editing, prune outdated content first.
- Keep AGENTS.md in sync for commands, stack, and boundaries.
