# AGENTS.md

## Overview

Semantic Code MCP Server — a local MCP server providing semantic code search for AI coding agents. Indexes codebases using tree-sitter AST parsing and sentence-transformer embeddings, stores vectors in LanceDB, and returns ranked code snippets by meaning rather than exact text matching.

## Build & Test Commands

```bash
uv sync                               # Install dependencies
uv run python -m semantic_code_mcp    # Run server
uv run pytest                         # Run tests
uv run ruff check src/                # Lint
uv run ruff format src/               # Format
```

All commands use `uv` — never use pip or manually activate a virtualenv.

## Tech Stack

- Python 3.14 (modern syntax: type parameter syntax, match statements, union types)
- FastMCP (MCP server framework)
- sentence-transformers (all-MiniLM-L6-v2, 384-dimensional embeddings)
- LanceDB (embedded vector database)
- tree-sitter + tree-sitter-python + tree-sitter-rust (AST-based code chunking)
- structlog (structured logging)
- pydantic + pydantic-settings (data models and configuration)

## Code Style

```python
import structlog

log = structlog.get_logger()


async def search_code(
    query: str,
    project_path: str,
    limit: int = 10,
) -> list[SearchResult]:
    """Search for code semantically similar to the query."""
    start = time.monotonic()
    results = await searcher.search(query, project_path, limit=limit)
    log.debug("search completed", duration=time.monotonic() - start, num_results=len(results))
    return results
```

Key conventions:
- Type hints on all public functions (modern syntax: `str | None` not `Optional[str]`)
- structlog for all logging, never print()
- Named models for complex types (`list[ChunkResult]` not `list[tuple[Chunk, float]]`)
- Async where appropriate (MCP server is async)

## Conventions

- **TDD**: Write tests before implementation
- **Modules**: No code in `__init__.py` — only re-exports
- **Error handling**: Specific exceptions, not generic Exception
- **Dependencies**: Add via `uv add`, never pip install
- **Logging**: DEBUG for timing/perf, INFO for operations, WARNING for recoverable issues, ERROR for failures
- **Commits**: Conventional Commits, required scope (`type(scope): description`). Types: `feat fix docs style refactor perf test build ci chore`

## Releases

Published to PyPI as `semantic-code-mcp`. Version derived from git tags (`hatch-vcs`). Push a `v*` tag to trigger automated build + publish via GitHub Actions (trusted publishers OIDC). Never hardcode a version in `pyproject.toml`.

## Boundaries

**Always do:**
- Run tests after changes (`uv run pytest`)
- Run linter after changes (`uv run ruff check src/`)
- Use type hints on public functions
- Log duration for performance-sensitive operations

**Ask first:**
- Changing data models or storage schema
- Adding new dependencies
- Modifying MCP tool signatures

**Never do:**
- Use pip or manually activate venv
- Put code in `__init__.py`
- Use print() for output
- Commit secrets or credentials
