# Semantic Code MCP Server

## Project Overview

Local MCP server providing semantic code search for Claude Code. Replaces iterative grep with embedding-based vector search.

## Quick Reference

```bash
uv sync                               # Install dependencies
uv run python -m semantic_code_mcp    # Run server
uv run pytest                         # Run tests
uv run ruff check src/                # Lint
uv run ruff format src/               # Format
```

## Documentation Structure

- **TODO.md** - High-level epics with rationale
- **CHANGELOG.md** - Completed work (Keep a Changelog format)
- **docs/decisions/** - Implementation plans and architectural decisions

## Rules (`.claude/rules/`)

Rules are auto-loaded by Claude Code. Some apply conditionally based on file paths.

| File | Applies To | Purpose |
|------|------------|---------|
| `python.md` | `**/*.py` | Python coding standards: type hints, structlog logging, error handling, modern syntax, no code in `__init__.py` |
| `testing.md` | `tests/**/*.py` | TDD philosophy, test-what-not-how, spec-driven development |
| `mcp-server.md` | `server.py`, `__init__.py` | MCP-specific: progress notifications, tool design, performance targets |
| `documentation.md` | All files | Documentation system: TODO→decisions→CHANGELOG flow, ADR format |
| `development-process.md` | All files | uv-only workflow, implementation steps, dependency management |

## MCP Tools Available

Use these tools during development:

### Context7
Use for looking up library documentation:
- `resolve-library-id` → `query-docs`
- Example: researching FastMCP API, LanceDB usage, tree-sitter bindings
- When: unsure about library API, need code examples, checking latest patterns

### Sequential Thinking
Use for complex problem solving:
- Breaking down multi-step problems
- Analyzing trade-offs between approaches
- Debugging complex issues
- When: stuck on a problem, need structured analysis

## Tech Stack

- **Python 3.14** - Latest, use modern syntax
- **uv** - Package and venv management (NEVER use pip directly)
- **FastMCP** - MCP server framework
- **sentence-transformers** - Local embeddings (all-MiniLM-L6-v2, 384d)
- **LanceDB** - Embedded vector database
- **tree-sitter** - AST-based code chunking
- **structlog** - Structured logging
- **pydantic** - Data models and settings

## Architecture Decisions

See `docs/decisions/` for detailed rationale. Key decisions:

1. **Model loading**: Lazy (on first query) with MCP progress notifications
2. **Index storage**: Configurable (`--index-dir` or `--local-index`)
3. **Distribution**: PyPI package, run via `uvx semantic-code-mcp`
4. **Languages**: Python first, then JS/TS, Rust, Go

## Project Structure

```
src/semantic_code_mcp/
├── __init__.py           # Re-exports only
├── __main__.py           # python -m entry point
├── cli.py                # main() entry point
├── server.py             # FastMCP server and tool definitions
├── config.py             # Pydantic settings, CLI args
├── models.py             # Data models (Chunk, SearchResult, IndexStatus)
├── indexer/
│   ├── chunker.py        # tree-sitter AST parsing
│   ├── embedder.py       # Embedding generation wrapper
│   └── indexer.py        # Orchestration, file scanning
├── storage/
│   ├── lancedb.py        # Vector store operations
│   └── cache.py          # File change detection (mtime)
└── search/
    └── searcher.py       # Query embedding + ranking
tests/
docs/
├── decisions/            # Architecture decision records
TODO.md
CHANGELOG.md
```

## Data Flow

**Indexing:**
1. Scan files (respect .gitignore)
2. Check cache (skip unchanged via mtime)
3. Parse with tree-sitter → extract functions, classes, methods
4. Generate embeddings (batch)
5. Store in LanceDB

**Search:**
1. Embed query
2. Vector similarity search in LanceDB
3. Post-filter (file pattern, min score)
4. Return file:line + snippets
