# semantic-code-mcp

Local MCP server that provides semantic code search for Claude Code. Instead of iterative grep/glob, it indexes your codebase with embeddings and returns ranked results by meaning.

**Python only** for now — multi-language support (JS/TS, Rust, Go) is planned.

## How It Works

```
Claude Code ──(MCP/STDIO)──▶ semantic-code-mcp server
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              AST Chunker      Embedder        LanceDB
             (tree-sitter)  (sentence-trans)  (vectors)
```

1. **Chunking** — tree-sitter parses Python into functions, classes, and methods
2. **Embedding** — sentence-transformers encodes each chunk (all-MiniLM-L6-v2, 384d)
3. **Storage** — vectors stored in LanceDB (embedded, like SQLite)
4. **Search** — hybrid semantic + keyword search with recency boosting

Indexing is incremental (mtime-based) and uses `git ls-files` for fast file discovery. The embedding model loads lazily on first query.

## Installation

### macOS / Windows

PyPI ships CPU-only torch on these platforms, so no extra flags are needed (~1.7GB install).

```bash
uvx semantic-code-mcp
```

**Claude Code integration:**

```bash
claude mcp add --scope user semantic-code -- uvx semantic-code-mcp
```

### Linux

> [!IMPORTANT]
> Without the `--index` flag, PyPI installs CUDA-bundled torch (~3.5GB). Unless you need GPU acceleration (you don't — embeddings run on CPU), use the command below to get the CPU-only build (~1.7GB).

```bash
uvx --index pytorch-cpu=https://download.pytorch.org/whl/cpu semantic-code-mcp
```

**Claude Code integration:**

```bash
claude mcp add --scope user semantic-code -- \
  uvx --index pytorch-cpu=https://download.pytorch.org/whl/cpu semantic-code-mcp
```

<details>
<summary>Claude Desktop / other MCP clients (JSON config)</summary>

```json
{
  "mcpServers": {
    "semantic-code": {
      "command": "uvx",
      "args": ["--index", "pytorch-cpu=https://download.pytorch.org/whl/cpu", "semantic-code-mcp"]
    }
  }
}
```

On macOS/Windows you can omit the `--index` and `pytorch-cpu` args.

</details>

## MCP Tools

### `search_code`

Search code by meaning, not just text matching. Auto-indexes on first search.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | required | Natural language description of what you're looking for |
| `project_path` | `str` | required | Absolute path to the project root |
| `limit` | `int` | `10` | Maximum number of results |

Returns ranked results with `file_path`, `line_start`, `line_end`, `name`, `chunk_type`, `content`, and `score`.

### `index_codebase`

Index a codebase for semantic search. Only processes new and changed files unless `force=True`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project_path` | `str` | required | Absolute path to the project root |
| `force` | `bool` | `False` | Re-index all files regardless of changes |

### `index_status`

Check indexing status for a project.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project_path` | `str` | required | Absolute path to the project root |

Returns `is_indexed`, `files_count`, and `chunks_count`.

## Configuration

All settings are environment variables with the `SEMANTIC_CODE_MCP_` prefix (via pydantic-settings):

| Variable | Default | Description |
|----------|---------|-------------|
| `SEMANTIC_CODE_MCP_CACHE_DIR` | `~/.cache/semantic-code-mcp` | Where indexes are stored |
| `SEMANTIC_CODE_MCP_LOCAL_INDEX` | `false` | Store index in `.semantic-code/` within each project |
| `SEMANTIC_CODE_MCP_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `SEMANTIC_CODE_MCP_DEBUG` | `false` | Enable debug logging |
| `SEMANTIC_CODE_MCP_PROFILE` | `false` | Enable pyinstrument profiling |

Pass environment variables via the `env` field in your MCP config:

```json
{
  "mcpServers": {
    "semantic-code": {
      "command": "uvx",
      "args": ["semantic-code-mcp"],
      "env": {
        "SEMANTIC_CODE_MCP_DEBUG": "true",
        "SEMANTIC_CODE_MCP_LOCAL_INDEX": "true"
      }
    }
  }
}
```

Or with Claude Code CLI:

```bash
claude mcp add --scope user semantic-code \
  -e SEMANTIC_CODE_MCP_DEBUG=true \
  -e SEMANTIC_CODE_MCP_LOCAL_INDEX=true \
  -- uvx semantic-code-mcp
```

## Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| MCP Framework | FastMCP | Python decorators, STDIO transport |
| Embeddings | sentence-transformers | Local, no API costs, good quality |
| Vector Store | LanceDB | Embedded (like SQLite), no server needed |
| Chunking | tree-sitter | AST-based, respects code structure |

## Development

```bash
uv sync                            # Install dependencies
uv run python -m semantic_code_mcp # Run server
uv run pytest                      # Run tests
uv run ruff check src/             # Lint
uv run ruff format src/            # Format
```

Architecture decisions are documented in `docs/decisions/`. Project planning lives in `TODO.md`.

## License

MIT
