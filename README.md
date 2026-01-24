# semantic-code-mcp

Local MCP server that provides semantic code search for Claude Code, replacing iterative grep searches with direct embedding-based queries.

## Architecture

```
Claude Code ──(MCP/STDIO)──▶ semantic-code-mcp server
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              AST Chunker      Embedder        LanceDB
             (tree-sitter)  (sentence-trans)  (vectors)
```

## Installation

```bash
# Via uvx (recommended)
uvx semantic-code-mcp

# Or install globally
uv tool install semantic-code-mcp
```

## Claude Code Integration

Add to `~/.config/claude-code/config.json`:

```json
{
  "mcpServers": {
    "semantic-search": {
      "command": "uvx",
      "args": ["semantic-code-mcp"]
    }
  }
}
```

### Options

```json
{
  "mcpServers": {
    "semantic-search": {
      "command": "uvx",
      "args": [
        "semantic-code-mcp",
        "--index-dir", "~/.cache/semantic-code-mcp"
      ]
    }
  }
}
```

- `--index-dir PATH` - Store indexes in specified directory (default: `~/.cache/semantic-code-mcp`)
- `--local-index` - Store index in `.semantic-code/` within each project

## MCP Tools

### `semantic_search`

Search code semantically by meaning, not just text matching.

```python
semantic_search(
    query: str,              # "function that validates user input"
    path: str = ".",         # Codebase root
    limit: int = 10,         # Max results
    file_pattern: str = None # "*.py", "tests/**"
) -> list[SearchResult]
```

Returns: `file_path`, `line_start`, `line_end`, `snippet`, `score`, `chunk_type`

### `index_codebase`

Index a codebase for semantic search.

```python
index_codebase(
    path: str,
    force: bool = False,     # Rebuild from scratch
    incremental: bool = True # Only changed files
) -> IndexStatus
```

### `index_status`

Check indexing status for a codebase.

```python
index_status(path: str = ".") -> IndexStatus
```

Returns: `is_indexed`, `last_updated`, `files_count`, `chunks_count`, `stale_files`

### `find_similar`

Find code similar to a specific location.

```python
find_similar(
    file_path: str,
    line: int,
    limit: int = 5
) -> list[SearchResult]
```

## Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| MCP Framework | FastMCP | Python decorators, simple STDIO transport |
| Embeddings | sentence-transformers | Local, no API costs, good quality |
| Vector Store | LanceDB | Embedded (like SQLite), no server needed |
| Chunking | tree-sitter | AST-based, respects code structure |

## Configuration

Optional config file at `~/.config/semantic-code-mcp/config.yaml`:

```yaml
embedding:
  model: "all-MiniLM-L6-v2"  # or "microsoft/unixcoder-base"
  device: "auto"

storage:
  cache_dir: "~/.cache/semantic-code-mcp"

chunking:
  target_tokens: 800
  max_tokens: 1500

ignore:
  patterns: ["node_modules/**", ".venv/**", "__pycache__/**"]
  use_gitignore: true
```

## Development

```bash
# Setup
uv sync

# Run server locally
uv run python -m semantic_code_mcp

# Run tests
uv run pytest

# Lint & format
uv run ruff check src/
uv run ruff format src/
```

## Performance

| Metric | Target |
|--------|--------|
| Search latency | < 100ms (after model loaded) |
| Index speed | > 50 files/sec |
| Cold start | < 3s (first query, model loading) |
| Memory | < 500MB |

## License

MIT
