# 001: Initial Architecture

**Status**: accepted
**Date**: 2025-01-24

## Context

Claude Code uses iterative grep searches to find code. This works but requires knowing exact terms. For concept-based searches ("function that validates user input"), grep fails because we don't know what the author named things.

We need semantic search that understands code meaning, not just text matching.

## Decision

Build a local MCP server with:

### Tech Stack
| Component | Choice | Rationale |
|-----------|--------|-----------|
| MCP Framework | FastMCP | Python decorators, simple STDIO transport |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Local, no API costs, 384d vectors, ~80MB model |
| Vector Store | LanceDB | Embedded like SQLite, no server needed |
| Chunking | tree-sitter | AST-based, respects code structure |

### Key Design Choices

1. **Lazy model loading**: Load embedding model on first query, not server start. Send MCP progress notifications during load.

2. **Configurable index storage**:
   - Default: `~/.cache/semantic-code-mcp/<path-hash>/`
   - Optional: `.semantic-code/` in project root via `--local-index`

3. **Distribution via uvx**: Same pattern as npx-based MCP servers. First run downloads deps, subsequent runs use cache.

4. **Python first**: Start with Python via tree-sitter-python, add other languages later.

### MCP Tools
- `semantic_search(query, path, limit, file_pattern)` - Main search
- `index_codebase(path, force, incremental)` - Build/update index
- `index_status(path)` - Check index state
- `find_similar(file_path, line, limit)` - Find similar code

## Alternatives Considered

### Remote API for embeddings (OpenAI, etc.)
Rejected: Sends code to external servers (privacy), requires API keys, has costs, needs internet.

### FAISS instead of LanceDB
Rejected: FAISS is more complex to set up, doesn't persist to disk as easily, LanceDB is simpler for our embedded use case.

### Line-based chunking instead of AST
Rejected: Breaks code mid-function, loses semantic boundaries. AST chunking respects code structure.

### Eager model loading
Rejected: 2-3s delay on every server start. Lazy loading means fast start, one-time delay on first query per session.

## Consequences

### Positive
- Fully local, private, no API costs
- Works offline
- Fast after initial model load
- AST chunking gives high-quality semantic boundaries

### Negative
- Large dependencies (torch ~2GB for sentence-transformers)
- First `uvx` run is slow (downloading deps)
- Cold start on first query per session (~2-3s)
- Initially Python only

### Risks
- sentence-transformers may not embed code well (mitigate: can swap to code-specific model like UniXcoder)
- LanceDB may not scale to very large codebases (mitigate: can add IVF indexing later)
