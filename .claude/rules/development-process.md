# Development Process Rules

## uv Only
- **NEVER** install dependencies manually (no `pip install`)
- **NEVER** activate venv manually (no `source .venv/bin/activate`)
- **ALWAYS** use `uv add <package>` to add dependencies
- **ALWAYS** use `uv run <command>` to run anything
- `uv sync` to install/update all dependencies

## Implementation Workflow
1. **Create comprehensive todo list FIRST** before any implementation
2. **TDD / Spec-driven development** - write tests before implementation
3. **Each step must be tested**:
   - Manual verification
   - Unit tests (isolated component behavior)
   - Integration tests (components working together)
   - E2E tests (full workflow)
4. Only move to next step when current step passes all tests

## Dependencies
- mcp (FastMCP) - MCP server framework
- sentence-transformers - embeddings
- lancedb - vector storage
- tree-sitter + tree-sitter-python + tree-sitter-rust - AST parsing
- structlog - logging
- pydantic + pydantic-settings - models and config
