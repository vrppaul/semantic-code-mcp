---
paths:
  - "**/*.py"
---

# Python Code Rules

## Language & Runtime
- Python 3.14 required
- Use modern Python features: type parameter syntax, match statements, walrus operator
- All async where appropriate (MCP server is async)

## Type Hints
- Required on all public functions
- Use modern syntax: `list[str]` not `List[str]`, `str | None` not `Optional[str]`
- Use Pydantic models for complex data structures
- **NEVER** use complex nested types like `list[tuple[X, Y]]` or `dict[str, tuple[int, str]]`
  - Create a named model/dataclass instead: `list[ChunkWithEmbedding]` not `list[tuple[Chunk, list[float]]]`
  - Types should be self-documenting and readable

## Logging
- ALWAYS use structlog: `log = structlog.get_logger()`
- NEVER use print() for any output
- Log levels:
  - DEBUG: timing, performance metrics, internal state
  - INFO: operations (indexing started, search completed)
  - WARNING: recoverable issues
  - ERROR: failures
- Always log duration for performance-sensitive operations

## Error Handling
- Use specific exceptions, not generic Exception
- Log errors with full context before re-raising

## Enums
- Use `StrEnum` + `auto()` for all string enums (lowercase values auto-generated from member names)
- Never use `str, Enum` base classes — always `StrEnum`
- For Python keywords (e.g. `class`, `for`), use an alias with explicit value: `klass = "class"`, `for_kw = "for"`
- Tree-sitter node types must be `NodeType(StrEnum)` enums per chunker — no raw string comparisons

## Module Structure
- Every `src/semantic_code_mcp/**/__init__.py` holds a docstring only; code lives in named modules and imports name the module (`semantic_code_mcp.models.domain`).
- `src/semantic_code_mcp/models/__init__.py` still re-exports; that goes in ADR 006, stage 4.
