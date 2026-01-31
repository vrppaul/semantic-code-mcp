# 003: Module Docstring Chunking

**Status**: implemented
**Date**: 2026-01-31

## Context

Searching conceptual queries like "dependency injection container" failed to find `container.py` because its module docstring ("Dependency injection container. Shares expensive resources...") was never indexed. The chunker only extracted functions, classes, and methods. Module docstrings are the best text to match conceptual queries about what a file does.

## Decision

Extract module-level docstrings as `ChunkType.module` chunks.

**Detection algorithm:**
1. Iterate `root.children` of the tree-sitter AST
2. Skip `comment` and `newline` nodes
3. First real child must be `expression_statement` containing a `string` child
4. If found: emit a MODULE chunk; if not: no module docstring exists

**PEP 257 compliance:** A module docstring must be the first statement in the file. Comments and blank lines before it are allowed, but any other statement (import, assignment) means there is no module docstring. A string literal appearing after an import is **not** a module docstring.

**Chunk properties:**
- `chunk_type`: `ChunkType.module`
- `name`: file stem (e.g., `container` for `container.py`)
- `content`: raw source including triple-quote delimiters (consistent with other chunk types)
- `line_start`/`line_end`: 1-indexed line range of the docstring expression

## Alternatives Considered

**Index entire file as a chunk:** Would dilute embedding quality with imports and boilerplate. Module docstrings are already a curated summary.

**Strip triple-quote delimiters from content:** Would be inconsistent with how function/class chunks store raw source. The embedding model handles them fine.

**Separate docstring index:** More complex, planned as a separate Tier 3 item. Module docstrings are a simpler, higher-value first step.

## Consequences

- Files with module docstrings now produce one additional MODULE chunk
- Conceptual queries ("dependency injection", "HTTP client", "configuration") will match files by their self-description
- Existing tests unaffected: no fixtures started with module docstrings, integration tests use relative assertions
- `sample_project` fixture files with docstrings produce extra MODULE chunks, but assertions are `> 0` not exact counts
