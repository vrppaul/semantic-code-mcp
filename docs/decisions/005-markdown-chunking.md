# 005: Markdown Chunking

**Status**: implemented
**Date**: 2026-02-01

## Context

The semantic search index only covers code files (Python, Rust). Projects also contain significant knowledge in markdown files — READMEs, docs, decision records, changelogs, rule files. When an agent searches for "how authentication works," relevant documentation doesn't surface because `.md` files aren't indexed.

`tree-sitter-markdown` (v0.5.1) is available on PyPI and follows the same `language()` pattern as the existing tree-sitter grammars.

### Spike findings (tree-sitter-markdown AST)

The AST uses **nested `section` nodes** rather than flat heading siblings:

```
document
  section                    # H1 scope
    atx_heading
      atx_h1_marker          # "#"
      inline                 # heading text (without markers)
    paragraph                # content under H1
    section                  # H2 nested inside H1
      atx_heading
        atx_h2_marker
        inline
      paragraph
  section                    # another H1 scope
    ...
```

Key properties:
- `section` nodes **nest by heading level** — H2 sections are children of H1 sections
- Pre-heading content gets its own `section` (no `atx_heading` child)
- No-heading documents: single `section` wrapping all content
- Empty documents: zero children
- `section` nodes carry correct line ranges spanning their full content

## Decision

### Chunking strategy: recursive section walking

Walk `section` nodes recursively. Each `section` that contains an `atx_heading` becomes a `ChunkType.section` chunk. Each `section` without a heading (preamble) becomes a `ChunkType.module` chunk.

**Flatten nested sections.** A document with `# H1`, `## H2a`, `## H2b` produces three separate chunks, not one big H1 chunk containing H2s. This keeps chunks small and focused — better for embedding similarity search.

For each section node:
1. Find its `atx_heading` child (if any) to determine the chunk name.
2. Extract the heading text from the `inline` child of the heading node.
3. Collect the section's own non-section content (paragraphs, code blocks, lists, etc.) as the chunk content — excluding nested sub-sections.
4. Emit a chunk spanning from the section's heading to the end of its direct content (before first sub-section, or end of section if no sub-sections).
5. Recurse into child `section` nodes.

### ChunkType extension

Add `section = auto()` to the `ChunkType` enum. Storage is string-based (LanceDB UTF-8), so fully backward-compatible. Existing indexes simply won't have rows with this value.

Using a new type rather than mapping to existing code types (`module`/`function`) because markdown sections are semantically different from code constructs and consumers may want to filter by type.

Exception: preamble content (before first heading) uses `ChunkType.module`, consistent with how Python/Rust chunkers use `module` for file-level docstrings.

### Chunk naming

- Section with heading: name = heading text (e.g., "Installation", "API Reference")
- Preamble without heading: name = file stem (e.g., "README", "CHANGELOG")
- Document with no headings: name = file stem, type = module

## Alternatives Considered

### A: Flat sibling iteration (tracking "current section" state)

The initial assumption before the spike. Would iterate document children, tracking heading boundaries manually. Rejected because the AST already provides `section` nodes with correct boundaries — reimplementing this logic would be redundant and error-prone.

### B: Map to existing ChunkTypes (headings as `module`, code blocks as `function`)

Avoids adding a new enum value. Rejected because it's misleading — a markdown heading isn't a module, and search result consumers may filter or display based on chunk type.

### C: Extract fenced code blocks as separate chunks

Would give code examples within docs their own chunks for better code-specific search. Rejected for now — adds complexity, fragments surrounding context the embedding model needs, and can be added later if search quality for in-doc code is poor.

## Consequences

- New `ChunkType.section` value — any code that exhaustively matches on `ChunkType` needs updating (check with grep)
- `tree-sitter-markdown` added as dependency (~small, pure wheel)
- `.md` files will be indexed on next `index_codebase` call — existing indexes untouched until re-indexed
- The `test_composite_chunker.py` test using `.md` as "unknown extension" needs updating
