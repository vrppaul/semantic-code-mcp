# 004: Multi-Language Chunker Architecture

**Status**: accepted
**Date**: 2026-01-31

## Context

The chunker only supports Python. JS/TS is needed for web projects, Go/Rust for systems work. Tree-sitter has grammars for all of these, so the question is how to structure multi-language support cleanly.

## Decision

**Hybrid base class + dispatcher pattern** (Option C + D).

### Architecture

```
MultiLanguageChunker (implements ChunkerProtocol)
  ├── dispatches by file extension to:
  ├── PythonChunker(BaseTreeSitterChunker)
  ├── JavaScriptChunker(BaseTreeSitterChunker)
  ├── TypeScriptChunker(BaseTreeSitterChunker)
  ├── GoChunker(BaseTreeSitterChunker)
  └── RustChunker(BaseTreeSitterChunker)
```

**`BaseTreeSitterChunker`** provides shared logic:
- `chunk_file(file_path)` — reads file, calls `chunk_string`
- `chunk_string(code, file_path)` — parses with tree-sitter, calls abstract `_extract_chunks`
- `_make_chunk(node, file_path, lines, chunk_type, name)` — line extraction and Chunk creation
- Parser creation from a `Language` object

**Each subclass** provides only:
- The tree-sitter `Language` to use
- `_extract_chunks(root, file_path, lines) -> list[Chunk]` — language-specific AST walking
- Supported file extensions (class attribute)

**`MultiLanguageChunker`** dispatches `chunk_file` by file extension. Single `ChunkerProtocol` interface for the rest of the system.

### System changes required

- `Indexer.scan_files()` — accept supported extensions instead of hardcoding `*.py`
- `Container.create_chunker()` — return `MultiLanguageChunker`
- Each language needs a `tree-sitter-<lang>` dependency

### What stays unchanged

`ChunkerProtocol`, `Indexer` pipeline, services, storage, `Chunk` model — all unchanged.

## Alternatives Considered

### Option A: One class per language (no shared base)

Maximum flexibility but duplicates ~40% of code (file reading, parsing, line slicing, Chunk construction). Adding a language means copying boilerplate.

### Option B: Generic chunker with declarative config

A single `TreeSitterChunker` with a `LanguageConfig` mapping node types. Works for simple languages but breaks down for Go (receiver methods at package level, not nested in structs), Rust (`impl` blocks), and JS (arrow-function-in-const patterns). Needs escape-hatch callbacks that effectively degrade into Option C with extra indirection.

## Consequences

### Why structural differences prevent a pure config approach

| Pattern | Python | Go | Rust | JS |
|---------|--------|----|------|----|
| Methods | Nested in class body | Receiver functions at package level | Functions inside `impl` blocks | `method_definition` in class body |
| Decorators | `decorated_definition` wrapper node | None | `attribute_item` preceding item | `decorator` child nodes |
| Module docs | PEP 257 docstring | Comment before `package` | `//!` inner doc comments | JSDoc comment at top |
| Classes | `class_definition` | None (structs + interfaces) | None (structs + enums + traits) | `class_declaration` |

Each language needs 50-100 lines of specific logic — enough to warrant real code, not enough to justify full duplication.

### ChunkType mapping

Current enum (`MODULE`, `FUNCTION`, `CLASS`, `METHOD`) maps well across languages. Go structs and Rust structs map to `CLASS`. Go interfaces and Rust traits could use an optional `INTERFACE` type added later. No need to model every language construct — the embedding model works on content, not type labels.
