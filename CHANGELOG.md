# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Multi-language chunker architecture (`BaseTreeSitterChunker` + `MultiLanguageChunker` dispatcher)
- Rust support — functions, structs, enums, traits, impl blocks, `//!` module doc comments
- Published to PyPI — installable via `uvx semantic-code-mcp`
- GitHub Actions workflow for automated publishing on tag push (trusted publishers OIDC)
- Platform-specific install docs (macOS/Windows vs Linux CPU-only torch)

### Changed
- Flattened package — `chunkers/` and `embedder.py` at top level, `indexer/` collapsed to single module
- `IndexService` owns scanning, change detection, chunking, status; `Indexer` handles embedding and storage only
- `MultiLanguageChunker` renamed to `CompositeChunker` with extension collision detection
- `__init__.py` files are docstring-only (no re-exports); all imports use full module paths
- `Indexer.scan_files()` accepts dynamic file extensions via `supported_extensions` (no longer hardcoded `*.py`)

### Fixed
- Tree-sitter `Parser` thread-safety — create fresh parser per `chunk_string` call (was shared, mutates on `parse()`)
- `mock_embedder.embed_batch` returns correct embedding count via `side_effect` (was hardcoded to 1)
- `Indexer` is now pure data pipeline — no `Settings`, no `FileChangeCache`, no `cache_dir`; all cache bookkeeping owned by `IndexService`
- Clean shutdown on Ctrl+C (SIGINT handler instead of traceback)

## [0.1.0] - 2026-01-31

### Added
- `IndexService` orchestrating full index pipeline (scan → detect → chunk → embed) with progress callbacks
- `SearchService` with auto-indexing via `IndexService` (replaces manual orchestration in server.py)
- `services/` package replacing `search/` directory
- `duration_seconds` on `IndexResult` for end-to-end timing
- Strict ruff rules: C901, DTZ, ASYNC, SLF, PIE, T20, PERF, FURB, PLC0415
- ty type-checking rules in pre-commit and pyproject.toml
- Profiling support with pyinstrument for dev (enable with `SEMANTIC_CODE_MCP_PROFILE=1`)
- MCP server with three tools: `search_code`, `index_codebase`, `index_status`
- Semantic code search using sentence-transformers embeddings (all-MiniLM-L6-v2)
- LanceDB vector storage for embeddings
- Module-level docstring chunking — conceptual queries now match files by their self-description (decision 003)
- Tree-sitter based AST chunking for Python (functions, classes, methods, module docstrings)
- Incremental indexing with mtime-based change detection
- Debug timing info in search results (status_check_ms, embedding_ms, search_ms)
- Hybrid search: keyword boost (up to 20%) and recency boost (up to 5% for files < 1 week old)
- Score threshold filtering (< 0.3 filtered as noise)
- Result truncation (> 50 lines shows "... truncated")
- Results grouped by file for cleaner output
- Pre-load embedding model at startup (avoids 2s cold start)
- Parallel file chunking with asyncio.gather
- Project documentation structure (CLAUDE.md, README.md, TODO.md, CHANGELOG.md)
- Claude Code rules in `.claude/rules/`
- Architecture decision records in `docs/decisions/`
- Pre-commit hooks (ruff, bandit, conventional commits)

### Fixed
- SQL injection in `delete_by_file` — escape single quotes in file paths
- Force reindex now clears vector store to prevent duplicate results
- Atomic cache writes (tempfile + rename) to prevent corruption on crash
- FTS index failures logged at WARNING instead of DEBUG
- Timezone-aware datetimes throughout (DTZ compliance)

### Changed
- CPU-only PyTorch via `[tool.uv.sources]` — venv reduced from 7.8GB to 1.7GB (no CUDA/nvidia/triton)
- Lazy `sentence-transformers` import — startup no longer loads torch (~4s saved)
- `server.py` is now a thin tool layer delegating to `IndexService`/`SearchService`
- `SearchOutcome.index_result` always present (default zeros) — eliminates None guards
- `container.create_search_service()` reuses indexer instead of creating duplicates
- Removed unused `status_cache_ttl` config option
- Chunker complexity reduced by extracting `_extract_decorated` and `_extract_class_with_methods`
- File scanning uses `git ls-files` for 100x speedup (falls back to os.walk for non-git repos)
- Removed sync `index()` method - only async version remains (no code duplication)
- Removed unused Searcher class
- Skip FTS index rebuild if already exists (~80ms saved per search)
