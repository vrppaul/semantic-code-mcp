# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- Tree-sitter based AST chunking for Python (functions, classes, methods)
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
