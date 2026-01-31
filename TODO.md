# TODO

## In Progress

### Search Quality & Output Improvements
Improve search result quality and output format for better usability.

**Tier 1 - Quick Wins:**
- [x] Score threshold filtering - filter out low-confidence results (score < 0.3)
- [x] Remove unused Searcher class - dead code in search/searcher.py
- [x] Truncate long results - cap at ~50 lines with "..." indicator
- [x] Group results by file - sort results so same-file chunks are together
- [ ] Configurable min_score parameter - let callers control the quality threshold (default 0.3)
- [x] Stronger exact match boost - exact phrase/keyword matches should get +30-50% boost, not just +5%
- [x] Hybrid search with keyword fallback - run semantic + keyword search in parallel, merge results. Ensures exact identifier matches surface even with low semantic similarity

**Tier 2 - Medium Effort:**
- [x] Keyword boost (hybrid search) - boost results containing query words literally
- [x] Parallel chunking - use asyncio.gather for parallel file processing
- [x] File recency boost - factor mtime into ranking

**Tier 3 - Larger Effort:**
- [x] Module-level code - extend chunker to capture module docstrings (see decision 003)
- [ ] Background re-indexing - return stale results while re-indexing
- [ ] Separate docstrings - index docstrings separately for better matching
- [ ] Code-specific embedding model - evaluate UniXcoder/CodeBERT

## Pending

### Multi-language Support
Currently Python only. Need JS/TS for web projects, Rust/Go for systems work. Tree-sitter supports all of these. See decision 004 for architecture analysis.

**Architecture (decided):** Hybrid base class + dispatcher pattern (Option C+D from analysis).
- `BaseTreeSitterChunker` — shared logic (parsing, line extraction, Chunk construction)
- Language-specific subclasses (`PythonChunker`, `GoChunker`, etc.) — AST walking rules only
- `MultiLanguageChunker` — dispatcher by file extension, single `ChunkerProtocol` interface

**Implementation steps:**
- [ ] Refactor `PythonChunker` into `BaseTreeSitterChunker` + `PythonChunker` subclass
- [ ] Add `MultiLanguageChunker` dispatcher
- [ ] Update `Indexer.scan_files()` to accept supported extensions (currently hardcodes `*.py`)
- [ ] Wire `MultiLanguageChunker` in container
- [ ] Add JavaScript/TypeScript chunker (`tree-sitter-javascript`, `tree-sitter-typescript`)
- [ ] Add Go chunker (`tree-sitter-go`) — receiver methods, package comments
- [ ] Add Rust chunker (`tree-sitter-rust`) — impl blocks, `//!` doc comments

### Performance Optimization
Profiling infrastructure added (pyinstrument). Use `SEMANTIC_CODE_MCP_PROFILE=1` to generate profiles.

**Completed:**
- [x] FTS index skip - avoid rebuilding if already exists (~80ms saved per search)
- [x] Batch embedding generation (already implemented)

**Remaining:**
- [ ] LanceDB index tuning (IVF partitions, PQ compression)

## Done

### Reduce Install Size (CPU-only PyTorch)
Configured uv to pull torch from CPU-only PyTorch index. Venv reduced from 7.8GB to 1.7GB (78% smaller). No CUDA/nvidia/triton packages installed.

### Code Quality & Architecture Cleanup
Post-DI cleanup pass. Improved consistency, type safety, and modularity.
- Specific exception types in chunker.py (`OSError`, `ValueError`, `UnicodeDecodeError`) and lancedb.py (`OSError`, `ValueError`, `RuntimeError`)
- `search_hybrid()` split into 30-line method + extracted `_merge_results()`
- Tree-sitter `Node` type hints already present on all chunker methods
- `ty` in pre-commit, all diagnostics fixed

### Services Layer & Strict Linting
Extracted `IndexService` and `SearchService`, architecture review fixes, strict lint config.
- `IndexService` orchestrates scan → detect → chunk → embed with progress + timing
- `SearchService` auto-indexes via `IndexService`, non-optional `SearchOutcome.index_result`
- `server.py` reduced to thin tool layer delegating to services
- SQL injection fix, atomic cache writes, FTS log levels
- Lazy `sentence-transformers` import (torch no longer loaded at startup, ~4s saved)
- Strict ruff rules (C901, DTZ, ASYNC, SLF, PIE, T20, PERF, FURB, PLC0415)
- ty type-checker added to pre-commit with targeted rules

### Dependency Injection Refactor
Converted from direct instantiation to proper DI with a composition root.
- Container with lazy model loading and per-project connection caching
- Global settings singleton (`configure_settings()` / `get_settings()`)
- `app.py` composition root (`create_app()`) wiring settings, logging, profiling, container
- `server.py` stripped to pure tool definitions — no init code
- Shared store/embedder between SearchService and Indexer (no redundant instances)
- `cache_dir` injected into Indexer (no internal `get_index_path` calls)
- Split `models.py` into domain models + `responses.py` for API types
- Real timing data in `SearchOutcome` instead of hardcoded zeros
- `_ensure_table()` made public, dead ranking code removed
- SearchService tests added (10 tests)
