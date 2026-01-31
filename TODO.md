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
- [ ] Module-level code - extend chunker to capture top-level statements
- [ ] Background re-indexing - return stale results while re-indexing
- [ ] Separate docstrings - index docstrings separately for better matching
- [ ] Code-specific embedding model - evaluate UniXcoder/CodeBERT

## Pending

### Code Quality & Architecture Cleanup
Post-DI cleanup pass. Improve consistency, type safety, and modularity.

**Type safety:**
- [x] Add `ty` to pre-commit hooks
- [x] Fix `ty` diagnostics (embedder.py return type, lancedb.py suppression, chunker.py null guards)
- [ ] Add type hints to tree-sitter `node` params in chunker.py (4 methods)

**Error handling:**
- [ ] Replace broad `except Exception` in chunker.py:57 with specific tree-sitter exceptions
- [ ] Replace broad `except Exception` in lancedb.py:78 (`_ensure_fts_index`) with specific exceptions
- [ ] Review lancedb.py overall — `search_hybrid()` is 76 lines, merge logic could be extracted

### Multi-language Support
Currently Python only. Need JS/TS for web projects, Rust/Go for systems work. Tree-sitter supports all of these, so it's mainly about adding language-specific chunking logic.

### Performance Optimization
Profiling infrastructure added (pyinstrument). Use `SEMANTIC_CODE_MCP_PROFILE=1` to generate profiles.

**Completed:**
- [x] FTS index skip - avoid rebuilding if already exists (~80ms saved per search)
- [x] Batch embedding generation (already implemented)

**Remaining:**
- [ ] LanceDB index tuning (IVF partitions, PQ compression)

### Reduce Install Size (CPU-only PyTorch)
Current .venv is ~7.8GB, mostly due to PyTorch with CUDA (nvidia: 4.3GB, torch: 1.8GB, triton: 643MB). Users installing via `uvx` don't need GPU support for a code search tool - CPU inference is sufficient for short queries.

**Solution:** Configure uv to use PyTorch CPU-only index:
```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

**Expected reduction:** ~6.8GB → ~200MB for torch portion. Total install ~1GB instead of ~8GB.

**Alternative considered:** FastEmbed (ONNX-based) - even smaller but embeddings may differ from sentence-transformers, requiring index rebuilds and potentially different search quality.

## Done

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
