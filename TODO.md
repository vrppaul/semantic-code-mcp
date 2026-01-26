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

### Convert to Dependency Injection
Currently classes instantiate their dependencies directly (e.g., Searcher creates Embedder and Indexer internally). This makes testing harder and couples classes tightly. Refactor to inject dependencies:
- Pass Embedder, Indexer, VectorStore as constructor parameters
- Use a factory or container for wiring in production
- Allows easier mocking in tests

### Multi-language Support
Currently Python only. Need JS/TS for web projects, Rust/Go for systems work. Tree-sitter supports all of these, so it's mainly about adding language-specific chunking logic.

### Performance Optimization
After MVP works, profile and optimize:
- Batch embedding generation
- LanceDB index tuning (IVF partitions, PQ compression)
- Incremental indexing efficiency
