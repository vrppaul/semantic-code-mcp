# TODO

## In Progress

### Semantic Code Search MVP
We need semantic code search because grep requires knowing exact terms. When exploring unfamiliar codebases or searching for concepts ("function that handles authentication", "error handling for API calls"), grep fails because we don't know the exact variable/function names used.

Semantic search lets us find code by meaning, not just text matching. This dramatically speeds up code exploration and reduces the iterative grep→read→refine cycle.

**Scope**: Python support only, single-machine local operation, MCP integration with Claude Code.

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

### Status Check Caching
The `status_check_ms` is currently ~500-850ms because we scan all files and check mtimes on every search. Options:
- Cache the file list and mtimes in memory, refresh periodically
- Use file system watchers (watchdog) to detect changes
- Add a "skip status check" option for repeated queries in same session
- Store last-checked timestamp and only rescan if > N seconds old
