# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed
- The ranking boost of files edited within the last week
- Settings that nothing read: `SEMANTIC_CODE_MCP_EMBEDDING_DEVICE`, `SEMANTIC_CODE_MCP_CHUNKING_TARGET_TOKENS`, `SEMANTIC_CODE_MCP_CHUNKING_MAX_TOKENS`
- The `pandas` dependency

### Added
- `index_status` reports the index's `manifest` (model, its revision, chunker version, storage schema version)
- Each `search_code` result names its `parent`: the class, impl, module or heading it sits in, as `Stack` for `Stack.push` and `Guide > Install` for a section
- `search_code` takes `include_debug`; the `debug` block comes back only when asked for. It now holds `timings`, `stats` and `indexing` (what was indexed before the search); `index_status` and `index_result` left it
- `index_status` reports `stale_files_count`
- `search_code` takes `chunk_kinds`, `languages`, `path_glob` and `exclude_glob`, applied before the best hits are taken, so a narrow search still fills its results (#3)
- `SEMANTIC_CODE_MCP_GIT_TIMEOUT_SECONDS`
- `SEMANTIC_CODE_MCP_MIN_SIMILARITY` (0.35) and `SEMANTIC_CODE_MCP_MIN_SIMILARITY_WHEN_WORDS_MATCH` (0.2): how close in meaning to the query a hit must be to stay, without and with its words matching the query's
- `python -m evals quality` asks eight questions no corpus answers and reports what they get
- `LICENSE` (MIT) and license, author and project URLs in the package metadata; the wheel used to ship without a license
- CI runs every pre-commit gate, the tests and a build on each push to `master` and each pull request
- `CONTRIBUTING.md`, `SECURITY.md` and a pull request template
- `python -m evals quality` and `python -m evals speed` measure search quality and speed on three pinned corpora through the server's tools, against a saved baseline
- End-to-end tests drive the three MCP tools through a client

### Changed
- Requires mcp 2.x: the server is an `MCPServer`, its state lives in the lifespan instead of module globals
- A tool that fails answers as an MCP error (`isError`) with its message, instead of a successful result holding `{"error": ...}`
- Each tool answers one model, without FastMCP's `{"result": ...}` wrapper
- `search_code` refuses a `limit` above 50, and every tool refuses a relative `project_path`
- A `search_code` result's `chunk_type` is now `kind`
- A directory inside a git repository is part of the repository's project: the tools index and report the whole repository once, and a search asked from the directory keeps to it, instead of building a second index from scratch for each directory. A directory of a large repository now costs the whole repository's first index, with no way to keep to the directory alone; `.semantic-code/` lies at the repository's root; indexes 0.4 built for subdirectories are left behind. A directory stays its own project outside a work tree, where the repository ignores it, or where a repository that hides untracked files (a home directory under git for dotfiles) tracks nothing under it
- `search_code` ranks by Reciprocal Rank Fusion of what the vector and the keyword searches found, in one version of the index, instead of a weighted sum of their scores: the right file's mean reciprocal rank went from 0.455 / 0.549 / 0.698 to 0.497 / 0.502 / 0.785 on click / fd / this repository
- The keyword search finds a chunk by the words of its name (`getUserName` by "user name", `HTTPServer` by "server") and by the words of what it is part of: a method by its class's, a section by its headings above
- A section of a document ranks below code that the search finds alike, its fused score weighed by 0.9: on the harness the code questions' file MRR rose from 0.539 / 0.698 / 0.828 to 0.700 / 0.754 / 0.828 and the documentation questions' fell from 1.0 / 1.0 / 0.875 to 0.8 / 1.0 / 0.635 on click / fd / this repository, measured with the search text's first build and to be measured again (#3)
- A search drops a hit below a cosine similarity of 0.35 to the query, or of 0.2 when its words match the query's, instead of below a blended score of 0.3: of eight questions far from any corpus, one result came back on one corpus
- A `search_code` result's `score` is now `similarity`, the cosine similarity of its chunk to the query; results come in the fused order, no longer grouped by file: the right file's mean reciprocal rank went from 0.497 / 0.502 / 0.785 to 0.539 / 0.585 / 0.825
- The index keeps each file by its path from the project root, so a project moved with its local index (`.semantic-code/`) keeps it
- The tools no longer send log notifications (`Searching for: ...`, `Found N results in X ms`); progress notifications remain, and now only move forward, the model load included
- Progress moves while the chunks embed, 256 at a time, instead of standing at 70% until the whole embedding is done
- The index keeps a record of each file and a manifest of what built it, in place of `file_mtimes.json`
- The default model loads at a pinned revision, not whichever the hub's `main` holds when it is fetched. A cache that holds only an older revision fetches the pinned one's small files once; with `HF_HUB_OFFLINE=1` the model does not load until it has
- Each model and its revision, chunker version and storage layout keeps an index of its own beside the others in the project's index directory: two sessions of different versions or models never build each other's index again, and switching back finds the old one. The first search after the upgrade builds a new index; one that no session read or wrote for a week is removed, 0.4's included, but never while a writer holds it
- An index run writes a slice of files at a time: a forced re-index keeps every file searchable throughout, instead of emptying the index first, and a first index stopped midway keeps the files it wrote
- An edit re-embeds only the chunks whose text it changed; a forced re-index embeds every chunk again
- A write no longer rebuilds the full-text index; the index is compacted, its new rows indexed and old versions pruned once twenty versions have gathered, and at once after an index run of many writes or a forced one
- `index_status` of a project never indexed no longer creates its index directory
- No `.env` file is read: the server runs in the user's project, whose `.env` is not its to read
- Profiles (`SEMANTIC_CODE_MCP_PROFILE`) are written under the cache directory, not `./profiles` in the user's project
- A release runs the whole CI suite on the tagged commit before publishing to PyPI
- `ty` fails on any warning; the hook used to end in `|| true`
- bandit 1.9.4 replaces 1.8.6, which crashed on every Python 3.14 file and passed
- The ruff hook runs the locked ruff instead of an older mirror
- New hooks: deptry, typos, a check that every test module loads, a check of the agent instruction files
- `CLAUDE.md` is a symlink to `AGENTS.md`, rewritten on the template of the owner's other projects; ADR 002 is superseded
- File discovery asks git for tracked files and for untracked files that `.gitignore` does not exclude; `use_gitignore=false` walks the file system instead, for repos whose `.gitignore` hides source on purpose (#2, #3)
- `ignore_patterns` apply in git mode too. A pattern without a slash matches a path component at any depth (`node_modules`); a pattern with a slash matches from the project root (`build/**`); a leading `/` anchors at the root, `./` and a trailing `/` are dropped, as in `.gitignore`
- The default `ignore_patterns` also skip `.mypy_cache`, `.ruff_cache`, `.tox`, `.cache`, `site-packages`, `.semantic-code` and top-level `target/`, `dist/`, `build/`
- Files inside git submodules are indexed
- Outside a git work tree, `.gitignore` is no longer parsed by hand; only `ignore_patterns` apply
- A Python or Rust file with no definitions — nothing, or nothing but a module docstring — is indexed whole, in `module` chunks of up to 100 lines, instead of not at all or by its docstring alone
- Files over 1 MB and binary files (a NUL among the first 8000 bytes, which includes UTF-16) are skipped with a warning
- Rust: required trait methods (`fn save(&self);`) become method chunks, `/** */` doc comments attach to the item below, `/*! */` counts as a module doc, and the module doc may follow inner attributes (`#![...]`)
- Rust: impls are named with their trait's type arguments and negation, as in `From<u8> for Level` and `!Send for Handle`

### Fixed
- `index_status` loaded the embedding model it never uses, blocking for seconds
- Two index runs of one project at once (parallel tool calls) stored every changed file's chunks twice
- A status or search during a forced re-index could fail on a dropped table; it still can at the moment a model replaced under a name nothing pins (a local path) changes the vectors' width
- A model of another dimension under a name nothing pins (a local path) crashed every index run; the index now starts again from that run's files, and the next run indexes the rest
- A failure inside a tool reached the agent as `Error executing tool ...` without its cause
- With `local_index`, two spellings of one project path got separate locks and stores
- A search of a project with nothing indexed loaded the model to embed its query
- Index status and change detection ran on the event loop
- Server failed to start on a fresh install: mcp 2.x renamed `FastMCP` to `MCPServer`; the server now runs on mcp 2.x (#3)
- Untracked files were never indexed (#2, #3)
- A tracked file deleted from disk was re-indexed on every search
- A file deleted from disk kept answering searches until an index run: the status that a search checks first did not count it as stale
- An edit made while its file was being indexed was lost: the file's modification time was read after indexing, not when it was scanned
- Two server processes indexing one project at once (two Claude Code sessions) could store its chunks twice: a write now holds a lock file in the index directory
- The index grew with every edit: old versions were never pruned
- A project-local index (`.semantic-code/`) showed as untracked in the project's `git status`
- A file name that is not valid UTF-8 failed every index run; such a file is now skipped with a warning
- Files with non-ASCII names were listed in git's quoted form and never found
- Linked worktrees, submodules and subdirectories of a repo were walked without `.gitignore`
- A conflicted file was listed once per conflict stage; a file reached through a symlink inside the project was indexed twice
- `GIT_DIR` or `GIT_WORK_TREE` in the server's environment pointed discovery at another repository
- A missing `git` binary or a git timeout raised instead of falling back to the walk
- One file that is not valid UTF-8 aborted indexing of the whole project, on every search; such bytes are now replaced
- A tracked symlink could pull a file from outside the project into the index
- Python: a comment between a decorator and `def`/`class` dropped the definition
- Rust: generic and path-qualified `impl` blocks (`impl<T> Stack<T>`, `impl fmt::Display for Stack<i32>`) produced no chunks, and trait impls were named after the trait alone
- Rust: items inside inline `mod` blocks, including `#[cfg(test)] mod tests`, were never indexed
- Markdown: the last section's `line_end` ran one line past the end of the file

## [0.4.0] - 2026-02-01

### Added
- Markdown support — indexes `.md` files by heading-based sections (`ChunkType.section`)
- Setext heading support (`Title\n=====` syntax) in addition to ATX headings (`# Title`)
- `ChunkType.section` enum value for document sections (backward-compatible with existing indexes)
- YAML frontmatter correctly excluded from chunks
- Decision doc: `docs/decisions/005-markdown-chunking.md`

### Changed
- `tree-sitter-markdown` added as dependency
- `CompositeChunker` now routes `.md` files to `MarkdownChunker`

## [0.3.0] - 2026-01-31

### Added
- Multi-language chunker architecture (`BaseTreeSitterChunker` + `CompositeChunker` dispatcher)
- Rust support — functions, structs, enums, traits, impl blocks, `//!` module doc comments
- Published to PyPI — installable via `uvx semantic-code-mcp`
- GitHub Actions workflow for automated publishing on tag push (trusted publishers OIDC)
- Platform-specific install docs (macOS/Windows vs Linux CPU-only torch)
- "Adding a New Language" guide in README

### Changed
- Flattened package — `chunkers/` and `embedder.py` at top level, `indexer/` collapsed to single module
- `IndexService` owns scanning, change detection, chunking, status; `Indexer` handles embedding and storage only
- `CompositeChunker` with extension collision detection (renamed from `MultiLanguageChunker`)
- `IndexService` takes `CompositeChunker` directly — no separate `supported_extensions` parameter
- All enums standardized to `StrEnum` + `auto()` — `ChunkType`, Python `NodeType`, Rust `NodeType`
- `__init__.py` files are docstring-only (no re-exports); all imports use full module paths
- Container uses `cached_property` for model/embedder singletons, per-project store caching

### Fixed
- Tree-sitter `Parser` thread-safety — create fresh parser per `chunk_string` call (was shared, mutates on `parse()`)
- `mock_embedder.embed_batch` returns correct embedding count via `side_effect` (was hardcoded to 1)
- `Indexer` is now pure data pipeline — no `Settings`, no `FileChangeCache`, no `cache_dir`; all cache bookkeeping owned by `IndexService`
- Clean shutdown on Ctrl+C (SIGINT handler instead of traceback)
- Removed redundant `.value` calls on `StrEnum` members

## [0.2.0] - 2026-01-31

### Changed
- README: configuration examples for environment variables, the release process and the project structure

## [0.1.1] - 2026-01-31

The first release on PyPI. No 0.1.0 was ever tagged; until 2026-10-25 this entry was headed 0.1.0.

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
