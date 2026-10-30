# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed
- A `search_code` result's score: by words a hit has no similarity, and one number on two scales misled; the order is the ranking
- The ranking boost of files edited within the last week
- Settings that nothing read: `SEMANTIC_CODE_MCP_EMBEDDING_DEVICE`, `SEMANTIC_CODE_MCP_CHUNKING_TARGET_TOKENS`, `SEMANTIC_CODE_MCP_CHUNKING_MAX_TOKENS`
- The `pandas` dependency
- sentence-transformers and torch: the embedder runs the model's ONNX export on onnxruntime, so the install no longer needs the `--index pytorch-cpu=...` flag on Linux

### Added
- `SEMANTIC_CODE_MCP_SEARCH_MODE=words_and_meaning` searches by meaning too: the model's vectors searched and fused with the words by rank, the chunks without vectors embedded first, inside the call (the first search of a repository takes minutes at 20k chunks and about an hour at 90k). The tool offers no switch of its own: in 480 sessions of Claude Code on two large repositories the model's meaning fused with the words gained agents nothing, and when offered they took it for a first search, which on a repository not yet embedded would wait for all of it (ADR 008, pilot 9)
- `index_status` reports `embedded_chunks_count`, the chunks with the server's model's vectors, and that `model` (its name, revision and embedder version); the index's `manifest` (the chunker and storage schema versions that built it); and `being_built`, whether a run, this session's or another's, holds the index as it answers, where during another session's first build it used to report a project not indexed and say nothing more
- Each `search_code` result names its `parent`: the class, impl, module or heading it sits in, as `Stack` for `Stack.push` and `Guide > Install` for a section
- `search_code` takes `include_debug`; the `debug` block comes back only when asked for. It now holds `timings`, `stats` and `indexing` (what was indexed before the search); `index_status` and `index_result` left it
- `index_status` reports `stale_files_count`
- `search_code` names the `project_root` whose index answered and the `searched_directory` it kept to, so a search kept to a subdirectory, as a session started there keeps every search, says so
- `index_status` and `index_codebase` name the `project_root` whose index they report: a directory inside a git repository reports the whole repository's
- The server sends instructions a client shows beside the tools: how to word a search and when to Grep instead, what `project_path` names, the languages indexed, and that indexing is automatic
- A skill, `skills/semantic-search`, that tells an agent which tool to use, how to phrase a query, how to narrow a search and read its answer, and the traps met so far
- A Claude Code plugin that installs the server and the skill together: `/plugin install semantic-code --marketplace vrppaul/semantic-code-mcp`
- `project_path` may be left out of every tool: the tool then works on the client's workspace root, the directory Claude Code was started in, where that lies in a git repository whose root does not hold the home directory; a client that names no root or several, or a root in no such repository (a folder of projects, the home directory), gets a tool error asking for `project_path`
- `search_code` takes `chunk_kinds`, `languages`, `path_glob` and `exclude_glob`, applied before the best hits are taken, so a narrow search still fills its results (#3). A glob reads as in a `.gitignore`, from the directory searched, without character classes; one holding a class, braces, an escape, a leading `!` or `..`, or of more than 1000 characters, is refused with the reason
- The model's libraries (onnxruntime, tokenizers, huggingface_hub) are the extra `semantic-code-mcp[meaning]`, needed only for a search by meaning: the install is 95 MB instead of 190, a server by words imports none of them (24 MB less and its imports about 135 ms sooner at the median), and a server set to search by meaning without the extra stops before it runs, with one line naming it
- `SEMANTIC_CODE_MCP_INDEX_STORE` picks what keeps the indexes: `sqlite`, the default, or `lancedb`, whose libraries come with the extra `semantic-code-mcp[lancedb]`; each keeps indexes of its own
- `SEMANTIC_CODE_MCP_GIT_TIMEOUT_SECONDS`
- `SEMANTIC_CODE_MCP_MIN_SIMILARITY` (0.35) and `SEMANTIC_CODE_MCP_MIN_SIMILARITY_WHEN_WORDS_MATCH` (0.25): how close in meaning to the query a hit must be to stay, without and with its words matching the query's
- `python -m evals quality` scores code and documentation questions apart, asks a sample of bare names, held-out questions, and questions no corpus answers, of other fields and of programming, and reports what they get
- `LICENSE` (MIT) and license, author and project URLs in the package metadata; the wheel used to ship without a license
- CI runs every pre-commit gate, the tests and a build on each push to `master` and each pull request
- `CONTRIBUTING.md`, `SECURITY.md` and a pull request template
- `python -m evals quality` and `python -m evals speed` measure search quality and speed on three pinned corpora through the server's tools, against a saved baseline
- `python -m evals speed` reports the memory of the server and of every process it starts, each one's proportional share summed and read every 50 ms while the server indexes or loads its model, so a first index's pool counts
- `python -m evals speed --corpus-file evals/generated/sentry-shaped.toml` times a repository of Sentry's shape made where it runs, 8,500 modules in 3,054 directories, as on CI's runners, where none so large can be cloned; `--scan` sets its servers to scan the files at each search, the system's watcher left off, to set beside a run that follows them; and every speed run times a search right after a burst of 10,000 files where the index never looks, as a build writes under `target/` (`--burst-files`). CI's `Speed` workflow runs both on Linux, macOS and Windows
- `python -m evals speed --words-only` leaves out the server set to search by meaning, whose first search embeds every chunk (an hour at Sentry's size), so a large corpus is timed by words in about a minute; the memory of that server is reported beside its speeds
- `python -m evals agents` asks Claude Code headless each question of the corpora it names, in arms named by what the agent is given (its own tools, or this server's search in a mode), and compares the arms question by question in turns and prompt tokens; every session is kept with its stream under the run's name, and `python -m evals rescore` scores a kept run again once a judge has settled the answers no key names
- End-to-end tests drive the three MCP tools through a client

### Changed
- `search_code` searches by words by default and loads no model: BM25 over each chunk's text, its name, the words of its identifiers and its parent's, the chunks named as the query first, a name bare or qualified (`Stack.push`), its case aside. On the harness's questions, against the search by meaning and words fused, the right chunk ranks lower (chunk MRR, click / fd / this repository: code 0.667 / 0.454 / 0.667 against 0.761 / 0.621 / 0.800, documentation 0.600 / 0.583 / 0.686 against 0.700 / 0.750 / 0.786), a bare name's chunk comes first nearly always (0.989 / 1.000 / 0.982 against 0.818 / 0.793 / 0.603), and a question no corpus answers gets a full list, as no floor holds back a hit by words. Agents search again with other words (ADR 008, pilot 8), and gained nothing from meaning offered beside the words (pilot 9)
- An index run, `index_codebase`'s or the one inside a search, stores the chunks and their words alone, in seconds where embedding took minutes; each model's vectors are kept apart inside the index, by the chunk they belong to, so another model embeds again without chunking again, and an edit keeps the vectors of every chunk whose text it left
- Two sessions on one repository never index it twice: a run holds the index for its whole length, and one that finds it held waits, as long as it takes, saying another session is building the index, then finds the work done
- A first index of SQLite is built whole, out of readers' sight, without a journal, and put in place at its end: Sentry's (8,546 files) by words takes about 14 s instead of 17; one stopped midway keeps nothing, and the next search builds it again
- An index run writes its files as it chunks them, never holding every chunk at once, and one of 8 MiB of source or more chunks them in up to four processes beside the server, half its cores, made for the run and gone after it, or with the server where it is killed: on Sentry (8,546 files, 67 MB of source) the first search of a repository never indexed answers in about 10 s instead of 14-17, and the server's processes hold 300 MB at their peak instead of 345 and 135 MB after instead of 220; Python's forkserver and resource tracker stay for the session, 35 MB together. A repository of less source, or a system that cannot run a pool of processes, is chunked in the server
- SQLite keeps the indexes: the chunks, their vectors and a full-text index of their words (FTS5) in one database file per index, a write in one transaction. Against LanceDB on the three pinned corpora, through the server: a warm search 30-36 ms against 49-52 at the median, `index_status` 14-18 ms against 23-31, a search after an edit 44-54 ms against 119-139, and an edit grows the index 0.01 MB against 0.9-2.5. On the harness's questions three answers fell from first to second place, swapped with a near tie (in reciprocal-rank sum, code -0.55, documentation 0, bare names -0.67). The install is 190 MB instead of 616: LanceDB, pylance and pyarrow moved to the optional extra `semantic-code-mcp[lancedb]`. Every index is built again once
- A search after an edit scans the project once: its index run takes the files the search's check found, scanning again only where another session wrote the index since, and plans against the index once it holds it, so on Sentry (8,546 files) it answers in 0.47 s instead of 0.68-0.70 s, at its slowest 0.55 s instead of 0.70-0.99 s; with the leaner check below, 0.29 s
- A search finds what changed since the index at a fraction of the cost: discovery judges and resolves each directory once for all its files, from where its parent lives, and matches the ignore patterns as one expression, and the files found and those the index recorded are compared by their paths' text, so on Sentry (8,546 files) `index_status` takes 0.15 s instead of 1.13 s and a warm search about 0.24 s instead of 1.18 s
- A search reads what changed since the last one instead of every file: the server follows each repository's files with the system's own watcher, read directly, with nothing to install (inotify on Linux, FSEvents on macOS, ReadDirectoryChangesW on Windows); a path that changed is read again by the scan's rules, git is asked for its listing again where it may have changed, and a full scan comes again where only it can tell (a `.gitignore`, an excludes file or git's configuration changed, a repository made or gone, a queue that overflowed), and every 5 minutes as a check, which logs any difference; a forced `index_codebase` scans whole. On Sentry (8,546 files) on Linux `index_status` takes 0.03 s instead of 0.15-0.20, a warm search 0.09-0.10 s instead of 0.23-0.26 and a search after an edit 0.15-0.18 s instead of 0.29-0.44; the server holds about 15 MB more. On Windows a directory holding one watched cannot be moved meanwhile, as with any watcher there: a linked worktree's main repository while the worktree is followed. `SEMANTIC_CODE_MCP_WATCH_FILES=false` turns it off; a directory outside a repository, a project with submodules, and a network or FUSE file system are scanned at each search as before
- `search_code` asks Claude Code to load it with the prompt rather than behind its tool search (`_meta` `anthropic/alwaysLoad`), so an agent no longer spends a turn looking it up before it searches; on a repository the agent did not know, it took the search unprompted in every session where it had taken it in two of three; `index_codebase` and `index_status`, seldom needed as every search indexes first, still wait behind tool search
- A server by words loads no model: on the pinned corpora it holds 103-112 MB, on Sentry (87,694 chunks) 236 MB after its work and 355 MB at its peak. One set to search by meaning hands back the embedder's working memory after each batch of documents: on the pinned corpora it peaks at 630-660 MB and holds 290-330 MB after, where 0.4's server kept its peak of about 1.1-1.3 GB for as long as the session ran
- The server's stderr holds its own JSON log lines alone: the libraries' records below WARNING (httpx's request of each model file, sentence-transformers' lines and progress bar) no longer land among them
- Embeddings run on onnxruntime over the model's ONNX export, with its own tokenizer, instead of sentence-transformers and torch: the install took about 0.6 GB on every platform (95 MB since the model's libraries became an extra), against 1.5 GB on Linux with the CPU-only torch index (3.5 GB without it), a chunk embeds 13-24% faster, and the model loads in a third of a second instead of four; a fresh server, which by words loads no model, answers its first search on an indexed project in 0.8 s instead of 5.5. The vectors are those sentence-transformers made, to a few parts in a million; an index built before is built again once, as the embedder's version is part of what names an index; the first start fetches the model's ONNX file (90 MB) once, and with `HF_HUB_OFFLINE=1` a cache that holds only what sentence-transformers fetched does not have it: the server says so and asks for one start online. A model set by `SEMANTIC_CODE_MCP_EMBEDDING_MODEL` needs mean pooling and an ONNX export in one file (`onnx/model.onnx`, no weights beside it); one without, or one whose export or tokenizer cannot be loaded or fed, is refused when it loads, with the reason and the file; only the default model's vectors are tested against sentence-transformers'
- Requires mcp 2.x: the server is an `MCPServer`, its state lives in the lifespan instead of module globals
- Every direct dependency is bounded below its next breaking release, so none reaches a fresh `uvx` install untested, as mcp 2.x reached 0.4.0; what they pull in follows their own bounds
- Requires LanceDB 0.30.2 exactly, and pylance 4.0, the Lance library it is built on, for that upkeep: another LanceDB may bundle another Lance
- A tool that fails answers as an MCP error (`isError`) with its message, instead of a successful result holding `{"error": ...}`
- Each tool answers one model, without FastMCP's `{"result": ...}` wrapper
- `search_code` refuses a `limit` above 50, and every tool refuses a relative `project_path`
- A `search_code` result's `chunk_type` is now `kind`
- A directory inside a git repository is part of the repository's project: the tools index and report the whole repository once, and a search asked from the directory keeps to it, instead of building a second index from scratch for each directory. A directory of a large repository now costs the whole repository's first index, with no way to keep to the directory alone; `.semantic-code/` lies at the repository's root; indexes 0.4 built for subdirectories are left behind. A directory stays its own project outside a work tree, where the repository ignores it, or in a repository rooted at or above the user's home directory, as a home under git for dotfiles is
- `search_code` ranks by Reciprocal Rank Fusion of what the vector and the keyword searches found, in one version of the index, instead of a weighted sum of their scores: the right file's mean reciprocal rank went from 0.455 / 0.549 / 0.698 to 0.497 / 0.502 / 0.785 on click / fd / this repository
- The keyword search finds a chunk by the words of its name (`getUserName` by "user name", `HTTPServer` by "server") and by the words of what it is part of: a method by its class's, a section by its headings above
- A search by meaning drops a hit below a cosine similarity of 0.35 to the query, or of 0.25 when its words match the query's, instead of below a blended score of 0.3: eight questions far from any corpus get no result, seven programming questions no corpus answers 19 over the three corpora; a hit named as the query comes back however far in meaning, as an agent asking for an identifier it knows wants its definition
- The keyword search keeps the words LanceDB drops by default, such as `a`, `the`, `is` and `not`: on the harness, with every right answer counted, the chunk reciprocal ranks of its 50 code and 16 documentation questions summed 36.37 and 10.60 with them dropped, 37.09 and 12.00 kept, while the held-out questions fell 0.52; on the corpora the full-text index grows 6-8% and a keyword search takes 0.1-0.3 ms more, at 100k chunks 9% and 1.5 ms (p50 3.1 → 4.6 ms). Prose is fused as code is: weighing it down gained nothing
- Results come in the ranked order, no longer grouped by file: the right file's mean reciprocal rank went from 0.497 / 0.502 / 0.785 to 0.539 / 0.585 / 0.825
- The index keeps each file by its path from the project root, so a project moved with its local index (`.semantic-code/`) keeps it
- The tools no longer send log notifications (`Searching for: ...`, `Found N results in X ms`); progress notifications remain, and now only move forward, the model load included
- Progress moves while the chunks embed, 256 at a time, instead of standing at 70% until the whole embedding is done
- The index keeps a record of each file and a manifest of what built it, in place of `file_mtimes.json`
- The default model loads at a pinned revision, not whichever the hub's `main` holds when it is fetched. A cache that holds only an older revision fetches the pinned one's small files once; with `HF_HUB_OFFLINE=1` the model does not load until it has
- Each chunker version and storage layout keeps an index of its own beside the others in the project's index directory, each model's vectors apart inside it: two sessions of different versions never build each other's index again, and switching back finds the old one. The first search after the upgrade builds a new index; one that no session read or wrote for a week is removed, 0.4's included, but never while a writer holds it
- An index run writes a slice of files at a time: a forced re-index keeps every file searchable throughout, instead of emptying the index first; a first index stopped midway keeps nothing on SQLite, whose first index is built whole, and the files it wrote on LanceDB
- After an edit, a search by meaning embeds only the chunks whose text it changed; after a forced re-index, every chunk again only for a model whose weights nothing pins, as the default's pinned revision vouches for its vectors
- A write no longer rebuilds the full-text index; every index run that wrote ends by folding its new rows into the full-text index and pruning old versions, and at a session's first such run, once in twenty after, and whenever the rows edits replaced pass 5% of the index, by compacting the index and rebuilding the full-text index whole. An edit no longer rewrites the whole index, as LanceDB's own upkeep did for each edit of a file other than the last: at 100k chunks an upkeep takes 69-77 ms against 669-672, a search 40 ms against 54, and the index peaks at 0.85 GB against 3.5
- `index_status` of a project never indexed no longer creates its index directory
- A damaged index database, a file that is no database, broken or emptied, is set aside beside the index with its write-ahead log, and the next run builds the index again, where every call used to fail on it
- A search's words are each read once, their case aside: a word repeated weighed more and took as long again
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
- Two server processes indexing one project at once (two Claude Code sessions) could store its chunks twice: a write now holds a lock file beside the index's directory
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
