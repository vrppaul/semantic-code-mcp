# 006: Rebuild on mcp 2.x — layers, storage, search, gates

**Status**: accepted
**Date**: 2026-10-25
**Supersedes**: 002 (agent instruction files); in part 001 (tool list, storage layout) and 004 (base chunker helpers; the per-language class layout stays)

## Context

On 2026-10-25 we reviewed master at `ec06c4a` (v0.4.0) from four sources:

1. A code review with measurements: bugs reproduced by script, indexing and search timed, search quality scored on 14 queries over this repo.
2. A comparison with the owner's newer projects, `../claude-review` (Python) and `../livereel` (TypeScript), which show how his practice has moved on: layer contracts, code-rule gates, a shipped skill and plugin, CI before release.
3. GitHub issue #3 (rbautista84): the package does not start on mcp 2.x; untracked files are invisible; prose outranks code. It includes a 10-query eval on a private repo.
4. GitHub PR #2 (same author): replace `git ls-files` with a filesystem walk and add a filename-based secret filter.

Measurement machine: i7-1260P (4P+8E cores), Python 3.14.3 (GIL build), lancedb 0.27.0, sentence-transformers 5.2.0, torch 2.10.0+cpu. Corpora: this repo (61 files, 606 chunks), pydantic (105 files, 2035 chunks), numpy (407 files, 11492 chunks). Baseline numbers are in Appendix B.

What we found, in short:

- **The published package does not start.** `mcp>=1.26.0` has no upper bound; mcp 2.x renamed `FastMCP` to `MCPServer`, so a fresh `uvx semantic-code-mcp` fails on `import mcp.server.fastmcp`. Claude Code reports only "Failed to connect".
- **Code goes missing from the index with no error.** Untracked files are skipped. A file deleted on disk but still tracked is re-indexed on every search. Non-ASCII file names never resolve. One non-UTF-8 file aborts indexing for the whole project, on every search. A comment between a decorator and `def` drops the definition. Generic and path-qualified Rust `impl`s produce nothing, and `mod` bodies are never read. Python files without top-level definitions produce nothing.
- **The index degrades with use.** Every write rebuilds the full-text index. Each file is deleted in its own transaction. Old versions are never pruned. Two concurrent `index()` calls duplicate rows. Nothing records which model built an index, so changing the model corrupts it with no error, or crashes if the dimension differs.
- **Search ranks prose over code**, and the score blends two scales that cannot be compared. In both evals, docs and docstrings beat the code they describe. The agent cannot filter by chunk kind, language or path.
- **Indexing is 20–50× slower than our own target.** On CPU, embedding takes about 99% of the time. The first search on a 400-file repo takes 7 minutes.
- **The structure costs more than it carries.** It has seven layers for about 900 lines of logic, four module-level globals, protocols used only by mocks, and two sources of truth for what is indexed.
- **Our gates pass without checking anything.** There is no CI, and a tag publishes untested code. The `ty` hook ends in `|| true`. Bandit 1.8.6 crashes on every file under Python 3.14 and still exits 0 (verified). The pre-commit ruff is two minors behind the locked one. None of the owner's code rules is checked by a tool, and the code breaks several of them.
- **The agent learns about the tools only from three docstrings**, and one of them is stale. There is no skill, no plugin, and no server `instructions`.

Appendix A lists every finding with an ID and the stage that resolves it.

## Decision

Fix what blocks users now, on the current code. Then rebuild the server on mcp 2.x around the structure below, with the gates in place first, so each later stage lands under them. Every search-quality change must show its effect on the eval harness (D12). Every speed change must show its effect on the benchmark (D12).

### D1. Hotfix the published package

Release 0.4.1 with `mcp>=1.26.0,<2` and the correctness fixes from D5 and D6 on the current structure, each with a test that fails before the fix. (The plan was a 0.4.1 with the pin alone and a 0.4.2 after; both were ready before either shipped, so they ship as one.) The chunkers keep their files through the rebuild, so these fixes carry over. The scanner fix moves into `discovery/` in stage 4.

### D2. Layers and composition root

```
semantic_code_mcp/
  domain/      models, errors, ABC interfaces — imports nothing but pydantic and the stdlib
  discovery.py git ls-files / filesystem walk → list[SourceFile]
  chunking/    tree-sitter chunkers, registry by Language
  embedding/   sentence-transformers adapter (model loaded in a thread)
  storage/     one LanceDB store: chunks, files, manifest
  services/    indexing.py — plan → chunk → embed → write, under a per-project lock
               search.py   — retrieve → fuse → filter → shape
  server/      MCP tools, lifespan, presenters (domain → response), error mapping, middleware
  app.py       composition root
  cli.py       entry point
```

- Layer order: `cli | app → server → services → discovery | chunking | embedding | storage → domain`. Import-linter contracts in `pyproject.toml` enforce it, each with its reason as a comment (shape: `../claude-review/pyproject.toml`).
- Forbidden-import contracts: `mcp` only in `server/`, `app.py`, `cli.py`; `lancedb`/`pyarrow` only in `storage/`; `sentence_transformers`/`torch` only in `embedding/`; `tree_sitter*` only in `chunking/`.
- No module-level mutable state. `app.py` builds `Settings` and passes a lifespan to `MCPServer(lifespan=...)` (verified on mcp 2.3.0). The lifespan yields an `AppContext(settings, embedder, projects)`, and tools read it from the request context. `projects` is a `ProjectRegistry` holding one `ProjectIndex` (store, lock) per resolved index path. This removes `configure()/get_container()`, `configure_settings()/get_settings()`, `logging._configured`, `profiling._profiling_enabled/_profiles_dir`, and the deferred `server` import in `app.py`.
- Interfaces are ABCs with `@abstractmethod`; implementations mark methods `@override`. `typing.Protocol`, `typing.Any` and `typing.cast` are banned through ruff `banned-api`, as in claude-review. Reason: an ABC fails at class definition when a member is missing. Protocol plus `MagicMock(spec=)` passes a fake that answers every call with a mock.
- Drop `Indexer` (one method and two pass-throughs) and the `LanceDBConnection`/`LanceDBVectorStore` split (one store per connection, both cached). `services/indexing.py` owns the pipeline. `storage/` exposes one store class.
- A parameter every caller must pass has no default (today `IndexService.cache_dir: Path | None = None` hides a required value).

### D3. Domain models

All domain models are frozen pydantic models; dataclasses are not used. Paths are `Path` in the domain and `str` in responses.

| Model | Fields | Replaces |
|---|---|---|
| `Language` (StrEnum) | `python`, `rust`, `markdown` | extension checks spread over the chunkers and the scanner |
| `ChunkKind` (StrEnum) | `module`, `function`, `method`, `klass = "class"` (the keyword alias `.claude/rules/python.md` prescribes), `section` | `ChunkType`. Rust `struct`/`enum`/`trait`/`impl` map to `class`, and the skill says so. |
| `SourceFile` | `path`, `language`, `mtime_ns`, `size_bytes` — taken at scan time | bare `str` paths, and the mtime read after indexing (R11) |
| `Chunk` | `chunk_id`, `path`, `language`, `kind`, `name`, `parent`, `line_start`, `line_end`, `content`, `embedded_text` | `Chunk`. `chunk_id` hashes path, kind, qualified name and start line, and replaces the `(file_path, line_start)` merge key (A5). `parent` is the owner's qualified name (`Stack` for `Stack.push`). `embedded_text` is what the model sees (D6 skeletons, D9 identifier words); `content` is what the agent gets. |
| `IndexManifest` | `model_name`, `dimension`, `chunker_version`, `schema_version` | nothing (R9) |
| `IndexDelta` | `to_index: list[SourceFile]`, `to_delete: list[Path]` | `FileChanges` and `ScanPlan`, which overlap; `ScanPlan.all_files` is never read |
| `IndexReport` | `files_indexed`, `chunks_indexed`, `files_deleted`, `duration_seconds`, `rebuild_reason: RebuildReason \| None` | `IndexResult` and its `duration_seconds = 0.0` default |
| `IndexStatus` | `is_indexed`, `files_count`, `chunks_count`, `stale_files_count`, `last_indexed_at`, `manifest` | `IndexStatus` with a full `stale_files` list that no tool returns |
| `SearchQuery` | `text`, `limit`, `chunk_kinds`, `languages`, `path_glob`, `exclude_glob` | positional `query, limit, min_score, vector_weight` |
| `SearchHit` | `chunk`, `similarity`, `vector_rank: int \| None`, `keyword_rank: int \| None`, `fused_score` | `SearchResult(Chunk)` with one blended `score` |
| `SearchReport` | `hits`, `timings`, `index_report: IndexReport \| None` | the mutable `SearchOutcome` dataclass, built with a zero `IndexResult` and patched afterwards |
| `SearchTimings` | named fields for each phase | unchanged in spirit |

Vectors do not live in domain models. The store takes `write(chunks: list[Chunk], vectors: NDArray[float32])`, and the numpy matrix goes straight into Arrow (`pa.FixedSizeListArray`). This replaces `ChunkWithEmbedding` with 384 Python floats per chunk and the `tolist()` round trip (S4).

Response models live in `server/`, built by presenter functions: `SearchCodeResponse(results, debug: SearchDebug | None)`, `IndexCodebaseResponse`, `IndexStatusResponse`. Each tool has one return type, so FastMCP stops wrapping the result as `{"result": ...}` (R10). `ErrorResponse` goes away (D10). Formatting logic (`FormattedSearchResult.from_domain`, `SearchStats.from_outcome`, the `// 4` token estimate) moves out of the model layer, which removes the import from `models` into `services` (A6).

`__init__.py` files hold only a docstring. The re-exports in `models/__init__.py` go, as CLAUDE.md requires, and imports name the module.

### D4. Storage

One LanceDB database per project index, with three parts:

- **`chunks` table**: `chunk_id`, `vector` (dimension from the embedder, not a constant), `path`, `language`, `kind`, `name`, `parent`, `line_start`, `line_end`, `content`, `search_text`. The full-text index covers `search_text` (D9).
- **`files` table**: `path`, `mtime_ns`, `size_bytes`, `chunk_count`. This is the single record of what is indexed. `file_mtimes.json` and `storage/cache.py` go (A4). A file's row is written after its chunks, so it marks the file as complete: after a crash between the two, the file counts as stale and gets re-indexed. Staleness comes from a projected scan of `files` (`path`, `mtime_ns`, `size_bytes`), never from a full table read with vectors.
- **`manifest.json`**, written atomically, holding `IndexManifest`. On open, a mismatch in model, dimension, chunker version or schema version drops and rebuilds the index. The tool response says why (`rebuild_reason`). Upgrading users get one automatic full rebuild.

Write path per `index()` run:
1. One `delete("path IN (...)")` for all deleted and changed files, with escaped literals.
2. One `add` of all new chunks.
3. One `merge_insert` of the `files` rows.
4. One full-text index rebuild.
5. `optimize(cleanup_older_than=timedelta(0))`, which compacts, prunes old versions and folds new rows into existing indexes (from the lancedb 0.27 docstring).

Whether `optimize()` alone keeps the full-text index current, so that step 4 can go, is checked in stage 5 and measured.

Reads use Arrow (`to_arrow()`, `to_list()`). `pandas` leaves the dependencies (S3). The vector search stays a flat cosine scan: 23–36 ms at 11.5k chunks. IVF/PQ tuning (TODO) waits until a corpus over 100k chunks shows the need.

With `local_index=true`, the index directory `<project>/.semantic-code/` gets its own `.gitignore` holding `*`, so the user's repo never shows it (P2).

### D5. Discovery

- **Detecting a work tree:** `git rev-parse --show-toplevel`, not `(project_path / ".git").is_dir()`. In a linked worktree or a submodule `.git` is a file, and in a subdirectory of a repo it is absent. Today both cases fall back to the walk without `.gitignore` (O5).
- **In a git work tree:** `git ls-files -z --cached --others --exclude-standard --deduplicate -- <globs>`, decoded with `os.fsdecode`, keeping only paths that exist as files. This finds untracked files, still honours `.gitignore`, drops tracked-but-deleted files and handles any file name (R2).
- **`ignore_patterns` filters the results of both modes.** Today they apply only to the walk, so a user cannot exclude `docs/**` in a git repo (P2).
- **Symlinks:** a file whose resolved path lies outside the project root is skipped in both modes. A tracked symlink must not pull outside files into the index.
- A missing `git` binary (`FileNotFoundError`), a timeout (`TimeoutExpired`) or a non-zero exit falls back to the walk and logs the reason (R3). The timeout becomes a named setting.
- **`use_gitignore`** governs both modes:
  - `true` (default) — in a git work tree, the git command above; outside one, the walk with `ignore_patterns`.
  - `false` — the walk with `ignore_patterns` everywhere. This covers the issue #3 case: a nested `.gitignore` of `/*` used as an allowlist.

  Today the setting affects only the walk fallback.
- **The walk** does not follow symlinked directories (taken from PR #2). It prunes `ignore_patterns`, which becomes the single list. The hard-coded `skip_dirs` set goes. The default grows to cover `target`, `dist`, `build`, `.mypy_cache`, `.ruff_cache`, `.tox`, `.cache` and `site-packages`, as well as today's entries.
- **An enclosing repository that ignores the project** (a home directory under git for dotfiles, an ignored `scratch/`) does not get to empty it: when the work tree's root is above the project and `git check-ignore .` says the project is ignored, discovery walks.
- **Submodules** are listed with `--recurse-submodules` on the tracked listing (git does not combine it with `--others`). An untracked repository nested inside the project is not looked into.
- **Git's environment:** `GIT_DIR`, `GIT_WORK_TREE` and the like are dropped from git's environment, so they cannot point discovery at another repository.
- **The hand-written `.gitignore` parser goes.** It has no negation or anchoring, and outside a git work tree the walk uses `ignore_patterns` only.
- **We add no filename-based secret filter** (PR #2 proposal). Only `.py`, `.rs` and `.md` files reach the chunkers. Git mode honours `.gitignore`, where `.env`-style files live. A pattern like `*token*` would drop `tokenizer.py` and `token_bucket.py`. The index stays on the user's disk. SECURITY.md states what we read and where we write.

### D6. Chunking

Keep one class per language on a shared base (ADR 004). Fix the base first, because JS/TS/Go would inherit every hole:

- **Read bytes and parse bytes.** Decode the text for lines as UTF-8 with `errors="replace"`. A file that cannot be read is logged and skipped, never fatal (R1). This also removes the dependence on locale encoding (Windows cp1252).
- **A span helper** `make_chunk(first_node, last_node, kind, name, parent)` replaces the five hand-built `Chunk`s in `rust.py` and the one in `markdown.py` (C1).
- **Fallback module chunks:** a Python or Rust file that yields no definitions — nothing, or only a module docstring — is indexed whole, in `module` chunks of at most 100 lines, as Markdown already covers files without headings (R6).
- **Files skipped:** over 1 MB (generated or data), or binary by git's rule (a NUL among the first 8000 bytes, which also catches UTF-16).
- **Python:** `_get_decorated_definition` looks for `function_definition` or `class_definition` by type, not for any `NodeType` member, which today matches `comment` (R4).
- **Rust:**
  - name an `impl` from `child_by_field_name("type")` and `child_by_field_name("trait")`, giving `Stack`, `Display for Stack` and `Default for Stack` (verified that the fields exist);
  - recurse into `mod_item` bodies, including `#[cfg(test)]` modules (R5);
  - collapse the struct, enum and trait cases of `_walk_items` into one data-driven branch.
- **Markdown:** `line_end` of the last section must not run past the last line. The tests that assert the off-by-one go (R7).
- **Class skeletons:** a class, `impl` or trait chunk embeds a skeleton: the signature, the docstring and the member signatures. Its `content` is the same skeleton, and its range spans the whole definition. Member bodies live only in their own chunks, with `parent` set. This ends double embedding and duplicate payloads (Q2), and cuts the text sent to the model (S1). Accept or reject it on the harness (D12).
- **`CHUNKER_VERSION`** is a constant in `chunking/`, bumped by any change that alters chunk output. It feeds the manifest (D4).
- **Parsing:** one `asyncio.to_thread` for the whole chunking step. Batches of 20 gain nothing under the GIL (S5), so `CHUNK_BATCH_SIZE` goes.
- **Tags queries:** before adding JS/TS/Go, spike tree-sitter tags queries (`tags.scm`) as the per-language definition of "what is a symbol". That spike belongs to the multi-language epic, not to this ADR (P3).

### D7. Embedding

- **Interface:** `Embedder` ABC with `model_name`, `dimension`, `embed_query(text) -> vector` and `embed_documents(texts, on_progress) -> matrix`. Queries and documents are separate, because candidate models need different prefixes. The prefixes live in a data table of model profiles, not in code.
- **Loading:** the sentence-transformers adapter loads the model inside `asyncio.to_thread` on first use and reports progress ("Loading embedding model"). Only tools that embed trigger it. `index_status` never loads the model (R12).
- **Progress:** `embed_documents` works in slices and reports progress after each one, instead of one multi-minute call behind a single "70%" (S1d).
- **New setting `torch_threads`**, applied with `torch.set_num_threads`. Its default comes from the benchmark. On this machine 4 threads gave 45.5 chunks/s against 31.6 with the torch default of 12.
- **Device:** sentence-transformers already picks CUDA or MPS when present. `embedding_device` is never read, so it goes. This likely explains the issue author's 391 files in ~7 s against our 1–3 files/s on CPU [suspected; ask in the reply].
- **ONNX:** evaluate the sentence-transformers `backend="onnx"` with int8 quantisation in the speed stage. If the benchmark shows a gain and the harness shows no quality loss, ship it as the default and make torch an optional extra. That would also shrink the 1.7 GB install.
- **The model choice waits for the harness.** Candidates, none measured yet:
  - same cost, 512 context: `BAAI/bge-small-en-v1.5`, `Snowflake/snowflake-arctic-embed-s`, `thenlper/gte-small`;
  - trained on code, about 4–5× the CPU cost: `jinaai/jina-embeddings-v2-base-code` (needs `trust_remote_code`), `nomic-ai/CodeRankEmbed`.

  Change the default only when the harness shows a clear gain at an indexing cost we accept, both recorded in the decision. The manifest rebuilds old indexes. UniXcoder and CodeBERT (TODO) are not sentence-embedding models and drop off the list.

### D8. Indexing pipeline and concurrency

- `services/indexing.py`: `scan → diff against files table → chunk → embed → write`, under the project's `asyncio.Lock` from `ProjectRegistry`. The write phase also holds a file lock (`filelock`) on `<index>/write.lock`, so two Claude Code sessions on one project cannot duplicate rows (R8).
- Search takes the same lock only when it must re-index first. Plain reads do not lock, because LanceDB serves a consistent version.
- All blocking work runs in threads: status, diff, table drop on rebuild, model load (A3). `get_status` runs once per request and is cheap (D4). Today it runs twice per search and costs 15–90 ms each time.
- Progress phases and their percentages are a data table, reported through a `ProgressReporter` class. This removes the nested `_progress` closures (C4) and the literal percentages (C6).
- A failed write raises a typed error (D11). The cache-after-success logic goes with the cache.
- Background re-indexing (answering from the old index while a new one builds) stays in TODO. Locks and the files table make it possible later.

### D9. Search

- **Candidates:** the vector top `limit × CANDIDATE_FACTOR` and the full-text top `limit × CANDIDATE_FACTOR`, both with the same `where` prefilter built from `SearchQuery` (kinds, languages, path globs). Filtering before the cut keeps a filtered result full (Q5, GH3).
- **Fusion:** Reciprocal Rank Fusion, `Σ 1/(RRF_K + rank)`, written by us (about 10 lines, covered by property tests), keyed by `chunk_id`. It replaces `0.5·(1−d/2) + 0.5·min(1, bm25/10)`, whose constant was tuned against the tantivy engine, while lancedb 0.27 uses native full-text search (Q3). Q4 shows that the keyword channel helps, so we keep it.
- **Similarity:** we compute cosine similarity for every candidate from its stored vector. A hit with no keyword match and similarity below `min_similarity` is dropped, so a nonsense query returns nothing. The default comes from the harness. It is a setting (`SEMANTIC_CODE_MCP_MIN_SIMILARITY`), not a tool parameter. `vector_weight` goes, since RRF has no weights (GH4).
- **`search_text`:** the content, plus identifiers split into words (`getUserName` → `get user name`, `duration_ms` → `duration ms`), plus `name` and `parent`. Today `getUserName` matches only the camelCase spelling (verified).
- **Prose against code:** the harness decides between no prior and a down-weight for `module` docstring and `section` chunks. The `chunk_kinds`/`languages` filters ship either way (GH3, GH5).
- **Grouping and recency:** grouping by file goes, and results come back in fused order (Q8). The recency boost goes unless the harness shows it helps. It has never been tested with a real boost.
- **Results:** `content` is cut at `MAX_RESULT_LINES` with `truncated: true`, and the agent reads the file for the rest.

### D10. MCP surface

- **`search_code(query, project_path, limit, chunk_kinds, languages, path_glob, exclude_glob, include_debug=false)`.** `limit` is capped at `MAX_LIMIT`. The `debug` block (timings, counts, index report) is returned only on request: it costs about 85 tokens per call today.
- **`index_codebase(project_path, force)`** returns `IndexReport`.
- **`index_status(project_path)`** returns `IndexStatus`, including the manifest and `stale_files_count`.
- **Errors raise `ToolError`** (`mcp.server.mcpserver.exceptions`, verified on 2.3.0). The client sees `isError: true`; today errors come back as `isError: false` with `{"error": ...}` (R10). One function in `server/` maps domain error kinds to messages. Tools never check `path.exists()` by hand.
- **Server `instructions=`** on `MCPServer` say when to use `search_code` and when Grep, that `project_path` is the repository root, which languages are indexed, and that indexing is automatic. Clients show this next to the tools even without the skill.
- **Docstrings** state the same facts. The stale "Scans Python files" goes (C8).
- **No `include_content` or `max_lines` parameters.** The line cap is a constant, and the agent Reads the range for more (P1).
- **One index per repository.** In a git work tree, `project_path` is normalised to the work-tree root. If the caller passed a subdirectory, results are limited to it by a path prefilter. Today a subdirectory builds a second, separate index from scratch (O6).
- **Spike: MCP roots** (`ListRoots` in mcp 2.x). If Claude Code reports workspace roots, `project_path` becomes optional and defaults to the root. If not, it stays required.
- **No new tools now.** `find_symbol`, `find_similar` (ADR 001), `list_indexed_files` and `clear_index` go to TODO. `index_codebase(force=true)` already clears.

### D11. Errors, config, logging, profiling, CLI

- **Errors:** a `SemanticCodeError` hierarchy in `domain/errors.py` — `ProjectNotFoundError`, `NoIndexableFilesError`, `IndexIncompatibleError` (raised internally, which triggers the rebuild), `ModelLoadError`, `IndexWriteError`. Code catches specific exceptions only, and `except ValueError: pass` in `ensure_table` goes (A7).
- **Settings:**
  - kept: `embedding_model`, `cache_dir`, `local_index`, `use_gitignore` (D5), `ignore_patterns`, `debug`, `profile`;
  - added: `torch_threads`, `min_similarity`, `git_timeout_seconds`;
  - removed: `embedding_device`, `chunking_target_tokens`, `chunking_max_tokens`, never read (C7);
  - `env_file=".env"` goes: the server's working directory is the user's project, and its `.env` is not ours to read (C9).
- **Constants** with a meaning are named: the index path hash length, timeouts, line caps, RRF constant, candidate factor and progress table (C6).
- **Profiling** becomes server middleware (`MCPServer(middleware=...)`, present in 2.3.0; the API is checked in stage 4) instead of a decorator with module globals. Profiles go under `cache_dir/profiles`, not `./profiles` in the user's project (C9). The decorator's `P`/`T` type variables disappear with it (C3).
- **Logging** is configured once in `app.py`, with no `_configured` flag.
- **CLI:** the `SIGINT` lambda calling `sys.exit` goes. Shutdown follows `MCPServer.run()` (checked against the mcp 2.x migration guide in stage 4) (C9).
- **Python** stays at `requires-python >= 3.14`. uv installs a managed 3.14 for `uvx` users, so the floor costs them nothing (P5).

### D12. Evaluation harness and benchmark

Both are built before the structural rebuild, so the rebuild proves it changed nothing that matters. They form one package, `evals/`, outside `src/` and outside the wheel, with two commands that share the corpora and the client.

- **The client** starts the server as an agent does, over stdio, and speaks to it only through its tools, so the harness stays valid across the rebuild; one function knows the shape of a tool's answer.
- **Corpora** are data, one TOML file each under `evals/corpora/`: a repository pinned by commit and its questions, each naming every answer that counts (a file and a chunk name). Three corpora, 51 questions: this repo at v0.4.0 (docs-heavy, which reproduces GH3/GH5), pallets/click (Python with a large Markdown manual) and sharkdp/fd (Rust). A run fails when an answer names a file its checkout lacks.
- **`uv run python -m evals quality`** scores hits@1, hits@3 and MRR twice — for the right file and for the right chunk — matching the two scores in issue #3.
- **`uv run python -m evals speed`** measures a full index (files/s, chunks/s), warm search p50/p95, the first search of a freshly started server (process start and model load), the server's peak memory (Linux), and index growth per single-file edit.
- **Baselines** live in `evals/baseline/`; each run prints its numbers beside them and `--save-baseline` replaces them. Speeds compare only on the machine they were measured on. Every search, chunking or speed change records its before and after in the commit message.
- It runs locally, not on every commit: it needs the model and clones. CI may run it on demand.
- The performance targets in `.claude/rules/mcp-server.md` are reset from its numbers in stage 7. Today all four are missed (Appendix B), and "50 files/s" on CPU is not reachable with any model we would consider [suspected].

### D13. Tests

- **Layout:**
  - `tests/unit` — chunkers, fusion, discovery, presenters, models;
  - `tests/integration` — LanceDB, the real model, real git repos in `tmp_path`;
  - `tests/e2e` — the server driven through mcp's in-memory client session, calling all three tools against a sample project.

  Today no test touches `server.py` (T3), and no test creates a git repo, which is how R2 and R3 slipped through.
- **Property tests (hypothesis):**
  - every chunk has `1 <= line_start <= line_end <= line count`, and its range text equals the file's lines;
  - RRF output is a permutation of the union of its inputs, and adding a match never lowers a hit's rank;
  - similarity stays in [-1, 1];
  - discovery never returns a path that does not exist.

  One property test replaces the example tests of its rule.
- **Fakes:** hand-written fakes that keep what they were given replace `MagicMock(spec=...)`. Unused fixtures go (T5).
- **Banned in tests:** branches; assertion-free tests (`isinstance(chunks, list)`, `is not None` after construction, the "weight adjustable" test) (T1); tests of private methods; exact progress strings; call counts on mocks (T2); `time.sleep` for mtimes, replaced by `os.utime` with explicit values (T4).
- **Rules:** the owner's naming and no-nested-function rules apply to tests too (`cosine_sim` nested in `test_embedder.py` goes). Magic-value and annotation rules do not.

### D14. Gates, CI, release

Pre-commit (local hooks through `uv run`, `default_stages: [pre-commit]`):
- **ruff** as a local hook, which ends drift from the lock (0.12.12 against 0.14.14). Rule families added from `../claude-review/pyproject.toml`: `ANN`, `BLE`, `EM`, `FBT`, `ARG`, `A`, `TRY`, `RET`, `PTH`, `G`, `LOG`, `ISC`, `INP`, `PGH`, `PLC`, `PLE`, `PLW`, `PLR0913`, and `PLR2004` with `allow-magic-value-types = []`. `banned-api` covers `typing.Protocol`, `typing.Any` and `typing.cast`. The dead per-file ignores (`D`, `ANN`) for tests go.
- **ty** without `|| true`, with `error-on-warning = true`, over `src` and `scripts` (tests excluded, as in claude-review). The current unused `type: ignore` at `chunkers/rust.py:337` gets fixed.
- **bandit 1.9.4.** 1.8.6 is a no-op under Python 3.14 (verified: an exception on every file, exit 0).
- **import-linter** contracts (D2).
- **`scripts/check_code_rules.py`**, adapted from claude-review: nested functions, short names, models taken apart by field name. The claude-review-specific parts go, and the file says which ones (its `.claude/rules/scripts.md` explains how to mark a partial copy).
- **`scripts/check_instruction_files.py`**, copied whole from claude-review, as livereel did (D15). We adopt livereel's rule for such copies (`../livereel/.claude/rules/scripts.md`): a script taken whole is changed in claude-review and copied here, and its tests live there.
- **`scripts/check_skill.py`**, ported from `../livereel/scripts/check-skill.ts`. It loads the server and lists its tools. It fails if the skill misses a tool or parameter, names one that does not exist, or lists languages other than the chunker registry's (D16).
- `pytest --collect-only` on every commit, plus deptry and typos. vulture lands in stage 4, dependency and hook together: on today's code it flags pydantic fields, MagicMock attributes and the dead settings D11 removes, so it starts on the rebuilt code.

The code-rule gates (PLR2004, check_code_rules, import-linter, banned-api) land in stage 4 with the code written to pass them. Stage 2 adds only the gates the current code passes after small fixes.

CI and release:
- **`.github/workflows/ci.yml`** with jobs lint (`pre-commit run --all-files`), test and build. It is callable through `workflow_call`, with actions pinned by SHA. The Hugging Face model cache is kept between runs.
- **`publish.yml`** calls `ci.yml` and publishes only after it passes, with `uv publish --check-url https://pypi.org/simple/`, so a re-run is safe (shape: `../claude-review/.github/workflows/release.yml`).
- **The `pypi` GitHub environment** gets the owner as a required reviewer: today `protection_rules` is empty (verified). This is the PyPI counterpart of livereel's npm staging approval. It is a repository setting the owner applies.
- **Versioning** stays with `hatch-vcs`.

### D15. Agent docs, rules, repo hygiene

- **`CLAUDE.md` becomes a symlink to `AGENTS.md`**, and the instruction-files gate holds the size and path limits. This supersedes ADR 002, whose reason for two files ("AGENTS.md: no rules references") both sibling projects contradict. `.claude/rules/agent-docs.md` goes, because the gate does its job.
- **`AGENTS.md`** is rewritten on the sibling template: Commands; How we work; Layers (D2 in one line, pointing at the contracts); Rules for all code (the owner's code standards); Only better; Rules by area, naming each rules file. The `searcher.search(...)` example, which names a class that no longer exists, goes.
- **Rules files:**
  - `development-process.md` merges into `AGENTS.md`: it repeats the uv rules and lists dependencies that `pyproject.toml` already lists;
  - `mcp-server.md` is rescoped to `server/`, drops `__init__.py` from `paths`, and its targets come from `python -m evals speed`;
  - `python.md` drops the re-export rule and states the docstring-only `__init__.py`;
  - `testing.md` is rewritten from `../claude-review/.claude/rules/tests.md` and D13;
  - `instructions.md` is added from claude-review ("what a tool can hold is a gate, not a rule");
  - every rules file opens with `paths:` and is named in `AGENTS.md`.
- **`TODO.md`:** one list per state, items removed when done (not ticked), the `## Done` prose moved to CHANGELOG, the duplicate `## In Progress` and the stale `search/searcher.py` reference removed. `.claude/rules/documentation.md` follows.
- **`CHANGELOG.md`:** add the missing 0.1.1 and 0.2.0 entries. Note that 0.1.0 was never tagged.
- **ADR 001:** status becomes "superseded in part by 006". It lists tools that never shipped (`semantic_search`, `file_pattern`, `find_similar`). Its date `2025-01-24` is a typo for 2026. Fix it with a note, so the history stays readable.
- **Repo files:**
  - `LICENSE` (MIT, as the README says), plus `license = "MIT"` and `[project.urls]` in `pyproject.toml`: the wheel ships without a license today;
  - `CONTRIBUTING.md` (shape: `../livereel/CONTRIBUTING.md`, with the release steps);
  - `SECURITY.md` (shape: `../claude-review/SECURITY.md`): a local tool that reads project files, writes under `cache_dir` or `.semantic-code/`, and downloads the model from Hugging Face on first run;
  - a PR template.
- **`.claude/commands/ship.md`** stops hard-coding `Claude Opus 4.5` in `Co-Authored-By` and takes the attribution from the session.
- **README:**
  - install through the plugin (D16);
  - the "adding a language" guide follows D6 (span helper, fallback chunk, `CHUNKER_VERSION`);
  - every setting documented (5 of 9 today);
  - discovery semantics (D5) and `.semantic-code/`;
  - the false claim that pre-commit enforces `ty` goes, since it will be true;
  - a hardware note on CPU versus GPU/MPS speed.
- **`.claude/agents/search-reviewer.md`:** a read-only reviewer that runs both `evals` commands on a change, reads the owner's code sections, `AGENTS.md` and the rules, and reports a verdict with `file:line` (shape: `../claude-review/.claude/agents/page-reviewer.md`).

### D16. Skill and plugin

- **`skills/semantic-search/SKILL.md`**, at most 200 lines, in the siblings' register. It is written after D10 settles; the outline is in Appendix C.
- **`.claude-plugin/marketplace.json`** with `source: "./"` and `skills: ["./skills/semantic-search"]` (shape: `../livereel/.claude-plugin/marketplace.json`). The plugin registers the MCP server through `.mcp.json` at the plugin root, so a single install brings the server and the skill. The README offers three routes: the plugin, `claude mcp add`, and `npx skills add` for the skill alone, as the siblings do.
- **The server command in `.mcp.json` must work on every platform.** On Linux, `uvx` needs the CPU torch index to avoid the 3.5 GB CUDA build. Stage 8 checks whether one command can serve all platforms. If D7 makes ONNX the default and torch optional, the problem goes away.
- **The plugin manifest carries no version**, so `hatch-vcs` stays the only source and no version has to be synced by hand.
- **`scripts/check_skill.py`** keeps the skill equal to the server (D14).

### Kept as is

- Version from git tags (`hatch-vcs`).
- Keep a Changelog format.
- ADRs in `docs/decisions/`.
- The split between domain and response models, now with presenters.
- Conventional Commits with `--force-scope`.
- The CPU-only torch index and the platform install notes.
- `paths:` frontmatter on rules.
- The `pypi` environment, now with protection.
- Hybrid search as a method (Q4).
- Module docstring chunks (ADR 003) and heading sections (ADR 005), with the prior decided by the harness.

### Decisions taken for the owner

The project rules say to ask before changing the storage schema, the tool signatures or the dependencies. The owner delegated these on 2026-10-25 ("тебе лучше знать"):

1. **Storage schema changes** (D4), with one automatic rebuild through the manifest. A migration path would cost more than one re-index.
2. **Tool signatures change** (D10): new filters, `include_debug`, `ToolError`, no `ErrorResponse`, no grouping. This breaks anyone parsing the old output; we ship it as 0.x minor bumps and say so in CHANGELOG.
3. **Dependencies:**
   - added: `filelock` (runtime); `import-linter` and `hypothesis` (dev); vulture, deptry and typos as pre-commit hooks;
   - removed: `pandas`;
   - decided on measurements in stage 7: ONNX runtime as default or extra, and torch as optional.
4. **Hotfix 0.4.1 before the rebuild.** The tag push itself waits for the owner's word, like every release.

### Stages and releases

| Stage | Content | Release |
|---|---|---|
| 0 | `mcp<2` pin | 0.4.1, with stage 1 |
| 1 | Correctness on current code: R1–R7, O5, O7 and `ignore_patterns` in git mode, with failing-first tests and real git repos in tests | 0.4.1 |
| 2 | Gates the current code passes, CI, release gated on CI, LICENSE/metadata, CONTRIBUTING, SECURITY, AGENTS.md + symlink + instruction gate, rules cleanup | — |
| 3 | Safety net: e2e tests through the in-memory client, `evals/` (quality and speed) with baselines | — |
| 4 | Rebuild on mcp 2.x: layers, lifespan, ABCs, errors, `ToolError`, single return types, presenters, settings cleanup, middleware, code-rule gates, import-linter | — |
| 5 | Storage and pipeline: tables, manifest, batched writes, `optimize`, locks, scan-time mtimes, Arrow, no pandas, cheap status, model loaded off the loop, embedding progress, D5/D6 remainder (skeletons, span helper, `CHUNKER_VERSION`) | 0.5.0 |
| 6 | Search: RRF, prefilters, `search_text`, similarity threshold, prose prior, no grouping, recency decision — each on the harness | 0.6.0 |
| 7 | Speed and model: threads, ONNX, model candidates, new targets in rules | 0.7.0 (if the default model changes, the manifest rebuilds) |
| 8 | Server `instructions`, docstrings, skill, plugin, skill gate, README, reviewer agent, MCP roots spike | 1.0.0 |
| 9 | Replies on GitHub. Issue #3: what changed, with numbers from `evals`; take up the author's offer of their harness and compare it with ours; ask about their hardware (O4). PR #2: thank the author, explain D5 and the alternative rejected above, and close. | — |

## Alternatives Considered

- **Patch the bugs and keep the structure.** Cheaper this week, but every later item touches the globals, the double source of truth and the pass-through layers. mcp 2.x forces edits to `app.py`, `server.py` and the container anyway, so the rebuild costs little extra on top of the migration.
- **Merge PR #2 (walk by default, filename secret filter).** It ignores `.gitignore` by default. In ordinary repos that indexes generated code, vendored copies and Rust `target/`, which its skip list misses. It keeps `use_gitignore` as a dead setting "so configs still parse", which is a compatibility shim. Its `*token*` pattern drops legitimate source files. Its tests call a private method. D5 fixes the untracked-file bug with `--others --exclude-standard`, and serves the author's allowlist case with `use_gitignore=false`.
- **Store mtime on each chunk row instead of a files table.** A file with zero chunks would then have no record and would be re-indexed on every search.
- **Keep the weighted score blend and retune its constants.** The two scales move with corpus size: keyword scores never reach the top on a small repo and pile up at the ceiling on a large one. Constants tuned on one corpus fail on the next. RRF uses ranks only.
- **LanceDB's built-in hybrid query with `RRFReranker`.** It is less code, but it hides the per-channel ranks we return and test, and it gives no cosine similarity to threshold on. Our fusion is about ten lines.
- **Expose `vector_weight` and `min_score` as tool parameters** (issue #3). Agents cannot tune them meaningfully, and RRF removes the weight. `min_similarity` becomes a setting. Kind, language and path filters give agents control they can use.
- **Switch the embedding model now.** Without the harness we cannot tell a better model from a different one, and every switch costs users a full re-index.
- **`typing.Protocol` with `MagicMock` fakes** (current). See D2: missing members surface late, and fakes answer anything.
- **Keep two hand-synced instruction files** (ADR 002). They have already drifted (CLAUDE.md 66 lines, AGENTS.md 86, different content). A symlink plus a gate cannot drift.
- **Lower `requires-python`.** We found no 3.14-only construct, but uv provides 3.14 to `uvx` users, and the owner's stack is 3.14.
- **Background indexing now.** It needs the locks and the files table first. It stays in TODO.
- **Declarative tags queries instead of per-language classes now.** That is a spike for the language epic. Fixing the base helps both paths.

## Consequences

- Users upgrading to 0.5.0 get one automatic full re-index. Later model or chunker changes do the same, with the reason in the response.
- Tool output changes shape in 0.5.0 and 0.6.0. CHANGELOG states each break.
- The 0.4.x line stays pinned to `mcp<2`. From 0.5.0 we require `mcp>=2`.
- New dependencies: `filelock` (runtime); `import-linter` and `hypothesis` (dev). `pandas` leaves; torch may become optional after stage 7.
- Commits get slower: more hooks, test collection, the skill gate. CI adds minutes and a model download, which the cache softens.
- Search quality and speed become numbers we track (`evals/baseline/`), not impressions. TODO's search epic is re-ranked by them.
- Owner actions: approve each release tag; add the required reviewer on the `pypi` environment; read and approve the GitHub replies in stage 9.

## Appendix A. Findings inventory

IDs: **G** = GitHub, **R/A/S/Q/C/T/P** = code review (reliability, architecture, speed, quality, code, tests, product), **X** = comparison with sibling projects, **O** = found while writing this ADR. "Verified" means reproduced by script, read in code or run. "Suspected" means reasoned but not run.

| ID | Finding | Evidence | Resolved in |
|---|---|---|---|
| G1 | Fresh install crashes on mcp 2.x (`mcp.server.fastmcp` removed) | verified: mcp 2.3.0 raises `ModuleNotFoundError`; PyPI 0.4.0 requires `mcp>=1.26.0` | D1, stage 0; D2, stage 4 |
| G2 | Untracked files never indexed | verified (R2); PR #2 | D5, stage 1 |
| G3 | Prose (docstrings, Markdown) outranks code | issue #3 eval; verified on this repo (Q5) | D9, stage 6 |
| G4 | `vector_weight`/`min_score` not exposed | `server.py` signature | D9 (setting, RRF) |
| G5 | Doc-heavy repos drown code; no "code only" switch | issue #3 (288 md vs 98 py) | D9/D10 filters, stage 6 |
| G6 | PR #2 approach (walk default, secret filter) | read the diff | Alternatives; D5; reply in stage 9 |
| R1 | Undecodable file aborts indexing of the project, on every search | verified | D6, stage 1 |
| R2 | git scan: untracked invisible, deleted-but-tracked re-indexed forever, non-ASCII names broken | verified | D5, stage 1 |
| R3 | Missing `git` or timeout raises instead of falling back | verified (`FileNotFoundError`) | D5, stage 1 |
| R4 | Comment between decorator and `def`/`class` drops the definition | verified (0 chunks) | D6, stage 1 |
| R5 | Rust generic and scoped impls dropped, `mod` bodies unread, trait impls misnamed | verified | D6, stage 1 |
| R6 | Python files without top-level defs yield no chunk | verified | D6, stage 1 |
| R7 | Markdown `line_end` one past EOF; tests enshrine it | verified (3-line file → 4) | D6, stage 1 |
| R8 | Concurrent `index()` duplicates rows; no lock in or across processes | verified (8 rows for 6) | D8, stage 5 |
| R9 | Model change corrupts (same dim) or crashes (other dim); no manifest | verified (768-dim crash) | D4/D7, stage 5 |
| R10 | Errors returned as success (`isError: false`); union return wrapped in `{"result": ...}` | verified via in-memory client | D3/D10, stage 4 |
| R11 | mtime recorded after indexing: edit in between lost | suspected | D3 `SourceFile`, stage 5 |
| R12 | `index_status` loads the model (~4 s) on the event loop | verified | D7, stage 5 |
| A1 | Pass-through layers (`Indexer`, Connection/Store split), incomplete protocols, unused mock fixtures | verified | D2, stage 4 |
| A2 | Four module globals; deferred import in `app.py` | verified | D2, stage 4 |
| A3 | Blocking work on the loop (status ×2 per search, drop table, model load) | verified | D8, stage 5 |
| A4 | Two sources of truth (`file_mtimes.json` and the table) | verified | D4, stage 5 |
| A5 | No stable chunk identity; merge on `(file_path, line_start)` | verified | D3 `chunk_id`, stage 5 |
| A6 | `models/responses.py` imports `services`; formatting in models; dataclass among pydantic | verified | D3, stage 4 |
| A7 | Dead `except ValueError`; raw `RuntimeError` from writes; dead branch in `get_status` | verified / suspected | D11, stages 4–5 |
| S1 | Embedding is ~99% of indexing; 1–3 files/s against a 50 files/s target | verified | D7/D12, stages 5 and 7 |
| S2 | Full-text index rebuilt per write; per-file delete transactions; no version cleanup; index grows per edit | verified | D4, stage 5 |
| S3 | Warm search 100–200 ms, mostly status; pandas for row iteration | verified | D4/D8, stage 5 |
| S4 | Cold start 1 s + 3–4 s model; 571 MB RSS; numpy → list → Arrow round trip | verified | D3/D7, stages 5 and 7 |
| S5 | "Parallel chunking" gains nothing | verified | D6, stage 5 |
| Q1 | 13–30% of chunks (34–62% of classes) exceed the 256-token window; dead token settings | verified | D6 skeletons, D7 model, D11 |
| Q2 | Class bodies embedded and returned twice | verified | D6 skeletons, stage 5 |
| Q3 | Blended score on incomparable scales; nonsense query returns results | verified | D9 RRF, stage 6 |
| Q4 | Hybrid improves MRR 0.699 → 0.745: keep the keyword channel | verified | D9, Kept |
| Q5 | Docs and tests crowd out code; no filters | verified | D9/D10, stage 6 |
| Q6 | Path/name header in embedded text did not help (MRR 0.654); no eval harness | verified | D12, stage 3 |
| Q7 | MiniLM is a weak choice for code | suspected | D7, stage 7 |
| Q8 | File grouping breaks score order; recency boost untested | verified | D9, stage 6 |
| C1 | Rust and Markdown build `Chunk` by hand; repeated cases | verified | D6, stage 5 |
| C2 | Duplicated row → result conversion with two score formulas | verified | D4/D9, stages 5–6 |
| C3 | Short names (`t0`, `r`, `f`, `e`, `df`…), `P`/`T` type vars | verified | D14 gates, stage 4 |
| C4 | Nested `_progress` closures | verified | D8, stage 5 |
| C5 | Tuples as types (`tuple[list[str], int]`, `list[tuple[SearchResult, float]]`, tuple keys) | verified | D3, stage 4 |
| C6 | Magic literals (progress %, `// 4`, 50 lines, timeout 10, `[:16]`, defaults, `"content"`) | verified | D11/D14, stage 4 |
| C7 | Dead code and settings (`embedding_device`, token settings, `ScanPlan.all_files`, cache methods, `embedding_dim` unused) | verified | D3/D4/D11, stages 4–5 |
| C8 | Stale comments ("tantivy", "Scans Python files", lazy-load promise) | verified | D10/D9, stages 4–8 |
| C9 | `./profiles` in user's project; untyped `_save_profile`; `SIGINT` lambda; `.env` from CWD; `type: ignore`s | verified | D11, stage 4 |
| T1 | Assertion-free tests | verified | D13, stage 3–4 |
| T2 | Tests fitted to output or implementation | verified | D13 |
| T3 | No git-repo tests, no server tests, no container/fusion/encoding/concurrency/manifest tests | verified | D13, stages 1, 3, 5 |
| T4 | `time.sleep` for mtimes | verified | D13 |
| T5 | Unused fixtures; test names do not match modules; nested test helper | verified | D13 |
| P1 | Tool surface lacks filters, debug opt-in, limit cap; payload ~2k tokens for 5 hits | verified | D10, stage 4–6 |
| P2 | Config: 3 dead settings, 5 of 9 documented, no ignore control in git mode, duplicate skip lists, `.semantic-code/` not ignored | verified | D5/D4/D11/D15 |
| P3 | Adding languages inherits base holes; tags-query spike | — | D6 |
| P4 | Docs drift (ADR 001, CHANGELOG versions, TODO headers, README `ty` claim, unmet targets) | verified | D15, D12 |
| P5 | No CI; ruff drift; `requires-python`; 1.7 GB install | verified | D14, D11, D7 |
| X1 | No CI; release does not wait for tests | verified | D14, stage 2 |
| X2 | No owner approval before PyPI | verified (`protection_rules: []`) | D14 (owner action) |
| X3 | Gates that pass without checking (`ty \|\| true`, ruff mirror, bandit 1.8.6, no test collection, no `default_stages`) | verified (bandit run) | D14, stage 2 |
| X4 | No skill, no plugin; docstrings incomplete | verified | D16/D10, stage 8 |
| X5 | No skill ↔ server gate | verified | D14/D16, stage 8 |
| X6 | Two instruction files; no size/path gate; stale and conflicting rules | verified | D15, stage 2 |
| X7 | Owner's code rules not enforced by tools | verified | D14, stage 4 |
| X8 | No layer contracts | verified | D2/D14, stage 4 |
| X9 | Module globals instead of lifespan; Protocol vs ABC | verified | D2, stage 4 |
| X10 | No error hierarchy; per-tool `path.exists()` checks | verified | D11/D10, stage 4 |
| X11 | Test layout, no e2e, no property tests, MagicMock fakes, `test_searcher.py` tests `SearchService` | verified | D13, stages 3–4 |
| X12 | No LICENSE/metadata, CONTRIBUTING, SECURITY, PR template; TODO convention; `ship.md` model name; ADR 001 date | verified | D15, stage 2 |
| X13 | No reviewer agent | verified | D15, stage 8 |
| O1 | Full-text search does not split identifiers (`getUserName` misses snake_case and word queries) | verified | D9 `search_text`, stage 6 |
| O2 | mcp 2.x offers `instructions=`, `middleware=`, `ListRoots` | verified on 2.3.0 | D10/D11, stages 4 and 8 |
| O3 | Walk follows symlinked directories (good catch from PR #2) | read | D5, stage 1 |
| O4 | The issue author's speed (391 files in ~7 s) is 20× ours on CPU: likely GPU/MPS | suspected | D7; ask in stage 9 |
| O5 | Work tree detected by `(project / ".git").is_dir()`: worktrees, submodules and subdirectories fall back to the walk without `.gitignore` | verified in code | D5, stage 1 |
| O6 | A subdirectory as `project_path` builds a second, separate index | verified in code (`config.py` hashes the resolved path) | D10, stage 4 |
| O7 | A tracked symlink can pull files from outside the project into the index | suspected | D5, stage 1 |

## Appendix B. Baseline measurements (v0.4.0)

| Metric | This repo (61 files) | pydantic (105) | numpy (407) | Target (rules) |
|---|---|---|---|---|
| Embedding throughput, chunks/s | 29.5 | 26.1 | 27.5 | — |
| Indexing, files/s | 3.0 | 1.3 | 1.0 | > 50 |
| First `search_code` (cold, full index) | 20.4 s | 78 s | ~7 min | — |
| Warm `SearchService.search` | 63 ms | 42 ms | 110 ms | < 100 ms |
| of which `get_status` | 29 ms | 15 ms | 90 ms | — |
| Warm call through MCP client | 97 ms | — | — | — |
| Full-text index rebuild per write | 112 ms | 177 ms | 1005 ms | — |
| Re-index of one edited file | 300–400 ms | ~200 ms | 800–1200 ms | — |
| Index growth per single-file edit | 0.2 MB | 0.53 MB | 2.35 MB | — |
| Token length p50 / p90 / max | 108 / 289 / 4068 | 117 / 603 / 27315 | 148 / 642 / 44291 | window 256 |

- Startup: `create_app()` 1.0 s and 178 MB RSS; model load adds 3–4 s; RSS with the model is 571 MB (target < 500 MB); cold start target is < 3 s.
- Torch threads on 4P+8E: 4 threads give 45.5 chunks/s, 12 (the default) 31.6, 16 give 22.9; `max_seq_length=128` gives 39–45.
- Search quality, 14 queries on this repo: vector-only MRR 0.699; current hybrid MRR 0.745; path/name header variant MRR 0.654.
- Issue #3 (private repo, 10 queries; right file top-1 / top-3; right code chunk top-1 / top-3):
  - as published: 0/10 on all four;
  - with untracked files indexed: 3, 7, 2, 4;
  - with docs also excluded: 4, 9, 3, 7.

## Appendix C. Skill outline (`skills/semantic-search/SKILL.md`)

Written in stage 8 against the final D10 surface. Every fact below is checked by `scripts/check_skill.py` or re-read against the code before release.

1. **Which tool.** A table: exact identifier, string or regex → Grep; every call site → Grep (search_code returns the top N only); a known path → Read; behaviour without a name → search_code; a concept spread over files → search_code, then Grep the names it reveals; an unindexed language → Grep. The loop that works: search → Read the range → Grep the names.
2. **Before the first search.** `project_path` is the repository root, absolute: a subdirectory gets its own empty index (`git rev-parse --show-toplevel`). The first call loads the model and may index, with progress; do not retry while it runs. If the tool is missing, tell the user how to install the plugin; do not install it yourself.
3. **Phrasing a query.** Describe what the code does, as a docstring would; one concept per query; the domain's nouns; three to ten words; no file names. Pairs of examples that work and that do not.
4. **Filters.** `chunk_kinds` (Rust types count as `class`), `languages`, `path_glob`/`exclude_glob` (for example, excluding `tests/**` or `docs/**`), `limit` and its cap.
5. **Reading the answer.** Every result field. Results come in rank order. `truncated` means Read the range before quoting. `similarity` versus rank. `include_debug` only when a search was slow or odd.
6. **The index.** Automatic incremental re-index on every search; `index_codebase(force=true)` after mass renames or when results point at code that no longer exists; `index_status` to check the root and the manifest, not before every search; where the index lives; automatic rebuild after a model or version change; `use_gitignore=false` for allowlist repos.
7. **Traps already met.** A `project_path` one level too deep; exact identifiers ranking below paraphrases; queries naming a file; a module docstring or class skeleton landing first, with the body in the next hit of the same file.
