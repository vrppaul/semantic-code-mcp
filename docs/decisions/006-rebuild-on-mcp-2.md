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

Release 0.4.1 with `mcp>=1.26.0,<2` and the correctness fixes from D5 and D6 on the current structure, each with a test that fails before the fix. (The plan was a 0.4.1 with the pin alone and a 0.4.2 after; both were ready before either shipped, so they ship as one.) The chunkers keep their files through the rebuild, so these fixes carry over. The scanner fix moves into `discovery/` in stage 4. Not released: the owner releases nothing until every stage is done (2026-10-26), so these fixes ship with the rebuild.

### D2. Layers and composition root

```
semantic_code_mcp/
  domain/       models, ABC interfaces — imports nothing but pydantic and the stdlib
  discovery.py  git ls-files / filesystem walk → list[SourceFile]
  chunkers/     tree-sitter chunkers on a shared ChunkCollector
  embedding.py  sentence-transformers adapter (model loaded in a thread on first use)
  storage/      one LanceDB store: chunks, files, manifest
  services/     indexing.py — plan → chunk → embed → write, under a per-project lock
                search.py   — retrieve → fuse → filter → shape
                projects.py — the registry of each project's services
  server/       MCP tools, responses, presenters (domain → response), profiling middleware
  app.py        composition root and lifespan
  cli.py        entry point
```

(As built in stage 4: `chunkers/` keeps its name, discovery and embedding are one module each. A domain error hierarchy waits for its second kind — today the one failure the agent hears about, a missing project path, raises `ToolError` at the edge.)

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
- **`uv run python -m evals speed`** measures the model load (a search of a one-file project: the load, a one-file index and one query), a full index with the model already loaded (files/s, chunks/s), warm search p50/p95, the first search of a freshly started server on an up-to-date index (process start and model load), the server's peak memory (Linux), and per single-file edit the index growth and the search that re-indexes the file before it answers. With 15–20 questions a corpus, p95 is close to the slowest search.
- **Answers are checked before a run:** every answer's file must exist in the checkout and its chunk name must be one the server's chunker gives that file; a run fails otherwise. Results keep each question's top hits, so a miss can be read without running the model again.
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
- **Direct dependencies are bounded below their next breaking release** (next major; for a 0.x, next minor; structlog, which numbers its releases by year, below the next year's), from the version the suite and the harness ran on. `uvx` resolves a fresh environment and ignores `uv.lock`, so an open bound ships whatever is newest: 0.4.0 broke on mcp 2.x that way, and on 2026-10-26 a fresh install would have taken lancedb 0.40, sentence-transformers 6, filelock 4 and pyarrow 26 against the 0.27, 5, 3 and 23 the tests ran on. The package is an application in an environment of its own, so a bound costs no user a conflict. The bounds hold only direct dependencies: a fresh resolve within them still takes transformers 5 and huggingface-hub 2 against the lock's 4.57 and 0.36 (a review's probe). So the lock moves to the newest versions within the bounds before a release, and the suite and both evals run on it; with stage 8's CI, a weekly job resolves the package as `uvx` does and runs the suite, and the release job refuses a lock that a fresh resolve would change. A bound moves only with a run on the new line.

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
- The CPU-only torch index and the platform install notes, until stage 7 removed torch.
- `paths:` frontmatter on rules.
- The `pypi` environment, now with protection.
- Hybrid search as a method (Q4).
- Module docstring chunks (ADR 003) and heading sections (ADR 005), with the prior decided by the harness.

### Decisions taken for the owner

The project rules say to ask before changing the storage schema, the tool signatures or the dependencies. The owner delegated these on 2026-10-25 ("тебе лучше знать"):

1. **Storage schema changes** (D4), with one automatic rebuild through the manifest. A migration path would cost more than one re-index.
2. **Tool signatures change** (D10): new filters, `include_debug`, `ToolError`, no `ErrorResponse`, no grouping. This breaks anyone parsing the old output; CHANGELOG says so.
3. **Dependencies:**
   - added: `filelock` and `numpy` (runtime; numpy was already installed through sentence-transformers and LanceDB, and is declared once the domain names the vector's type); `pylance` 4.0 (runtime, stage 6), the Lance library LanceDB builds its own core from, for an upkeep LanceDB does not offer: LanceDB 0.30.2 builds in Lance 4.0.0 and pylance 4.0.2 writes the same tables, while LanceDB 0.30.0 built in Lance 3.0, so LanceDB is held at 0.30.2 exactly and pylance at 4.0.2 or a later 4.0; it costs a 60 MB download, 160 MB installed, and 17 MB of a process's memory on import, 59 MB after a fold (a review's probe); `huggingface-hub` (runtime, stage 7), installed through sentence-transformers before, declared once the embedder turns its progress bars off; `import-linter` and `hypothesis` (dev); vulture, deptry and typos as pre-commit hooks;
   - removed: `pandas`; sentence-transformers and torch (stage 7), as the embedder runs the model's ONNX export itself;
   - moved: lancedb from 0.27 to 0.30.2, whose filtered full-text search keeps a narrowed search's matches, and no further: 0.32 to 0.40 scan vectors two to three times slower (stage 6); every direct dependency bounded below its next breaking release (D14);
   - added in stage 7, on its measurements: `onnxruntime` and `tokenizers`, the embedder's only libraries (stage 7 below).
4. **Nothing is released until every stage is done** (the owner, 2026-10-26), the hotfix 0.4.1 included: one release ships the whole rebuild, and its version is settled then.

### Stages

Progress: stages 0–6 are done on branch `rebuild-on-mcp-2` (0–4 on 2026-10-25, 5 and 6 on 2026-10-26); stage 7 is under way. Stage 4 also landed the D10 parts that need no new search features (`ToolError`, one return type per tool, `include_debug`, the `limit` cap) and D11's settings clean-up; D3's new chunk fields wait for the storage schema of stage 5.

As built, stage 4 differs from D2, D7, D8, D10 and D11 in these points:
- `cli.py` builds `Settings` and configures logging; `app.py` builds the server, and its lifespan yields the `ProjectRegistry` itself rather than an `AppContext` around it.
- The tools check the project path at the edge (absolute, existing, a directory, resolved) and turn the failure families of the parts (`OSError`, `RuntimeError`, `ValueError`) into `ToolError` with their cause; mcp 2.x drops the message of anything else. A domain error hierarchy still waits for its second kind.
- Reads take no lock. A forced run empties the table instead of dropping it, so a reader sees an empty or a full index, never a missing table; stage 5 replaces the index at write time instead, after embedding.
- The model loads in a worker thread right after its own progress report; an index run inside a search reports within the search's band of progress.
- Property tests (hypothesis) move to stage 6, with the fusion they are most needed for. The old chunker tests filtered in comprehensions; stage 6 rewrote them.
- One index per repository (D10, O6) is not built: the tools resolve the path and nothing more. It moves to stage 6, with the path prefilter it needs to keep a subdirectory's results to that subdirectory.

Stage 5 is done (2026-10-26). As built, it differs from D2, D3, D4, D6, D7, D8 and D11 in these points:
- `embed_documents` takes no progress callback: the indexing service embeds in slices of `EMBEDDING_SLICE_SIZE` chunks and reports before each, so the embedder stays a plain call in a worker thread and progress stays on the event loop.
- The domain imports numpy for one thing, the type of a vector (`Vector`, `VectorMatrix` in `domain/interfaces.py`); numpy is a declared dependency.
- The manifest holds an `IndexIdentity` (model name, the revision of its weights, chunker version), the store's schema version and the time of the last write, not the vectors' dimension: the model's name and revision fix it and the vector column records it, so reading a manifest never needs the model. The store writes it on every write. The chunks table is made by the first write that has vectors, as wide as they are.
- A project's index directory holds an index per identity and schema version, each in a directory of its own (`schema-<version>-<digest of the identity>`), instead of one index that D4 drops and rebuilds on a mismatch. Two sessions of different models or versions on one project would otherwise rebuild it in turn on every search, and a user who tries a model and switches back would pay two full indexes; now each keeps its own, and nothing is rebuilt in place. An index no session read or wrote for seven days is removed, and so is 0.4's layout (`chunks.lance`, `file_mtimes.json`), a week after 0.4 last wrote it. A write renames a new manifest into an index's directory and a read touches a stamp there once the stamp is an hour old, so an index's age is the newest change to its directory or to what lies directly inside it; the stamp's own mtime keeps the hour, on the wall clock the remover reads, since a monotonic clock stops while a laptop sleeps and a session waking after a week would not mark its index before another removed it (a review's probe). Each store whose index exists looks for unused indexes as often as it marks its own, on a read as on a write, so an index of a version nobody runs any more goes even if the project is never written again. A writer makes its index's directory under its write lock, so a removal never takes it between the two. A store removes another index only after taking its write lock without waiting, reading its age again under the lock, and renaming its directory out of every store's reach: a review's probe showed that a removal during the owner's write left records without chunks, which nothing indexes again. The write lock lies beside the index's directory, not in it, so the rename works on Windows too, where an open file pins its directory [not tested on Windows]; lock files are never deleted, since a writer may be waiting on one. The project's index directory, not each index, holds the `.gitignore`, so what 0.4 left in `.semantic-code/` is hidden too. With it go D11's `IndexIncompatibleError`, the identity check of a write, its replan, and every `RebuildReason` but `incomplete`, which went with the writes per slice below: a review of `2243d0f` found that all of them, with the records-first order, compensated for this one cause. A new index does not answer from the old one while it builds, as first proposed: a search waits for its index run, and another model's vectors cannot answer this model's query; background re-indexing stays in TODO.
- No content hash per file, against an edit inside the scan's timestamp granularity that keeps the size (git's "racily clean" case): Linux 6.13 and later stamp a file stat'ed since its last write with a fine-grained mtime, and APFS and NTFS keep sub-microsecond stamps, so the window is at most a tick of the system's clock [not measured], while a hash means reading every file on every status. Taken up if a missed edit is ever reported.
- Upkeep keeps old versions for 10 seconds, not D4's `cleanup_older_than=timedelta(0)`: readers take no lock, and a reader failed when upkeep deleted the files of the version it was reading (3 to 7 failures in 60 rounds at 0; none at 5 s).
- Upkeep (`optimize`) ran once a table had 20 versions since its last upkeep, and at once after an index run of more than one write or a forced one, which rewrites every row (`IndexStore.keep_up`), not after every write; since stage 6 every index run that wrote ends in it. On click a delete-and-add write took about 15 ms, `optimize` about 250 ms, and the full-text rebuild it replaces 560 to 790 ms. LanceDB 0.27's full-text search reads rows its index does not hold yet and drops deleted ones (verified), so the index was made once and upkeep kept it current; since stage 6 a store also compacts the tables and rebuilds it whole at its first upkeep and once in 20 after.
- A run writes a slice of whole files at a time, at least `EMBEDDING_SLICE_SIZE` chunks but the last, not once after embedding everything: a run stopped midway (a client that hung up during a first index of minutes) keeps the slices it wrote, and the next run indexes the rest. A slice's chunks take the place of what the index held of its files, and of the files removed, which go with the first slice, in one `merge_insert` keyed by `chunk_id` with `when_not_matched_by_source_delete` filtered to those paths: one commit, so a reader finds a file's old chunks or its new ones, never neither; a chunk whose id is stored already is updated where it stands, its lines and parent new. On click's tables a merge took about 21 ms against 7 ms for a delete and an add (probed). The records follow the chunks, so a write stopped between the two leaves the old records, which the next run reads as changed files. A forced run replaces file by file too, so no reader ever sees an empty index, and `WriteScope`, `RebuildReason` and the `rebuild_reason` fields go: nothing is rebuilt whole any more.
- A chunk whose id the index holds keeps its stored vector, so an edit embeds only the chunks whose text changed; a forced run embeds everything again, for a model replaced under a name nothing pins. A slice whose stored vectors are of another width than those it embeds now embeds them again, after the rest, and its write starts the index again.
- A run plans its deletions from the paths the chunks table holds as well as from the records (one column read, no vectors), so the chunks of a file deleted after a write stopped between its chunks and its record leave the index; nothing else would remove them (a review of `8c7ae9c`). A status reads it too, or a file deleted after such a crash would go on answering searches with nothing to start a run. Made a `Path` once per distinct path, not per row, the read costs 2.5 ms at click's 1.7k chunks in process (a review measured 35 ms at 100k); through the server status p50 went 32 / 20 / 28 → 39 / 26 / 30 ms on click / fd / this repo, against 43 / 31 / 32 with a `Path` per row. At scale the read costs 7.7 ms at 20k chunks and 29 ms at 100k (a review's probe). Open: a marker the write lays in the index's directory before its chunks and lifts after its records would let status read the column only while one is left, at a few microseconds a status.
- Progress: embedding and writing share the band from `IndexPhase.embedding` to the end; the model loads inside it, reported where the first chunk to embed stands.
- The registry builds a project's services without touching the disk, so no `await` sits between its lookup and its insert; the store connects on first use and makes its directory only on its first write. `app.py` builds the store and the chunkers and hands them in.
- One lock guards a write: the store's file lock, taken by a new `FileLock` per write, which keeps out another process and another thread of this one alike, a cancelled request's thread among them. The service holds no thread lock of its own.
- The files table holds the scan's records (`SourceFile`), never stat'ed again at write time, so an edit or a deletion during a run is seen by the next one. It holds no chunk count: nothing read it.
- Reads open their table again on every call, which sees another process's writes (tested with two stores on one directory); no consistency setting is needed.
- The identity holds the revision of the model's weights, and each model the server knows loads at a pinned one (`PINNED_REVISIONS` in `embedding.py`; the default's is the revision the harness measured). A hub model's name follows its newest revision: this machine's cache holds two snapshots of all-MiniLM-L6-v2, fetched 2026-03-20 and 2026-10-25, the second by a plain load (only its README differs; the weights are one blob). A cache that holds only another snapshot fetches the pinned one's small files once, and offline it cannot load the model until it does. So newer weights never meet old vectors under one index: they would get an index of their own, built as any first index, under the indexing lock. Nothing compares widths at search time any more: a review of `3955cd8`, which did, found the check outside the indexing lock, so every search at once ran its own forced re-index; it missed a width change that came with an edit and answered from part of the index; and the slice it embedded again embedded its new chunks twice, with progress going back. A model nothing pins (a local path, a hub model the server does not know) has no revision and is trusted to stay what it was. Replaced at the same width, it is not seen. At another width, every search fails on the query's width (`query dim(...) doesn't match the column vector dim(...)`, through `ToolError`) until a run that embeds a chunk starts the index again, as below: a forced run, or an edit that adds text. A touch, a deletion, or an edit that only removes or moves code embeds nothing and writes the old vectors back. An edit's run starts the index from that file, so the search it ran for answers from that file alone, and the next search indexes the rest (a review's probe). Two names of one model, `all-MiniLM-L6-v2` and `sentence-transformers/all-MiniLM-L6-v2`, load the same weights at the same pin and still keep two indexes; and a directory of the model's name in the server's working directory would shadow the hub's, under the hub's pin. Stage 7's profiles can give each model one name.
- Vectors of another width in an index's directory (a model replaced under a name nothing pins) drop its records, then its chunks table, and the write starts the index again; the files it did not write are unrecorded, so the next run indexes them. No write is refused, so `domain/errors.py` goes with the identity check: D11's error hierarchy waits again for a failure the agent must act on. Two live sessions of one name nothing pins and two widths (a local model replaced while an old session runs) drop each other's tables whenever one embeds a chunk; between those writes, the session whose width the table does not hold fails every search on the width, as a `ToolError`. Rare enough to leave.
- A write that drops the tables for another width leaves a reader at that moment able to fail on the missing table (a review's probe: 45 of about 270 reads while the width alternated).
- A file name that is not UTF-8 is skipped with a warning: no column can store it, and one such name failed every run. The check runs after the ignore patterns, so an ignored directory never warns; it still warns once per scan, and status scans on every search.
- D3's chunk model: `Chunk(chunk_id, path, language, kind, name, parent, line_start, line_end, content)`, with `ChunkKind` in place of `ChunkType` and a `Language` enum. Paths are the file's from the project root, in POSIX form in both tables, refused if absolute; only the tools' answers make them absolute, so a project moved with its local index keeps it. `SearchResult` holds a `Chunk` and its score, and the hits of two searches meet by `chunk_id`. It differs from D3 in three points:
  - `chunk_id` digests the file's path, the text the model embeds and how many chunks of that text come before it in the file, not D3's path, kind, qualified name and start line: a line inserted above leaves every id as it is, and a vector stored under an id is the vector of its text, which the next chunk reuses.
  - No `embedded_text` yet: the model embeds `content` as it is, until D9's search text or a model's prefixes (D7) make the two differ; the id then digests the embedded text.
  - `SourceFile` carries no `language`: the composite chunker picks a file's chunker by its extension, and nothing reads a file's language before it is chunked.
- `parent` is the qualified name of what a chunk sits in, joined as its language writes it: `Outer.Inner` in Python, `outer::Stack` in Rust (modules, impls, traits), `Guide > Install` in Markdown (the headings above a section).
- `SCHEMA_VERSION` is 2, and `CHUNKER_VERSION` 3 (2 with the chunk model, 3 with an empty Markdown heading's fix): each names a new index directory, so nothing is migrated, and users of 0.4 still build one index on upgrade.

- D6's class skeletons are not adopted: the harness shows no gain. With a class's, impl's or trait's chunk holding its outline, each method's body cut to `...`, file MRR went from 0.455 to 0.427 on click, from 0.549 to 0.502 on fd and from 0.698 to 0.729 on this repo, and chunk MRR 0.328 → 0.32, 0.31 → 0.293 and 0.544 → 0.544. With 20, 15 and 16 questions those moves sum to −0.56, −0.70 and +0.50 in reciprocal rank: about one question moving a rank or two per corpus, so neither way is shown. A full index took 65.3 / 22.3 / 19.4 s against 68.0 / 23.3 / 20.7, inside click's spread of 68.0 to 72.9 s. The measured change is kept in the stash "D6 class skeletons: rejected on the harness" of the owner's clone, not in the repository. Q1 (34–62% of classes past the 256-token window) and Q2's double embedding and payload stay open for stage 7's 512-token models, measured on more questions.

Open from stage 5:
- An edit inside a method re-embeds its class's chunk too, which holds the method's text; the harness's edit appends a definition, which shows the reuse of vectors but not that cost.
- A first index of click took 71.2 and 72.9 s in two runs after writes went per slice, against 68.4 s before, and 68.0 s in a third; the store's seven writes account for under half a second (probed), so the spread is the run-to-run noise, wider than the 10% assumed. Peak memory on click spread from 1103 to 1269 MB over four runs, on fd and this repo within 30 MB; the cause is not found.
- The by-source delete makes each merge scan the whole table: one edited file's write costs 12 / 47 / 124 ms at 1.7k / 20k / 100k chunks against 5 to 8 ms for a delete and an add, and a first index of 100k chunks spends 52 s in its 385 merges, about 1.3% of its embedding (a review's probes). If that matters, upsert by `chunk_id`, then delete `path IN (...) AND chunk_id NOT IN (...)`: two commits, between which a reader finds a file's old and new chunks both, never neither. A first index could also skip the lookup of stored vectors for files the table holds no chunk of.
- Whether `when_matched_update_all(where=...)`, skipping the rows a write leaves as they were, saves index growth (0.10 MB per edit on click when stage 5 measured it).

Stage 6 is done (2026-10-26). Figures are chunk or file MRR on click / fd / this repo through the server unless said otherwise; each commit's message holds its own figures and checks, and the rejected builds live in the commit history. As built, it differs from D3, D9, D10 and D14 in these points:

- **Fusion.** Each channel brings `limit × CANDIDATES_PER_RESULT` (2) candidates from one open table, which keeps the version it was opened at, each with the cosine similarity of its stored vector to the query's, clipped to [-1, 1] (float32 oversteps by up to 5e-7). The services fuse them by RRF, k = 60, ties going to the closer in meaning (`services/fusion.py`; `Candidate` and `SearchCandidates` are domain models D3 does not name). File MRR 0.455 / 0.549 / 0.698 → 0.497 / 0.502 / 0.785. Results come in fused order, no longer grouped by file (file MRR → 0.539 / 0.585 / 0.825), `score` is `similarity`, and the recency boost is gone: the harness's checkouts give every file one mtime, so it could not show a gain.
- **The keyword channel** earns its place: with the search text's first build, the vector channel alone gave chunk MRR 0.380 / 0.363 / 0.492 against the fused 0.411 / 0.340 / 0.682 (probed in process); fd's chunk MRR is what fusion costs. It searches `search_text`: the content, then the name and the parent, each followed by the words of every identifier whose case changes inside (`HTTPServerError` → HTTP Server Error), split in one linear pass; LanceDB's tokenizer splits `_` and stems on its own. The parent gives its words alone, or every method crowds its class out of the channel for the class's own name. The model still embeds `content`. `SCHEMA_VERSION` went to 3 with the search text, 4 with the parent's words alone and 5 with stop words kept. Stop words are kept (`remove_stop_words=False`): by the criterion below they gained 2.1 in reciprocal-rank sum on the chosen questions, the held-out ones lost 0.5, the full-text index grows 6-8% and at 100k chunks a keyword search takes 1.5 ms more.
- **Prose is fused as code is.** A down-weight of sections (0.9, then 0.95) gained on keys that missed right answers; on pooled keys no weight wins (49.09 against 48.94 at 0.95), so `KIND_WEIGHTS` is gone.
- **Two floors and a name pass.** A hit no keyword matched stays from a similarity of 0.35, one whose words match from 0.25 (`min_similarity`, `min_similarity_when_words_match`, settings, as they suit one model's scale), and a hit named as the query, its case and the blanks around it aside, stays however far: fd's `main` and `correct` rank first in both channels at similarities of 0.09 and -0.04. One floor for both does not do: 0.35 drops 11 of 352 sampled names' chunks, 0.2 lets 16 or more results through to the off-topic questions. Eight far and seven near off-topic questions get 0 and 19 results over the three corpora. A keyword hit that shares only a stop word with the query gets the lower floor; giving it only to hits sharing another word drops one near off-topic result of 19 and moves no rank (probed): not built.
- **The floor rule**, binding stage 7: for each model, sweep pairs of floors (meaning-only, word matches; the name pass holds for every pair) by 0.05 over the model's own scale, from below the lowest similarity at which a chosen question's right answer reaches its first ten to above the highest any off-topic hit reaches, far or near, the floor of word matches no higher than the other (for this model 0.2 to 0.45: fd's near question on a zip archive reaches 0.423). Keep the pairs whose chosen code questions, chosen documentation questions and bare names each lose less than 1.0 in chunk reciprocal-rank sum against that model's best pair for that measure; among them take the lowest pair, ordered by the floor of word matches, then the other, whose off-topic results are fewer than 10 more than the fewest among them. With no pair passing, the pair that stood stays, and a new model takes the lowest pair of those best on code questions. This model's pair, 0.35 and 0.25, was chosen on a narrower grid (code 37.09 at each word floor, bare names 118.89 / 118.93 / 119.77 and off-topic results 41 / 19 / 14 at 0.2 / 0.25 / 0.3); the rule was written after those figures were seen and the held-out set is spent for floors, so stage 7 writes fresh held-out questions, and chosen ones whose right answers lie near the floors, and has their keys pooled and judged as below.
- **The harness** (`evals/`). Each question says its kind, code (50) or documentation (16), and is scored apart, and apart again by whether it holds a word of an answer's name or heading. Answer keys are pooled: every chunk any swept rule put in a question's first ten, about 1,000, judged blind by an Opus judge per corpus who read each in the checkout; 73 answers were added. An answer names its parent where chunks of several parents in its file share its name; a run refuses an answer whose file, name or parent no chunk of the server's own discovery and chunker matches. A bare name is asked of every fifth function, method or class name outside the tests, its own chunk scored; eight far and seven near questions no corpus answers count the results they get; seven questions a corpus are held out, printed only with `--held-out` and never read while a rule is chosen. The criterion for a search rule: the sum of chunk reciprocal ranks of the chosen questions of both kinds over the three corpora; a rule is a candidate when it beats the plainest search by 1.0 or more and neither kind, nor either wording group of a kind, loses 1.0 or more, the bare names lose less than 1.0 and the off-topic questions get fewer than 10 results more; among candidates within 1.0 of the best, the simplest wins. A held-out loss of 1.0 or more is reported and sends the choice to a review, not to another sweep.
- **Prefilters.** `search_code` takes `chunk_kinds`, `languages`, `path_glob` and `exclude_glob`, held in a `SearchQuery`'s `SearchScope`, which the store turns into one filter both channels apply before they cut. LanceDB 0.27's filtered full-text search lost matches: over the corpora 6 of 90 filtered keyword searches returned another set than the unfiltered ranking cut to the filter (a review's sweep: 12 of 324, dropping matches scored up to 14.1, searches asked from a subdirectory among them), and it lost a chunk of the scope that one word of the query matched while chunks outside the scope matched the others. LanceDB 0.30.2 keeps them (86 of 90 the same set, the rest one tie at the 20th place), and the scope test holds both that case and a scope's chunk ranked below the cut, on every kind of scope. 0.32 to 0.40 keep them too but scan vectors two to three times slower (a flat search of 100k rows: 50-90 ms on 0.27 and 0.30, 145-190 ms on 0.32 to 0.40; at 100k rows the store's unnarrowed search takes 57-97 ms on 0.27, 56-86 on 0.30.2 and 150-163 on 0.40, and one file's edit 210-360 ms on 0.27 against 100-170 on 0.30.2 and 0.40), so the package is held at 0.30.2. Filtering after the cut kept the right set in 7 of 90.
- **Globs** are read once, in `domain/globs.py`, as in a `.gitignore`, from the directory searched: a name without a slash at any depth below it, never the directory's own name; a leading `/` or an inner slash a path from it; blanks, `.` parts and doubled slashes go; `""`, `/`, `.` and `./` narrow nothing. A glob holding `..`, a character class, braces, an escape or a leading `!`, or longer than 1000 characters, is refused with a tool error saying why, before git is asked anything. Classes are refused because Rust's regex reads them apart from a glob (`--` as set difference, `[:alpha:]`, `/` inside a class). The store turns a glob into a regex for `regexp_like`, its characters escaped as Python's `re.escape` writes them, which Rust's regex reads alike (a property test holds that LanceDB keeps the paths Python's `re` keeps). Matching globs in Python over the indexed paths and handing LanceDB the set would read classes too and cost 51-63 ms against 99-105 at 100k chunks for small sets, but the set grows with the paths it names; it is the way to take up a narrowed search's cost if that matters.
- **One index per repository** (O6). Every tool asks the registry where the directory it was given lies (`ProjectRegistry.locate`, git in a worker thread) and works on the root of the work tree around it; a search asked from below the root keeps to the directory through `SearchScope.directory`, a `starts_with` prefix both channels filter on (29 ms against 40 for a regex at 100k rows). The directory is a project of its own outside a work tree, where the repository ignores it, and where the work tree's root is the user's home directory or above it, as a home under git for dotfiles is. Reading `status.showUntrackedFiles` instead split monorepos that set it for speed. A dotfiles repository rooted elsewhere that holds projects still climbs to its root, and a home that is itself a project (a container with `HOME` at the code's root) gets an index for each directory a tool names. Each call asks git 1.4 ms more at a root, 2.7 below one.
- **Upkeep after every run, and a compaction now and then.** Rows the full-text index has not folded in score on another scale, on LanceDB 0.30.2 a lower one (11.59 against 12.04 for one text), and the stage-5 rule left them out for up to 20 writes: a review's probe rewrote the files that answer the chosen questions, one index run each, through the server, and code reciprocal-rank sums fell 12.10 → 9.17 on this repo and 9.11 → 6.66 on fd after 18 runs, documentation 5.50 → 3.34. So every index run that wrote now ends in upkeep. LanceDB's `optimize` compacts at every call, merging every fragment under a million rows, so it rewrote the whole chunks table after each edit of a file other than the last: at 100k chunks an upkeep took 669-672 ms and the index peaked at 3.5 GB (probed). Upkeep goes through pylance instead. Each one folds the run's rows into the full-text index (`optimize_indices`) and prunes versions older than 10 s; what a version replaced stays on disk until an upkeep at least 10 s later, so an idle index keeps its last run's copies. At a store's first upkeep, as it cannot know what an earlier process left, and once in `UPKEEPS_PER_COMPACTION` (20) after, it first compacts the tables into fragments of up to `ROWS_PER_FRAGMENT` (4096) rows and rebuilds the full-text index whole, so a session's first search after an edit pays that (2.3-2.4 s at 100k chunks, below). Between, each fold adds the index a part none merges (3 files an upkeep on click; `num_indices_to_merge` merges none), the rows an edit replaced still count in its word statistics (after three edits of click's largest files 3,492 scores of three questions stood off a rebuild's, by up to 0.54 of about 12), and a run's fragment stays apart from folded ones. Through the server that moves little: the probe above gives this repo's code and documentation 12.0 / 5.5 → 12.5 / 5.5, 12.5 / 5.0 and 12.0 / 5.0 after 4, 8 and 18 runs, and fd's code 9.11 → 8.96, 9.04 and 8.88, up as often as down. The worst case was one large file edited again and again in one session: a review's probe appended to click's `core.py` (161 chunks) 19 times and compared every question against a compacted and rebuilt copy; the first ten differed for 8 of 32 questions after one edit, 21 after five and 24 after 19, while 4 answers moved one place (code 19.98 → 19.14, documentation 4.75 → 5.25). Lance drops a fragment whose rows were all replaced, so the table's own count of deleted rows misses such edits, while their words still weigh in the index (one row's score fell 1.06 → 0.62 after five, and came back to 1.09 after a rebuild). The index's own statistics count them (`num_docs` against `num_indexed_rows`, 7 ms to read at 100k chunks), so upkeep also compacts at once when they pass 5% of the rows (`REPLACED_ROWS_SHARE_BEFORE_COMPACTION`): on click, appends to `core.py` now compact at each edit and leave every question's first ten as a rebuilt copy's over six edits; appends to `_compat.py` (46 chunks) compact at every second, and between, the first ten of 4 of 32 questions differ, with no answer moved (the review's probe). On the corpora that rule compacts more often, as an edit of one of their files often replaces 5% of the rows: the index grows 2.4 / 1.0 / 0.89 MB an edit against 1.38 / 0.55 / 0.47, and a search after an edit takes 172 / 150 / 163 ms, the slowest 365 / 244 / 226 (`evals speed` at `9b3bcb9`). At 100k chunks an upkeep now takes 69-77 ms, the check included, one that compacts 2.3-2.4 s against 2.7-2.9 for `optimize` with the rebuild, and the index peaks at 0.85 GB; a search takes 40 ms against 54, as a flat vector search reads the fragments side by side (33-43 ms at 4096 rows a fragment, 50-74 at 8192, probed). On the corpora, with edits in turn, a search after an edit took 200 / 149 / 166 ms against 229 / 200 / 181 under `optimize`, the slowest 403 / 264 / 286, which pays the compaction, and the index grew 1.38 / 0.55 / 0.47 MB an edit against 7.3 / 2.5 / 2.3, the versions kept for 10 s included (`evals speed` at `f72d93c`, before the 5% rule above). A fresh index has ended in a rebuild since `133747e`, which took this repo's "MCP tool that searches code semantically" from 10th to outside the first ten (code chunk MRR 0.807 → 0.800; `evals quality`, the same on `9438ff7`'s code in two runs); no other of the 66 questions moved. Readers stayed safe through `optimize` and rebuilds (about 22,000 searches beside them, none failed or wrong, a review's probe); pylance's fold, compaction and pruning are the steps `optimize` runs, and a review's probe found none of about 16,700 searches on click beside 92 compactions, nor of 675 narrowed ones at 100k chunks, failed or came back short. The servers' peak memory in `evals speed` went from 1,073 / 1,000 / 987 MB to 1,155 / 1,023 / 1,029, about what pylance adds (stage 5 saw click's spread 170 MB from run to run).
- **Property tests** (D13) hold fusion (a permutation of the union; a match added never lowers a hit), similarity's range, a glob's reading, every chunk's line range holding exactly its content, and every discovered path being a file inside the project.
- **Dependencies** are bounded below their next breaking release (D14).

Weighed and turned down, so nobody brings them back: a class-grammar parser for globs; the `showUntrackedFiles` rule; a word-set name pass (the exact name lets no off-topic result through); one floor for both; the prose weight; a floor of word matches on words other than stop words; filtering the keyword channel after its cut; upkeep at most once per a clock's window (it leaves an edit's rows out exactly when an agent searches right after editing); compaction at every upkeep, bounded as above (scores always a rebuild's, but 300-430 ms an upkeep at 100k chunks, most of it remapping the full-text index); compaction with that remap deferred (23-83 ms, but it leaves a fragment-reuse index pylance gives no way to clean up); compaction decided by the table's state alone, which would spare each session's first compaction: the replaced rows show in the full-text index's own count, which upkeep reads, but the parts each fold adds show in no statistic of LanceDB's or pylance's, and the table's fragments do not count them, as a file edited again and again leaves their number as it was, so a store still counts its upkeeps; the files of the index's directory show them, which the tests read, but that is Lance's layout on disk, not a promise; D6's class skeletons (stage 5).

Open for stages 7 and 8:
- The floors of each model by the floor rule, on fresh held-out questions; profiles that carry each model's revision, prefixes and floors, held whole by `IndexIdentity`, which today names the model and revision only (backend, quantization and prefixes change the vectors too; the versions of onnxruntime and tokenizers could as well, which the embedder's reference test would catch: stage 7 below).
- A narrowed search costs more at scale (at 100k chunks on LanceDB 0.30.2, in process: 56-86 ms unnarrowed, 142-187 for two kinds and `*.py`): the speed run asks none.
- Two names of one model (`all-MiniLM-L6-v2`, `sentence-transformers/all-MiniLM-L6-v2`) keep two indexes; profiles can give each model one name.
- `index_status` and `index_codebase` of a directory report the whole repository without saying which root: a `project_root` field (stage 8).
- README's list of hooks leaves out import-layers, vulture and code-rules (stage 8).
- Q1 and Q2 (Appendix A): 13-30% of chunks run past the model's 256-token window, and a class's chunk embeds and returns its methods' bodies again. The 512-token candidates take up the first; the second waits for a measurement of a class's chunk against its methods'.

Stage 7, as built so far:
- **The lock at the newest versions within the bounds** (D14), as a fresh `uvx` install resolves them: sentence-transformers 5.2 → 5.7, transformers 4.57 → 5.21, huggingface-hub 0.36 → 2.3, tokenizers 0.22 → 0.23, torch 2.10 → 2.14, numpy 2.4 → 2.5, and the dev tools with them (ruff 0.17, ty 0.0.92). The default model's vectors stayed the same bit for bit (512 of click's chunks and a query, embedded by the server's embedder under each lock), and `evals quality` gave every rank and hit the same, so `IndexIdentity` names no library version: a later move of the lock checks the vectors again. On one code (`9b3bcb9`), `evals speed` gave a full index 23.4 / 27.1 / 28.6 chunks a second under the old lock and 25.3 / 29.6 / 29.0 under the new, and a search after an edit 172 / 150 / 163 ms against 165 / 136 / 158; the model loads 0.3-0.8 s slower (3.9 s) and the servers' peak memory is 65-104 MB higher (1,276 / 1,180 / 1,125 MB). The grammars moved too (tree-sitter-markdown 0.5.1 → 0.5.4, tree-sitter-rust 0.24.0 → 0.24.2) and every chunk of the three corpora, 3,036, came out the same (a review's probe), so `CHUNKER_VERSION` stays. transformers 5 draws a bar on stderr as it loads the weights, among the server's log lines; it reads huggingface-hub's switch when first imported, so the embedder turned the hub's bars off before it loaded the model, and huggingface-hub is a dependency of its own (2.3 to below 3), kept to the embedder by the import contracts; the embedder below loads no transformers, and the hub's bars stay off for its downloads. The new tools asked for three changes: a matrix splits into its rows by `numpy.unstack`, which numpy 2.5's stubs type, where iterating gave a union with a scalar; a blocking rename in an async test goes to a thread (ruff's ASYNC240); and tree-sitter-markdown 0.5.4's stub names `CapsuleType` without importing it, so the markdown chunker's grammar line ignores the deprecated overload the checker then picks.

- **An embedder of its own on onnxruntime** (D7's ONNX step), in place of sentence-transformers: it reads the model's tokenizer, `onnx/model.onnx` and its two configs from the hub at the pinned revision (or from a directory), cuts a text at the model's limit (256 tokens, over the 128 the tokenizer's file holds), batches 32 texts longest first, takes the mean of the token states over the text's tokens and makes it unit length, as the model's modules say. A model that pools otherwise is refused when it loads (`UnusableModelError`); stage 7's other candidates bring that. Its vectors are sentence-transformers' to 1e-5 in every component, a long text past the limit among them (a test holds a reference sentence-transformers made), so `IndexIdentity` stays and an index built before keeps answering; `evals quality` gives every rank and hit the same (87 of 87). On 512 of click's chunks, longest first as sentence-transformers sorts them, the fp32 export embeds 29.5 / 30.9 / 31.9 chunks a second at onnxruntime's default threads / 4 / 8 against torch's 25.8-26.2 (probed, the machine otherwise idle); O1 to O3 gain nothing, and D7's thread setting goes unbuilt: onnxruntime's default comes within 8% of the best count. The install without dev tools takes 616 MB against 1.5 GB with torch's CPU-only build on Linux (3.5 GB with PyPI's CUDA one, which the README's `--index` flag avoided), and needs no extra index anywhere. In `evals speed` (saved), the model loads in 0.29-0.37 s against 3.9, so a fresh server answers its first search in 2.0 s against 5.5-5.7; a full index runs at 25.9 / 29.4 / 32.4 chunks a second against 25.3 / 29.6 / 29.0, chunking and writing taking the rest; the servers' peak memory is about the same (1,259 / 1,126 / 1,112 MB against 1,276 / 1,180 / 1,125). MCPServer sets Python's own logging to INFO with a handler on stderr, so the records of the libraries under it (httpx's for each file the hub fetches; sentence-transformers' and its bar at every query before) landed among the server's JSON lines: it now sets WARNING, and with the model in the hub's cache the server's stderr holds its own lines alone (a review's probe, and `evals speed`'s log); a first start with an empty cache adds the hub's six-line warning about unauthenticated requests. From the review of `b7b3a35`: onnxruntime's threads spun for about half a second after each run, so a 2 ms query burned some 6 CPU-seconds, now off (`session.intra_op.allow_spinning`) at no cost in speed; the embedder reads `modules.json` and makes only a transformer, its mean pooling and a unit length, reads the pooling by flags or, as sentence-transformers 5 saves it, by name, and takes the limit from the tokenizer's config when the model's names none; the configs come before the 90 MB export, so a refused model costs no download; a model it cannot load fails with `UnusableModelError` naming the file (offline, a cache holding only what sentence-transformers fetched asks for one start online). The servers' peak memory stays torch's because onnxruntime's memory arena keeps the largest batch's buffers: off, a probe peaked at 576 MB at 8-10% less speed, a choice for stage 7's idle measurements. An export without `token_type_ids` (DistilBERT, MPNet) has no test yet. The int8 exports embed 37-40 (avx2) and 55-63 (avx512-vnni) chunks a second, but their vectors move (cosine to torch's 0.990 mean, 0.954 at the lowest) and each needs its CPU: they wait for the harness to weigh them, as a choice a model's profile names.

Open, and left as they are for now:
- A file that cannot be read is recorded as indexed with no chunks, from the scan's record: one replaced while it was read has a new mtime and is read again; one without permission stays unread until it changes or a forced run reads it, as a `chmod` changes no mtime. Not recording it would make every status stale and every search run an index.
- A search from inside a submodule builds the submodule an index of its own, though the superproject's holds its files: a submodule is a repository of its own, and routing it to the superproject would make its first search pay the superproject's whole index.
- A project path whose case differs from git's on macOS may fail [suspected; not tested].
- A file whose definitions parse but whose tail does not (`def fine():` and its body, then `def broken(`) gives the definitions only, and the broken lines are dropped without a word (a review's probe). A window over the lines no chunk covers would keep them; not taken up.


| Stage | Content |
|---|---|
| 0 | `mcp<2` pin |
| 1 | Correctness on current code: R1–R7, O5, O7 and `ignore_patterns` in git mode, with failing-first tests and real git repos in tests |
| 2 | Gates the current code passes, CI, release gated on CI, LICENSE/metadata, CONTRIBUTING, SECURITY, AGENTS.md + symlink + instruction gate, rules cleanup |
| 3 | Safety net: e2e tests through the in-memory client, `evals/` (quality and speed) with baselines |
| 4 | Rebuild on mcp 2.x: layers, lifespan, ABCs, errors, `ToolError`, single return types, presenters, settings cleanup, middleware, code-rule gates, import-linter |
| 5 | Storage and pipeline: tables, manifest, batched writes, `optimize`, locks, scan-time mtimes, Arrow, no pandas, cheap status, model loaded off the loop, embedding progress, D5/D6 remainder (skeletons, span helper, `CHUNKER_VERSION`) |
| 6 | Search: RRF, prefilters, `search_text`, similarity threshold, prose prior, no grouping, recency decision — each on the harness; one index per repository (O6) |
| 7 | Speed and model: threads, ONNX, model candidates, new targets in rules |
| 8 | Server `instructions`, docstrings, skill, plugin, skill gate, README, reviewer agent, MCP roots spike |
| 9 | After the release, replies on GitHub. Issue #3: what changed, with numbers from `evals`; take up the author's offer of their harness and compare it with ours; ask about their hardware (O4). PR #2: thank the author, explain D5 and the alternative rejected above, and close. |

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

- Users upgrading from 0.4 get one automatic full re-index. Later model or chunker changes do the same; the old index stays until no session has used it for a week, so switching back costs nothing.
- Tool output changes shape. CHANGELOG states each break.
- The rebuild requires `mcp>=2`.
- New dependencies: `filelock` (runtime); `import-linter` and `hypothesis` (dev). `pandas` leaves; torch may become optional after stage 7.
- Commits get slower: more hooks, test collection, the skill gate. CI adds minutes and a model download, which the cache softens.
- Search quality and speed become numbers we track (`evals/baseline/`), not impressions. TODO's search epic is re-ranked by them.
- Owner actions: approve the release tag; add the required reviewer on the `pypi` environment; read and approve the GitHub replies in stage 9.

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
| Q1 | 13–30% of chunks (34–62% of classes) exceed the 256-token window; dead token settings | verified | open: D7's 512-token models, stage 7 (D6 skeletons not adopted, stage 5); D11 |
| Q2 | Class bodies embedded and returned twice | verified | open: stage 7 (D6 skeletons not adopted, stage 5; stage 6's fusion did not take it up) |
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
| O6 | A subdirectory as `project_path` builds a second, separate index | verified in code (`config.py` hashes the resolved path) | D10, stage 6 |
| O7 | A tracked symlink can pull files from outside the project into the index | suspected | D5, stage 1 |

## Appendix B. Baseline measurements (v0.4.0)

Taken by the review with scripts of its own. From stage 3 on, the reference numbers are `evals/baseline/`, recorded by `python -m evals`. On the 0.4.1 code: file MRR 0.698 / 0.430 / 0.523 and chunk MRR 0.531 / 0.303 / 0.300 on this repo / click / fd; 3.2 / 1.9 / 1.4 files/s; warm search p50 87 / 101 / 87 ms; first search 6.4–7.5 s; peak memory 1.0–1.35 GB. After stage 4 and with more equally right answers in the corpora: file MRR 0.698 / 0.455 / 0.549, chunk MRR 0.544 / 0.328 / 0.318. Stage 4 moved no rank: the stage-3 code scores the same on today's corpora, so the change comes from the answers added. A review's A/B on this repo, alternating runs under the same load, found indexing unchanged (20.7–20.8 s at stage 3 against 19.9–22.3 s), warm search p50 down from 91 to 57–59 ms and peak memory from 1.30–1.39 to 0.95–1.01 GB; unchanged code drifts about 10% between sessions, so one run cannot show a smaller change. Read `evals/baseline/` for the current figures.

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
2. **Before the first search.** `project_path` is absolute; a subdirectory of a repository is searched in the repository's one index and its results stay in that subdirectory (D10). The first call loads the model and may index, with progress; do not retry while it runs. If the tool is missing, tell the user how to install the plugin; do not install it yourself.
3. **Phrasing a query.** Describe what the code does, as a docstring would; one concept per query; the domain's nouns; three to ten words; no file names. Pairs of examples that work and that do not.
4. **Filters.** `chunk_kinds` (Rust types count as `class`), `languages`, `path_glob`/`exclude_glob` (for example, excluding `tests/**` or `docs/**`), `limit` and its cap.
5. **Reading the answer.** Every result field. Results come in rank order. `truncated` means Read the range before quoting. `similarity` versus rank. `include_debug` only when a search was slow or odd.
6. **The index.** Automatic incremental re-index on every search; `index_codebase(force=true)` after mass renames or when results point at code that no longer exists; `index_status` to check the root and the manifest, not before every search; where the index lives; automatic rebuild after a model or version change; `use_gitignore=false` for allowlist repos.
7. **Traps already met.** A `project_path` one level too deep; exact identifiers ranking below paraphrases; queries naming a file; a module docstring or class skeleton landing first, with the body in the next hit of the same file.
