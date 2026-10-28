---
name: semantic-search
description: Find code by the words it uses with the semantic-code MCP server's search_code, and know when Grep or Read serves better. Use when you look for behaviour, a concept or a definition in a project's Python, Rust or Markdown files.
---

The semantic-code server splits a project into functions, classes, methods, module
docstrings and document sections and finds them by the words their code uses, ranked, the
chunks named as the query first. It answers what Grep cannot: the few chunks that best match, not every line. It does not
answer what Grep does better.

## 1. Which tool

| You want | Use |
|---|---|
| a string, a regex | Grep |
| the definition of a name you know | `search_code` with the name |
| every call site of a name | Grep: `search_code` returns its best matches only |
| a file whose path you know | Read |
| behaviour you can describe but not name | `search_code` |
| a concept spread over several files | `search_code`, then Grep the names it shows |
| a language the server does not index | Grep |

The loop that works: `search_code` → Read the line range of the best hit → Grep the names
you found there.

The server indexes Python, Rust and Markdown files: `languages` takes `python`, `rust`
and `markdown`.

## 2. Before the first search

- `project_path` is absolute. Give the repository's root, or a directory inside it to keep
  the search to that directory: the repository keeps one index either way, and
  `project_root` in the answers of `index_status` and `index_codebase` names it (a
  directory the repository ignores, or one in a repository that holds the home directory,
  is a project of its own, and names itself). Left out,
  it is the client's workspace root, the directory Claude Code was started in, which must
  lie in a git repository: a folder of projects, or the home directory, would be indexed as
  one. Give it whenever the work lies outside that directory or outside any repository; a
  client that names no single root, or a root no repository holds, gets an error asking
  for `project_path`.
- The first call indexes the whole project's words, seconds even on a large one; on a
  server its user set to search by meaning too, the first search also embeds every chunk,
  minutes on a large one, with progress. Wait for it; do not call again while it runs.
- If the tools are missing, tell the user the server is not installed and point them to
  semantic-code-mcp's README (github.com/vrppaul/semantic-code-mcp); do not install it
  yourself.

## 3. Phrasing a query

Write `query` in the words the code would use: identifiers, their parts, the domain's
terms. On a miss, search again with other words. Name no file: narrow with `path_glob`
instead.

| Works | Does not |
|---|---|
| "retry backoff delay attempts" | "how does the code work" — no word of the code |
| "parse_args config options" | "progress stuff in utils.py" — a file name |
| "release lock write error" | "the thing that handles it" — no concept |

A name, bare or qualified, as `greet` or `Stack.push`, brings the chunks of that name
first. For every place a name is used, Grep.

## 4. Narrowing a search

- `chunk_kinds`: `module`, `function`, `method`, `class`, `section`. Rust's structs, enums,
  traits and impls are classes; a Markdown heading with its text is a section.
- `languages`: `python`, `rust`, `markdown`.
- `path_glob` and `exclude_glob`, read as in a `.gitignore`, from the directory
  `project_path` names: `exclude_glob="tests"` leaves out every `tests` directory,
  `path_glob="src/**/*.py"` keeps to Python under `src`. A glob with a character class,
  braces, an escape, a leading `!` or `..` is refused, with the reason.
- `limit`: how many results at most; the tool's schema gives its default and its cap.

## 5. Reading the answer

`search_code` returns `project_root`, the project's root (its repository's, or outside
git the directory itself), and
`searched_directory`, the directory it kept to: where the two differ, nothing outside
`searched_directory` was searched, and `project_root` as `project_path` searches it all.
Then `results`, best first. Each has:

- `file_path` (absolute), `line_start`, `line_end`: Read that range for the whole chunk;
- `name`, `kind`, and `parent`: the class, impl, module or heading it sits in, as `Stack`
  for `Stack.push` or `Guide > Install` for a section;
- `content`, cut at a limit of lines; `truncated` says it was cut: Read the range before
  quoting.

A search can return nothing: that is an answer, not a failure.

Ask `include_debug=true` only when a search was slow or odd: `debug` then holds
`timings` (`embedding_ms`, `search_ms`, `total_ms`, and `indexing_ms` where it indexed
first), `stats` (`results_count`, `filtered_out`, the hits a server set to search by
meaning too dropped below its similarity floors, `unique_files`, `tokens_estimate`) and `indexing`, what the search indexed before it
searched (`files_indexed`, `chunks_indexed`).

## 6. The index

- Each search first re-indexes the files changed since the last; you never need to index
  before searching.
- Call `index_codebase` with `force=true` only when results point at code that is gone:
  it reads every file again, and on a server set to search by meaning too, a model whose
  weights nothing pins has its vectors dropped, so the next search embeds every chunk
  again; renames and deletions need nothing.
  It answers `project_root`, `files_indexed`, `chunks_indexed`, `files_deleted` and
  `duration_seconds`.
- Call `index_status` to see which root a directory belongs to and what built its index,
  not before every search. It answers `project_root`, `is_indexed`, `last_updated`,
  `files_count`, `chunks_count`, `embedded_chunks_count` (the chunks a server set to search
  by meaning too reads without embedding first), `stale_files_count` and the `manifest`: `model_name`,
  `model_revision`, `embedder_version`, `chunker_version`, `schema_version`.
- Each chunker version keeps an index of its own, built once on the first search after a
  change, and each model's vectors are kept apart in it; an index no session used for a
  week is removed.
- In a git repository, git lists the files: tracked ones, and untracked ones `.gitignore`
  does not exclude. The server's settings are its environment's, which the user sets, not
  a tool's parameters: a repository whose `.gitignore` hides source on purpose needs
  `use_gitignore` off (`SEMANTIC_CODE_MCP_USE_GITIGNORE=false`), so ask the user for it;
  with `local_index` the index lives in `.semantic-code/` inside the project; with
  `SEMANTIC_CODE_MCP_SEARCH_MODE=words_and_meaning` every search searches by meaning too,
  where the server was installed with its extra `semantic-code-mcp[meaning]`; without it the
  server does not start.

## 7. Traps already met

- A `project_path` one level too deep, or a session started in a subdirectory, keeps the
  search to that directory, and the answer in the directory above never comes back.
  `searched_directory` shows it; search from `project_root` when unsure.
- A glob reads from the directory searched: from inside `src`, `src/**/*.py` names
  `src/src/...` and finds nothing; write `**/*.py`, or search from the root.
- By words, a chunk that repeats the query's words can rank above the one that does what
  it asks; search again with the code's own terms.
- A query that names a file finds chunks that mention the file. Use `path_glob`.
- A class's chunk holds its methods too, so a class and one of its methods can both come
  back for one question; the method's chunk is the narrower answer.
- A module's docstring can come first for a question about the module; the code that does
  the work is usually the next hit of the same file.
