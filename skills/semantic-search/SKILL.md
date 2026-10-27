---
name: semantic-search
description: Find code by what it does with the semantic-code MCP server's search_code, and know when Grep or Read serves better. Use when you look for behaviour or a concept whose names you do not know, in a project's Python, Rust or Markdown files.
---

The semantic-code server splits a project into functions, classes, methods, module
docstrings and document sections, embeds each, and finds them by meaning and by words
together. It answers what Grep cannot: code whose names you do not know yet. It does not
answer what Grep does better.

## 1. Which tool

| You want | Use |
|---|---|
| an exact identifier, a string, a regex | Grep |
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
  `project_root` in the answers of `index_status` and `index_codebase` names it.
- The first call loads the model and may index the whole project, with progress. Wait for
  it; do not call again while it runs.
- If the tools are missing, tell the user the server is not installed and point them to
  the project's README; do not install it yourself.

## 3. Phrasing a query

Write `query` as the code's docstring would say what it does: one concept, the domain's
own nouns, three to ten words. Name no file: narrow with `path_glob` instead.

| Works | Does not |
|---|---|
| "retry a failed request after a growing delay" | "retry" — one word: Grep it |
| "parse command-line options into a config" | "how does the code work" — no concept |
| "release the lock when a write fails" | "LanceDBIndexStore.upkeep" — an identifier: Grep it |
| "show a progress bar while iterating" | "progress stuff in utils.py" — a file name |

A bare name, as `greet`, is answered too: a chunk of that name is never dropped for its
meaning. For every place a name is used, Grep.

## 4. Narrowing a search

- `chunk_kinds`: `module`, `function`, `method`, `class`, `section`. Rust's structs, enums,
  traits and impls are classes; a Markdown heading with its text is a section.
- `languages`: `python`, `rust`, `markdown`.
- `path_glob` and `exclude_glob`, read as in a `.gitignore`, from the directory
  `project_path` names: `exclude_glob="tests"` leaves out every `tests` directory,
  `path_glob="src/**/*.py"` keeps to Python under `src`. A glob with a character class,
  braces, an escape, a leading `!` or `..` is refused, with the reason.
- `limit`: 10 by default, at most 50.

## 5. Reading the answer

`search_code` returns `results`, best first by meaning and words together. Each has:

- `file_path` (absolute), `line_start`, `line_end`: Read that range for the whole chunk;
- `name`, `kind`, and `parent`: the class, impl, module or heading it sits in, as `Stack`
  for `Stack.push` or `Guide > Install` for a section;
- `content`, cut to 50 lines; `truncated` says it was cut: Read the range before quoting;
- `similarity`: how close its meaning is to the query's alone. Rank counts words too, so a
  hit its words found can rank high with a low similarity. A search can return nothing
  when nothing is near enough: that is an answer, not a failure.

Ask `include_debug=true` only when a search was slow or odd: `debug` then holds
`timings`, `stats` and `indexing`, what the search indexed before it searched.

## 6. The index

- Each search first re-indexes the files changed since the last; you never need to index
  before searching.
- Call `index_codebase` with `force=true` after mass renames, or when results point at
  code that is gone. It answers `project_root`, `files_indexed`, `chunks_indexed`,
  `files_deleted` and `duration_seconds`.
- Call `index_status` to see which root a directory belongs to and what built its index,
  not before every search. It answers `project_root`, `is_indexed`, `last_updated`,
  `files_count`, `chunks_count`, `stale_files_count` and the `manifest`: `model_name`,
  `model_revision`, `chunker_version`, `schema_version`.
- Each model and version keeps an index of its own, built once on the first search after a
  change; one no session used for a week is removed.
- In a git repository, git lists the files: tracked ones, and untracked ones `.gitignore`
  does not exclude. A repository whose `.gitignore` hides source on purpose needs the
  setting `use_gitignore` off (`SEMANTIC_CODE_MCP_USE_GITIGNORE=false`); with `local_index`
  the index lives in `.semantic-code/` inside the project.

## 7. Traps already met

- A `project_path` one level too deep keeps the search to that directory, and the answer
  in the directory above never comes back. Search from the root when unsure.
- An exact identifier can rank below a chunk that paraphrases it. Grep exact names.
- A query that names a file finds chunks that mention the file. Use `path_glob`.
- A class's chunk holds its methods too, so a class and one of its methods can both come
  back for one question; the method's chunk is the narrower answer.
- A module's docstring can come first for a question about the module; the code that does
  the work is usually the next hit of the same file.
