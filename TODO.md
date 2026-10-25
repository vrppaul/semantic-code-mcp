# TODO

What we mean to build and why. How each epic is built lives in `docs/decisions/`; an item done is removed and recorded in `CHANGELOG.md`.

## In Progress

### Rebuild on mcp 2.x
The published 0.4.0 does not start on mcp 2.x, the index dropped code without a word, search ranks prose over the code it describes, indexing runs 20–50× slower than our own target, and our gates passed without checking anything. `docs/decisions/006-rebuild-on-mcp-2.md` holds the plan in ten stages, from a hotfix to the skill and the plugin; the stages and their state live there.

## Pending

### JavaScript, TypeScript and Go
Web and systems projects are invisible to search today. Tree-sitter has the grammars. Wait for ADR 006's chunker base (span helper, fallback chunk, `CHUNKER_VERSION`) so new languages do not inherit its old holes, and spike tree-sitter tags queries as the per-language definition of a symbol first.

### Background re-indexing
A search after a large change waits for the whole re-index. Answering from the old index while a new one builds would hide that wait. It needs the per-project lock and the files table of ADR 006 first.

### More tools: find_symbol, find_similar, list_indexed_files
"Where is X defined" across languages, "code like this", and "what is indexed" are cheap given the index, and Grep answers the first only with every usage mixed in. ADR 006 defers them until the search surface settles.

### LanceDB index tuning
A flat cosine scan answers in 23–36 ms at 11.5k chunks. IVF or PQ indexes pay off only on much larger corpora; revisit when one over 100k chunks shows the need.

### Debt: the chunker tests filter in comprehensions
`tests/unit/test_python_chunker.py`, `tests/unit/test_rust_chunker.py` and `tests/unit/test_markdown_chunker.py` predate the rule that a test holds no branch; about 27 of their assertions pick chunks with `[chunk for chunk in chunks if ...]`. Rewrite them on `summarize()` from `tests/helpers.py`, as the newer tests in those files do.
