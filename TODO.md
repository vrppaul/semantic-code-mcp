# TODO

What we mean to build and why. How each epic is built lives in `docs/decisions/`; an item done is removed and recorded in `CHANGELOG.md`.

## In Progress

### Rebuild on mcp 2.x
The published 0.4.0 does not start on mcp 2.x, the index dropped code without a word, search ranks prose over the code it describes, indexing runs 20–50× slower than our own target, and our gates passed without checking anything. `docs/decisions/006-rebuild-on-mcp-2.md` holds the plan in ten stages, from a hotfix to the skill and the plugin; the stages and their state live there.

## Pending

### JavaScript, TypeScript and Go
Web and systems projects are invisible to search today. Tree-sitter has the grammars. Wait for ADR 006's chunker base (span helper, fallback chunk, `CHUNKER_VERSION`) so new languages do not inherit its old holes, and spike tree-sitter tags queries as the per-language definition of a symbol first.

### Background re-indexing
A search after a large change waits for the whole re-index. Answering from the old index while a new one builds would hide that wait. The per-project lock and the files table it needed came with ADR 006's stage 5; a new index for another identity could be built beside the old one the same way.

### More tools: find_symbol, find_similar, list_indexed_files
"Where is X defined" across languages, "code like this", and "what is indexed" are cheap given the index, and Grep answers the first only with every usage mixed in. ADR 006 defers them until the search surface settles.

### Indexes of projects nobody uses
With the global cache (`~/.cache/semantic-code-mcp/`), a project moved or deleted, or one of the many worktrees Claude Code makes, keeps its whole index there for good: a store looks for unused indexes only inside its own project's directory. Sweep the sibling project directories by the same rule, an index nobody used for a week, under the same lock. A project directory keeps its lock files, which are never deleted while a writer may wait on one, so it needs a rule for when the directory itself may go.

### Debt: chunker tests named by their input
About forty tests of the three chunker test files are named by what they feed the chunker (`test_simple_function`, `test_struct_definition`), not by the behaviour they hold, as `.claude/rules/testing.md` asks; stage 6 renamed only those it rewrote.

### LanceDB index tuning
A flat cosine scan answers in 23–36 ms at 11.5k chunks and, over fragments of 4,096 rows read side by side, about 40 ms at 100k on LanceDB 0.30.2 (54 over one fragment). LanceDB 0.32 to 0.40 scan two to three times slower (150–190 ms at 100k), which holds the package at 0.30.2; a vector index (IVF or HNSW) past some row count would let it move on, and pays off on large corpora anyway.
