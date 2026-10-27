# TODO

What we mean to build and why. How each epic is built lives in `docs/decisions/`; an item done is removed and recorded in `CHANGELOG.md`.

## In Progress

### Rebuild on mcp 2.x
The published 0.4.0 does not start on mcp 2.x, the index dropped code without a word, search ranks prose over the code it describes, indexing runs 20–50× slower than our own target, and our gates passed without checking anything. `docs/decisions/006-rebuild-on-mcp-2.md` holds the plan in ten stages, from a hotfix to the skill and the plugin; the stages and their state live there.

### The index store behind a contract
The store's interface speaks LanceDB, the index's life sits inside the LanceDB store, and the tests hold the implementation, so the store cannot be read, tested or replaced apart from LanceDB; and a run cancelled as its last write lands leaves that write's rows out of the full-text index. SQLite was tried behind the plan and turned down: its keyword search is 13 times slower at 100k chunks, or worse ranked when made fast. `docs/decisions/007-storage-behind-a-contract.md` holds the plan.

## Pending

### Search quality: a bigger harness, a reranker, what is embedded
Every ranking decision of ADR 006 rests on 66 chosen questions, where its criterion's step of 1.0 is about one question, judged by one judge per corpus with no agreement figure. A harness of 200 and more questions over more corpora and languages, with judge agreement, comes first; on it, a small cross-encoder reranking the first 30 hits (often the largest gain in code search, an estimated 20-40 ms on a CPU), and what the model embeds: a header of path, parent and signature, windows over bodies past the model's 256 tokens (13-30% of chunks), a class as its outline rather than its methods again (ADR 006, Q1 and Q2). A change of what is embedded builds every index again.

### What an agent gets: compact hits, a repository map, a CLI
`search_code` answers up to ten chunks of up to 50 lines, which an agent mostly Reads again anyway: compact hits (path, lines, name, signature, a few lines) would give more per token, and are a change of a tool's answer, best made once. A map of a repository's symbols, and a command line beside MCP that any agent with a shell can call, without a server's life to manage.

### One daemon per user
Each Claude Code session runs its own server: one model in memory each (about 550 MB held), the same projects checked and indexed by each, every search asking git four or five times. One process per user holding the model and every index, with one writer per index and a file watcher indexing in the background, would remove that; sessions would reach it through a thin MCP layer or the CLI. It costs a process's life to manage (start, upgrade, stop) and sockets on every platform.

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
A flat cosine scan answers in 23–36 ms at 11.5k chunks and 50–90 ms at 100k over one fragment on LanceDB 0.30, about 40 ms over the fragments of 4,096 rows that upkeep now keeps, read side by side. LanceDB 0.32 to 0.40 scan two to three times slower over one fragment (150–190 ms at 100k), which holds the package at 0.30.2; whether they gain as much from such fragments is not measured; a vector index (IVF or HNSW) past some row count would let it move on, and pays off on large corpora anyway.
