# TODO

What we mean to build and why. How each epic is built lives in `docs/decisions/`; an item done is removed and recorded in `CHANGELOG.md`.

## In Progress

### Rebuild on mcp 2.x
The published 0.4.0 does not start on mcp 2.x, the index dropped code without a word, search ranks prose over the code it describes, indexing runs 20–50× slower than our own target, and our gates passed without checking anything. `docs/decisions/006-rebuild-on-mcp-2.md` holds the plan in ten stages, from a hotfix to the skill and the plugin; the stages and their state live there.

### A first search in seconds, not after the whole index
The first search of a repository waits for its whole index, in one tool call: the owner's algotrading (19,432 chunks) took 733 s, Sentry (87,694 chunks) about an hour, the model taking about 95% of it; and a warm search on Sentry spent a second finding files before it searched (0.3 to 0.4 s since ADR 009's step 3). These are the repositories where search helps an agent (ADR 008, pilots 5 and 8): "Этот продукт нельзя использовать" (the owner, 2026-10-27). `docs/decisions/009-words-first-meaning-on-request.md` holds the plan: words by default, their index built whole in seconds; meaning on request, embedded in parts by a child process at the lowest priority; a lean stale check; SQLite the default store behind ADR 007's contract.

## Pending

### The waits a user meets, cut where they were measured
After words became the default, a warm search on Sentry spends 85 to 90% of its 340-400 ms checking which files changed, a search after an edit scans the files twice (800-1,140 ms), a first answer takes 12.5 s (chunking about 7 s, the writes 4.5 s) and a first index holds every chunk at once, its peak 383 MB. `docs/decisions/010-waits-measured-and-what-would-cut-them.md` holds the split and the changes ranked: a watcher of the files instead of a scan per search, a lean scan, one pass after an edit, a first index in a pool, streamed and in few writes, the model's libraries imported with the model. The owner put them after ADR 009's other steps.

### Search quality: a bigger harness, a reranker, what is embedded
Every ranking decision of ADR 006 rests on 66 chosen questions, where its criterion's step of 1.0 is about one question, judged by one judge per corpus with no agreement figure. A harness of 200 and more questions over more corpora and languages, with judge agreement, comes first; on it, a small cross-encoder reranking the first 30 hits (often the largest gain in code search, an estimated 20-40 ms on a CPU), and what the model embeds: a header of path, parent and signature, windows over bodies past the model's 256 tokens (13-30% of chunks), a class as its outline rather than its methods again (ADR 006, Q1 and Q2). A change of what is embedded builds every index again.

### What an agent gets: a repository map, a CLI
A map of a repository's symbols, and a command line beside MCP that any agent with a shell can call, without a server's life to manage. Compact hits were judged by agents and turned down (ADR 008, pilot 7): cut to five lines, a hit sent the agent to Read the file in four sessions of fifteen against one, at 46% more context read.

### One daemon per user
Each Claude Code session runs its own server: one model in memory each (about 550 MB held), the same projects checked and indexed by each, every search asking git four or five times. One process per user holding the model and every index, with one writer per index and a file watcher indexing in the background, would remove that; sessions would reach it through a thin MCP layer or the CLI. It costs a process's life to manage (start, upgrade, stop) and sockets on every platform.

### JavaScript, TypeScript and Go
Web and systems projects are invisible to search today. Tree-sitter has the grammars. Wait for ADR 006's chunker base (span helper, fallback chunk, `CHUNKER_VERSION`) so new languages do not inherit its old holes, and spike tree-sitter tags queries as the per-language definition of a symbol first.

### More tools: find_symbol, find_similar, list_indexed_files
"Where is X defined" across languages, "code like this", and "what is indexed" are cheap given the index, and Grep answers the first only with every usage mixed in. ADR 006 defers them until the search surface settles.

### Indexes of projects nobody uses
With the global cache (`~/.cache/semantic-code-mcp/`), a project moved or deleted, or one of the many worktrees Claude Code makes, keeps its whole index there for good: a store looks for unused indexes only inside its own project's directory. Sweep the sibling project directories by the same rule, an index nobody used for a week, under the same lock. A project directory keeps its lock files, which are never deleted while a writer may wait on one, so it needs a rule for when the directory itself may go.

### Dependencies' code
The owner wants the code of a project's dependencies searchable too, last and in turn after the project's own (2026-10-27). Discovery leaves out `.venv`, `site-packages` and `node_modules`, and a chunk's path must lie under the project's root (`Chunk.path`), so it needs a decision of its own and a change of the domain model; ADR 009 leaves it out.

### Debt: chunker tests named by their input
About forty tests of the three chunker test files are named by what they feed the chunker (`test_simple_function`, `test_struct_definition`), not by the behaviour they hold, as `.claude/rules/testing.md` asks; stage 6 renamed only those it rewrote.

### Versions held by what makes their output
The chunker's, the embedder's and each store's schema's versions are held by tests of hand-picked samples, and a review found nine chunker rules and the full-text index's settings outside them, so a change of output can keep an old version and meet an index of another making. A digest of what makes each output, held by a gate, would catch it (ADR 007, D6).

### LanceDB index tuning
For the LanceDB store, an optional extra since ADR 009. A flat cosine scan answers in 23–36 ms at 11.5k chunks and 50–90 ms at 100k over one fragment on LanceDB 0.30, about 40 ms over the fragments of 4,096 rows that upkeep now keeps, read side by side. LanceDB 0.32 to 0.40 scan two to three times slower over one fragment (150–190 ms at 100k), which holds the package at 0.30.2; whether they gain as much from such fragments is not measured; a vector index (IVF or HNSW) past some row count would let it move on, and pays off on large corpora anyway.
