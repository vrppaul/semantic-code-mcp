# 009: A first answer in seconds: words by default, meaning on request, SQLite behind a store contract

**Status**: accepted (2026-10-28)
**Date**: 2026-10-27; revised 2026-10-28 after two Fable reviews and the owner's answers
**Continues**: 006 (D4, D7, D9, the speed targets), 007 (the contract; D7 reversed), 008 (pilot 8)

## Context

The first search of a repository waits for its whole index inside one tool call: 733 s on the owner's algotrading (1,399 files, 19,432 chunks), about an hour on Sentry (8,546 files, 87,694 chunks). Split by phase on algotrading, the model takes about 95% of it: finding the files takes 0.6 s, chunking about 11 s, tokenizing 6.5 s (all under load), and 2.85 million tokens through a six-layer transformer the rest. A rate was measured and targeted ("a full index over 20 chunks a second", which allows 15 minutes on 19k chunks), never the wait; stage 7 raised the rate by percents. And 26% of chunks run past the model's 256 tokens, so 42% of the code's text never reaches it.

A warm search pays too: before every search the server walks the tree to find stale files, 0.68 to 1.13 s on Sentry, of which git takes 0.04 s and stating 8,546 files the rest. The pinned corpora have 32 to 133 files, so no target saw it.

How the product is used, as the owner set it (2026-10-27, 2026-10-28): any developer, first the repositories one works in often (worktrees, branch switches, several sessions at once), then first contact with a stranger's large repository; any agent, Claude Code first; code never leaves the machine; no hidden load on the user's CPU; dependencies' code last, indexed in turn; the word index whole, meaning in parts; the product keeps its name.

Whether the tool is needed, and what part of it: 180 sessions of Claude Code headless (ADR 008, pilot 8):

| | own tools | words only | words and meaning |
|---|---|---|---|
| algotrading, right | 25 of 30 | 30 of 30 | 30 of 30 |
| algotrading, turns / prompt / time | 4.8 / 95k / 13.7 s | 2.6 / 59k / 9.8 s | 2.5 / 56k / 8.8 s |
| algotrading, sessions that searched | | 28 of 30 | 28 of 30 |
| Sentry, right | 27 of 30 | 30 of 30 | 30 of 30 |
| Sentry, turns / prompt | 4.5 / 86k | 3.9 / 83k | 4.2 / 89k |
| Sentry, sessions that searched | | 11 of 30 | 8 of 30 |

On code no model has seen, the search halves the turns and cuts the prompt by 40%; on Sentry, which the model knows, the agent greps names from memory and gains little. Words alone match words and meaning on both: turns equal in 10 of algotrading's 15 questions, the rest split 3 to 2. The agent writes its query in the words code uses (`lazily create database connection pool bound to running event loop`) and searches again when it misses. The words-only arm ran with no similarity floor and answered 30 of 30.

The harness's 66 questions ask once, in the question's own wording, and there meaning has a niche. Chunk MRR with words alone / words and a static model (Model2Vec's potion-code-16M-v2) / words and MiniLM (today's model), both fused as today, the floors off (2026-10-28; MiniLM without floors scores as with them, within 0.03):

| | code | documentation | code, no word of the answer's name | bare names | held-out code |
|---|---|---|---|---|---|
| click | 0.66 / 0.67 / 0.80 | 0.60 / 0.70 / 0.70 | 0.24 / 0.50 / 0.39 | 0.68 / 0.68 / 0.83 | 0.53 / 0.63 / 0.80 |
| fd | 0.47 / 0.54 / 0.61 | 0.56 / 0.66 / 0.75 | 0.16 / 0.44 / 0.53 | 0.78 / 0.71 / 0.77 | 0.40 / 0.36 / 0.43 |
| this repository | 0.66 / 0.69 / 0.80 | 0.69 / 0.56 / 0.81 | 0.35 / 0.50 / 0.75 | 0.58 / 0.53 / 0.60 | 0.62 / 0.75 / 0.87 |

Words alone are LanceDB's full-text index; SQLite's FTS5 over the same search text ranks within 0.07 of it in every cell. MiniLM leads in 12 of 15 cells and ties the static model in one; words alone lead on fd's bare names. The static model recovers part of what words miss on code, but loses to words alone on bare names in fd and this repository, on this repository's documentation and on fd's held-out code. Without vectors the floors go: 15 off-topic questions get 140 results against 1 to 14 with MiniLM's floors. A Fable reviewer's proxy (1,000 docstrings as queries each, which favours words) put words and MiniLM 0.046 MRR@10 above words alone on algotrading and 0.101 on Sentry.

Without the model, a first index on an idle machine (found, chunked, search text, written, full-text index built): on SQLite, algotrading 2.6 s and Sentry 19.0 s (1.13 + 10.6 + 0.89 + 6.42); on LanceDB 2.7 s and 15.5 s. The keyword channel over the same chunks, the harness's questions and the agents' own queries (median / slowest): LanceDB 2.4 / 3.7 ms at 19k chunks and 9 / 15 ms at 88k; FTS5 15 / 35 ms and 105 / 265 ms, and 385 ms at its slowest under load in ADR 007's spike. The vector channel, from that spike at 100k chunks: a matrix held in the process searches in 6 ms, loads in 263 ms and takes another connection's edit of 50 chunks in 27 ms, against LanceDB's flat scan of 40-90 ms. LanceDB, Lance and pyarrow are 422 MB of the install (117, 160 and 145); onnxruntime and tokenizers 75 MB. The static model embeds about 2,500 chunks a second (Sentry in 34 s), the transformer about 26.

Elsewhere: Cursor embeds server-side, caches embeddings by chunk content and lets a clone start from a teammate's index (p99 time to first query 4.03 h to 21 s); Sourcegraph's Cody dropped embeddings for keyword search; Claude Code's own search is grep and glob; Semble (MinishLab) indexes in under a second with BM25 and static embeddings and reports BM25 with its ranking rules at 0.834 NDCG@10 against 0.854 for its hybrid, on questions a model wrote from the code.

## Decision

### D1. Words by default, an exact name first

`search_code` answers from the keyword channel: BM25 over each chunk's search text (its content, its name, the words of its identifiers, its parent's words). The model is not loaded for it. Hits keep up to 50 lines (ADR 008, pilot 7).

A query that is a chunk's name, its case and the blanks around it aside (`name_asked`), puts the chunks of that name first, then the rest by BM25; a qualified name (`Stack.push`) is read as a parent and a name. Today the name only keeps its chunk past the floors; by words alone a bare name ranks under the chunks that use it most (click's 89: chunk MRR 0.83 with MiniLM, 0.68 by words). The rule is held to ADR 006's criterion on the bare names.

### D2. Meaning on request

`search_code` takes `by_meaning`, false by default; a setting gives its default for a user who wants meaning always. With it, the search fuses both channels as today. The model is MiniLM, as today: the static model ranks below words alone where an agent asks most, on bare names, in two corpora of three (Context).

- A hit with a vector is held to the similarity floors as today; one without (its file not yet embedded, D3) came by words and stays by their rules.
- `similarity` leaves the hit: a hit by words has none, and a number on two scales misleads; the order is the ranking.
- The tool's description says: the default matches the words of code, so write the identifiers and terms code would use and search again with others on a miss; set `by_meaning` for behaviour described without the code's words, after a search by words missed; on a repository not yet embedded it starts the embedding and answers from what is there. Pilot 8's agents wrote queries in code's words unprompted; the wording is settled by agents (step 8).
- The model's libraries stay in the base install; its weights come with the first search by meaning.

### D3. The word index whole, meaning in parts

- **Words.** A call that finds the word index missing or stale builds it whole before it answers: 2.6 s on algotrading, 19 s on Sentry. The run takes the project's index lock before it looks for files, so a second session finding the lock held waits, checks again and reads; two sessions never chunk one repository twice. The wait has no limit and reports that another session builds the index: today's 60 s (`WRITE_LOCK_TIMEOUT_SECONDS`) is shorter than Sentry's build under load, where chunking ran about five times its idle time, and the lock is a file lock, freed when its holder's process dies. Parts would save about 13 s once per large repository, and a 733 s run completed inside one call (ADR 008).
- **Vectors.** Embedded by D6 in parts, in this order: files changed in the working tree and the last commits, the directory the search names, the rest of the source, documentation, tests. Each part is written as a unit; a search by meaning fuses the vectors there are.
- **Coverage.** An answer by meaning says what share of the project's chunks has vectors and, until it is all, that the rest came by words alone, so an agent knows what meaning did not see. An answer by words covers the whole index and says nothing more.

### D4. Staleness from git

The manifest records the commit the index was built at and a digest of the settings that change discovery (ignore patterns, gitignore use). A search asks git once, with `--no-optional-locks` so it never writes `.git/index` or trips on the user's own lock: `status --porcelain=v2 --branch --untracked-files=all` gives HEAD and the working tree's changes (else an untracked directory is one line). Only when HEAD moved from the recorded commit does a second call list the paths between them (`diff --name-only --no-renames`, so a renamed file's old path shows too). Those paths, kept if they pass discovery's filters (extension, ignore patterns), and the files the index took while changed or untracked in the working tree (else a deleted untracked file is never seen), are checked against their records as today (size, time, then the content's digest); a submodule's path stands for every indexed file under it. The tree is walked only when git cannot answer: not a repository, git failing, the recorded commit gone, the settings' digest changed, or a change to `.gitignore`, `.git/info/exclude` or the global excludes, which make indexed files vanish from the status. On Sentry git takes 0.04 s against the walk's 0.7 to 1.1 s.

### D5. Off-topic questions

Without vectors no floor holds back an off-topic hit. Agents were not hurt by that (pilot 8's words-only arm, no floor, 30 of 30 on both corpora), so no rule is built now: `evals quality` reports the off-topic results of each mode, and a rule comes when an agents' run shows the harm.

### D6. Meaning in the background, in a child process

- **A process, not threads.** onnxruntime's own pool does the work, and Python sets no priority on it; a child process takes the lowest priority the platform gives a process (`SCHED_IDLE` on Linux, nice 19 on macOS, `IDLE_PRIORITY_CLASS` on Windows, all from the standard library) and runs the model on two threads. The query's vector is embedded in the server, at the user's priority, the model loaded there on the first search by meaning as today, and before the child starts, so two processes never fetch its weights at once. A session with meaning on so holds the model twice, the server (571-693 MB resident, `evals/baseline/speed.json`) and the child (785-860 MB at its peak); a query served by an idle child would stall whenever anything else wants the CPU.
- **One child per server**, embedding the projects that asked for meaning one at a time. Before a project it takes the project's embedding lock without waiting; held, it leaves the project to the session holding it, whose vectors this server reads as they land (ADR 007, D7), and tries again after.
- **Owned by the registry.** The registry starts the child on the first search by meaning and stops it when its lifespan ends; the child also stops when the pipe from its parent closes. So it dies with its session: no load once the user closes the agent, and the next session's first search by meaning resumes from the vectors written. At the lowest priority its rate on a busy machine has no floor, so `index_status` reports the vectors' share and the rate. Each part is written in one transaction (SQLite) or under the write lock (LanceDB).
- **Vectors by content, per machine.** A cache keyed by the model's identity and the digest of the text embedded, a SQLite file per model under the cache directory: before embedding a part the child takes what the cache holds, so worktrees, clones and branch switches embed only what is new (six worktrees of algotrading hold 23,273 distinct chunks among 81,682). The cache is only a cache: each project's store keeps its own vectors, so the sweep of unused indexes drops cached vectors no part has used within its period. A vector is 1.5 KB.
- Whether the model gets each chunk's first 64 tokens, three to four times cheaper and as good fused on the Fable reviewer's probe, is settled on the harness.

### D7. The index's identity, split

`IndexIdentity` holds the model's name, its revision, the embedder's version and the chunker's version (`domain/models.py`), so with words the default, a change of a model setting never used would build the word index again. It splits: the word index is identified by the chunker's version, the search text's version and the store's kind, which name its directory; the vectors by the model's name, revision and the embedder's version, kept apart by that identity inside the store, so a change of model embeds again without chunking again. The vectors of a model no longer used stay in a live index until it is built again; a change of model is rare. This changes the domain model and the storage schema, which AGENTS.md asks about first; the owner left the technical construction to the authors and the Fable reviewer (2026-10-28).

### D8. Stores behind one contract: SQLite by default, LanceDB as an option, more to come

- The contract (ADR 007, D1, D2) in the index's own terms: the manifest, the files' records, a part written as a unit, chunks without vectors, vectors added to stored chunks by model identity, the candidates of words and of meaning read from one state, the counts coverage needs.
- One suite of tests holds the contract over a store factory, run on every store; a store's own behaviour is tested apart. A new store is a module, an optional extra of the package with its libraries, and the suite passing; import-linter keeps stores from importing one another.
- A setting picks the store; its kind is part of the index's directory name, so stores never meet one another's files, and the sweep of unused indexes removes any kind.
- SQLite is the default: one database file in WAL mode; a part's write in one transaction; FTS5 over the search text (`porter unicode61`), a query's tokens each quoted and joined by OR, so the question is never read as FTS5's syntax; vectors as float32 blobs in a table of their own, searched by a matrix the process holds and refreshes by rows when another connection wrote (ADR 007, D7).
- LanceDB is the extra `semantic-code-mcp[lancedb]`: today's store, with chunks without vectors added and its upkeep kept to itself. It stays because users already keep their vectors in it, and a second store shows the contract holds and how one is added.

### D9. Targets

They replace those of `.claude/rules/mcp-server.md` that this changes, each held by `evals speed`:
- the first answer on a repository never indexed: within 10 s at 20k chunks and 60 s at 100k (measured idle on SQLite: 2.6 s and 19.0 s);
- the stale check of a search: within 100 ms at 100k chunks (git: 40 ms on Sentry);
- a warm search by words: within 500 ms at p95 at 100k chunks (FTS5: 265 ms at its slowest idle, 385 ms under load), what an agent's turn of seconds does not notice, in place of 80 ms at the median;
- a search after an edit of one file, and `index_status`, as today;
- embedding in the background: one child process, two threads, the lowest priority; "a full index over 20 chunks a second" goes;
- the server's memory without meaning, and the server's and the child's with it: set from the first measurement of steps 7 and 9.

### D10. Measurement

- `evals speed` reports the first answer on a cold repository by phase (files, chunks, search text, write, full-text index), the stale check and a warm search on a corpus of Sentry's size, and the memory of the server and the child with meaning on.
- `evals quality` reports words and meaning apart: code, documentation, bare names, off-topic results.
- `evals agents` takes ADR 008's open fixes, those pilot 8 already ran (arms rotated, project settings shut out, no API key, a stop on an error result or a server down, every stream kept) among them, and arms of own tools, words, and words with `by_meaning` offered.

### D11. ADR 007's turn

ADR 007 turned SQLite down for its keyword channel: 96 ms at the median at 100k chunks against LanceDB's 7, and a fast form that ranks worse. Against an agent's turn of seconds that cost does not show, and the target moves (D9); with words the default channel, SQLite's exact full-text index after every write, its chunks without vectors and an install 422 MB smaller decide. ADR 007's contract, directory and suite (D1 to D5) become this decision's steps; the identity's split and the vector cache follow them.

### D12. A faster first index and search

A first index of Sentry, idle, on SQLite: chunking 10.6 s, writing and the full-text index 6.4 s, finding the files 1.1 s, the search text 0.9 s. Each search after D4: git about 40 ms, FTS5 105 ms at the median (131 ms on the agents' own queries, 265 ms at the slowest). A probe on Sentry measures each change below before it is built; the estimates are not measurements.

- **Chunking, profiled first**: whether tree-sitter's parse, in C, or the walk of its nodes in Python takes the time. Then the files go to a pool of processes, at most half the cores: the load runs inside a call the user waits for, seen and seconds long, and gains less on a loaded machine, where chunking ran five times its idle time. If the walk takes the time, tree-sitter's queries do it in C. The pool makes the search text too. Estimate: 11.5 s to about 2 s.
- **Writing**: FTS5 reads the text from the chunks table (an external-content table) instead of holding a copy. A first build writes a new database under a temporary name without waiting for the disk at each step (`synchronous=OFF`) and renames it into place when whole, so a crash leaves only a file to delete. Estimate: 6.4 s to 3-4 s.
- **The files of a first index**: git lists them (`ls-files`, 0.04 s on Sentry) and the server stats them; the walk stays for what git cannot answer. Estimate: 1.1 s to about 0.2 s.
- **Together**, estimated: Sentry from 19 s to 5-7 s, algotrading from 2.6 s to about 1 s.
- **Each search**: one git call in place of three (D4), from 40 ms to an estimated 10-30 ms. A query's words that most chunks hold (half of them, as the first candidate) are left out of it: they barely move BM25 and make FTS5 score nearly every chunk; held to ADR 006's criterion, which kept stop words in the index (stage 6).
- **Not lower**: FTS5 never prunes weak candidates, which gives LanceDB its 9 ms; that on SQLite needs an index of our own in memory, and bm25s built slower than FTS5 (43 s on Sentry under load). Below about 150 ms an agent's turn of seconds does not notice.

### Outside this decision

Dependencies' code: discovery leaves out `.venv`, `site-packages` and `node_modules` (`config.py`), and a chunk's path must lie under the project's root (`Chunk.path`). Indexing them is a feature of its own, with a change of the domain model; TODO holds it.

## Order

| Step | Content |
|---|---|
| 1 | FTS5 in the Python a user gets: a CI job on macOS and Windows checks `sqlite_compileoption_used('ENABLE_FTS5')` in uv's managed Python 3.14; found on Linux (SQLite 3.50.4), not yet on the others. If one lacks it, that platform's default is decided there |
| 2 | `evals speed` measures the first answer on a cold repository by phase, and the stale check and a warm search at Sentry's size, before anything moves them; a probe on Sentry measures D12's options (chunking profiled, the pool, FTS5's forms) and the vectors' rate on two threads |
| 3 | Staleness from git in one call (D4): today's product, whose warm search on Sentry spends a second walking the tree |
| 4 | A faster first index (D12): chunking by its measured cause, the pool, the files from git; today's product too |
| 5 | The contract, the directory and the suite over a factory, LanceDB the only store; no behaviour changes (ADR 007, D1, D2, D5) |
| 6 | The SQLite store passing the suite, the default, FTS5 reading the chunks table, a first build under a temporary name; LanceDB an extra; the identity split; search behaves as before |
| 7 | Words by default with an exact name first; common words left out of a query, on the harness; `by_meaning`, embedding inside the call as today; `similarity` out of the hit; the word index whole under the index lock, its wait unbounded |
| 8 | Agents, with the owner's word, as it spends the subscription: own tools, words, and words with `by_meaning` offered, on algotrading and Sentry, with questions written to need meaning, through the pilot's runner. It settles the description and the setting's default |
| 9 | Meaning in the background (D6): the child process, the parts and their order, coverage, the share and rate in `index_status`; the cache by content unless step 8 shows agents do not take meaning, when it waits for a measured need; the 64-token head on the harness |
| 10 | `evals quality` in both modes; `evals agents` fixed by ADR 008's open list |
| 11 | Tool descriptions, the skill, the server's instructions, README, CHANGELOG, `.claude/rules/mcp-server.md`'s targets |

Each of steps 3 to 10 is a chunk of work with one independent review over its range; a Fable review closes the block.

## Alternatives Considered

- **A faster transformer** (int8, a 64-token head, threads): two to four times, not a hundred; Sentry stays at minutes on every core.
- **Vectors in the background by default** (the Fable reviewer's second tier): agents showed no gain from them, and they cost hours of a capped CPU on a large repository.
- **A static model as the meaning** (potion-code-16M-v2): about a hundred times the transformer's speed, so no background and no parts; but against words alone it is better in 10 cells of 15, equal in 1 and worse in 4 (fd's bare names and held-out code, this repository's documentation and bare names), and MiniLM leads it in 13 and ties in one. Kept as the way out if step 9 shows the transformer's cost cannot be borne.
- **Words in parts too**: parts, a background run of words and coverage by part, to save 13 s once per large repository; the owner chose the word index whole.
- **Threads at the lowest priority**: the work runs on onnxruntime's pool, whose priority Python cannot set, and the model would stay in every server.
- **A child per repository**: a session with meaning on in three repositories would load three models; one child takes them in turn.
- **A rule against off-topic hits now**: no agent has shown the harm, and ADR 006's criterion (fewer than 10 more off-topic results) is unlikely to pass on the near questions at 140 against 1 to 14.
- **A remote embedding API**: the code leaves the machine.
- **Indexes shared across a team** (Cursor's): needs a server; vectors kept by content give the same within one machine.
- **Words alone, meaning dropped**: the harness shows its niche, questions without the code's words (fd's code 0.53 against 0.16).
- **LanceDB by default**: 422 MB more for every user; rows not yet folded into its full-text index score on another scale; chunks without vectors need a schema change there too.
- **A daemon per user**: one embedder and one watcher for every session, at the cost of a process's life on three platforms; later, if one child per server falls short.

## Consequences

- A first search costs seconds on any repository up to Sentry's size (2.6 s on algotrading, 19 s on Sentry; an estimated 1 s and 5-7 s after D12), and a warm search on Sentry stops spending a second walking the tree.
- What it costs a user by words: a first index of one core for 2.6 s on algotrading and 19 s on Sentry (several cores for seconds after D12); each search about 40 ms of git and 15 or 105 ms of FTS5 at the median; an edit about 150 ms; the index on disk 48 MB and 276 MB in the probe, which held the text twice; nothing in the background. The server's memory without the model is not yet measured; with it today, 571-693 MB.
- What it costs with meaning: about 1.5 GB while the vectors are made (the server 571-693 MB, the child 785-860 MB at its peak); two of this machine's 16 threads at the lowest priority, for a time not yet measured, estimated at 40-60 min on algotrading and 3-5 h on Sentry, a quarter of that with the 64-token head; 1.5 KB a vector (29 MB and 130 MB), the cache's copy beside them; the model's weights, about 90 MB, fetched once.
- By default the harness's questions rank worse: code by 0.14 chunk MRR in every corpus, documentation by 0.09 to 0.19, questions with none of the answer's words by 0.15 to 0.40, held-out code on click from 0.80 to 0.53 and on this repository from 0.87 to 0.62, bare names on click from 0.83 to 0.68 until the exact-name rule. Agents made up for it in pilot 8 by searching again, and meaning is one parameter away.
- The model is in a server only after a search by meaning there; the child that embeds lives as long as its session.
- Every user's index is built again once, in a new directory; without meaning that takes seconds.
- LanceDB's upkeep, its pin at 0.30.2 and pylance move into an optional store; the default store has neither, and the base install is about 420 MB smaller.
- `search_code`'s input gains `by_meaning`, its hits lose `similarity`, its answer by meaning gains coverage; the descriptions, README, the skill and the server's instructions say plainly that the default matches words and meaning is a parameter. The product keeps its name.
- Stage 7's work on the model's speed now serves meaning alone; TODO's "LanceDB index tuning" serves the LanceDB store alone.
