# 007: The index store behind a contract, and SQLite tried behind it

**Status**: accepted; D7, SQLite, turned down on its spike (2026-10-27)
**Date**: 2026-10-27
**Continues**: 006 (D4 storage), before its release

## Context

ADR 006 built the index on LanceDB, and a noticeable part of its stages 5 and 6 went into working around how LanceDB is made rather than into search: rows a write adds score on another scale until upkeep folds them into the full-text index, so every index run ends in upkeep; rows a write replaced keep weighing in that index's word statistics, so upkeep compacts once in 20 and at once past 5%; a reader failed when upkeep pruned the version it read, so versions live 10 s; pylance came in for the folding alone; LanceDB is held at 0.30.2 exactly (0.27's filtered full-text search lost matches, 0.32 to 0.40 scan vectors two to three times slower, 0.30.0 built in another Lance); and a write orders chunks, records and manifest, while every status reads the chunks' paths again to catch a write stopped between them. LanceDB is made for large columnar datasets written in versions; this index is thousands to about 100k rows, replaced a file at a time, read by other processes, and must give a reader a file's old chunks or its new ones. That is what SQLite is made for: a write in one transaction, readers on a snapshot beside one writer (WAL), and FTS5's full-text index kept exact row by row.

Trying SQLite should be a matter of putting a second implementation behind the store's interface. Today it is not, though the store is injected (`StoreOpener`, chosen in `app.py`), import-linter keeps `lancedb` inside `storage/`, and the services, RRF fusion, the similarity floors and the name pass work on ranks and cosine similarity, so they would not change:
- The interface speaks LanceDB: `IndexStore.keep_up()` is described as folding into the index's own indexes, pruning old versions and compacting; `chunked_paths()` exists because a LanceDB write lays a file's chunks, then its record, then the manifest, and one stopped between them leaves chunks no record names; and `write` is described in that order.
- The index's life is inside the LanceDB store: the directory of each identity and layout, the write lock beside it, the manifest, the use stamp, the sweep of unused indexes and of 0.4's leftovers, about 200 of `storage/lancedb.py`'s 810 lines, none of them about LanceDB. A second store would copy them.
- The tests hold the implementation, not the contract: `tests/integration/test_lancedb_store.py` builds the class, the indexing tests build it through `open_index_store`, two of them edit its tables through `lancedb`, the fake of a cancelled write subclasses it, and unit tests import `scope_filter` and `cosine_similarities` from its module; evals import one of its constants. Tests build the store in 21 places and reach into it (its directory, its write lock) in 18.

The release that follows ADR 006 builds every user's index again once, as the embedder's version is part of an index's identity. A change of the storage's schema made before it costs users nothing more; made after it, every user builds again. So the SQLite probe comes before the release; making the store swappable costs users nothing whenever it lands, and comes first because the probe stands on it. The owner asked for this record and approved its steps on 2026-10-27: the SQLite probe before the release, as nothing presses, and compact hits in `search_code`'s answer before it too (judged by agents and turned down, ADR 008); and asked that search be judged by agents using it, as the harness ranks chunks but cannot say whether an agent finds its answer sooner.

## Decision

### D1. A contract in the index's own terms

`IndexStore` keeps its duties: the manifest, the files' records, the counts, stored vectors by chunk id, a write applied as a unit, and the candidates of both channels read from one consistent state. Its documentation names no version, fragment, compaction or order of tables, and three members change:
- `keep_up()` leaves the interface. A run's last write says so (a field of `IndexUpdate`), and a store with work at a run's end, as LanceDB folds the run's rows into its full-text index (ADR 006, stage 6: rows left unfolded score on another scale), does it inside that write, before the files' records: a write stopped before the fold leaves those files stale, so the next run writes and folds them. Today the fold is a call of its own after the last write, and a run cancelled while that write lands skips it: a review's probe cancelled an edit's run so, and left its 6 chunks outside the full-text index, 0 files stale at the next status and 0 to index at the next run, so they score on another scale until another edit. A first index is spared, as making the table builds its full-text index whole. A call after the write in `finally` would not do: under mcp's anyio cancel scope an await there is cancelled again at once, under `asyncio.shield` too (the probe). The test cancels the run's task during a write, as anyio does; the fake that raises `CancelledError` from the write's thread tells the two constructions apart no better than chance.
- `chunked_paths()` folds into `indexed_files()`, which reports, beside the record of each file, the paths the index holds chunks of without one, which a write stopped midway leaves in a store that lays chunks before records. LanceDB reads them as it does today, 29 ms a status at 100k chunks (ADR 006); a store whose write is one transaction holds none. Planning asks the store one question where it asked two, and only the store that needs the read pays for it. If LanceDB stays after D7, a flag in the files table, false before a file's chunks and true with its record, gives those paths from the read `indexed_files()` makes already, without a sentinel value in the domain. What `indexed_files()` returns, and the field of `IndexUpdate`, are domain models, shown to the owner before they are written (AGENTS.md asks first).
- `write` says what a reader finds, a file's old chunks or its new ones and never neither, and what a stopped write leaves, not the order one store lays its tables in.

### D2. The index's life in a component of its own

`storage/index_directories.py` holds `IndexDirectory`: the directory of an index, named by its store's schema version and the identity's digest as today (`schema-<version>-<digest>`), with the store's kind after them for any store but LanceDB (`schema-<version>-<digest>-<kind>`), so the sweep's one pattern finds every kind's index; the write lock beside it; the manifest, written atomically; the stamp of use; the sweep of the project's unused indexes and 0.4's leftovers, under the rules ADR 006 settled (seven days, the lock taken without waiting, the rename out of reach). A store is built on a ready `IndexDirectory` and owns only its tables and its search. Each store's class names its kind and schema version, so another store, or another schema, is another directory, and the old one goes by the sweep: no migration. LanceDB's directories keep their names exactly, or every index would be built again for nothing. The sweep runs inside the read that finds the stamp an hour old, so a search can pay for removing a whole index [read, not measured]: the step measures that on an index of 100k chunks and takes the sweep off a request's path only if it costs more than a search's run-to-run spread. The sweep renames the directory of an index nobody marked for a week, which a session idle that long may still hold open: a SQLite connection held across operations goes on writing into the deleted file without an error, while a new one at the index's path finds an empty database (a review's probe); LanceDB is safe only because each read opens its table again. So the contract has a store open its files for each operation, or check before each that they are still its directory's. Import-linter's contract that keeps the write lock (filelock) in the store names `storage/index_directories.py` instead, and AGENTS.md's architecture lines change with it.

### D3. The composition root picks the store

`app.py` builds the opener from `IndexDirectory` and a store's class; swapping stores is that class. A setting that picks the store comes only with a second store, to run `evals` through the real server in turns.

### D4. What no store owns, out of any store's module

Cosine similarity moves to `storage/similarity.py`. The reading of globs stays in `domain/globs.py`, and `storage/globs.py`'s regex serves any store with a regex function, LanceDB's `regexp_like` or a REGEXP on Python's `re`. Turning a `SearchScope` into a filter stays with each store, as each speaks its own SQL, and each applies it to both channels before they cut (ADR 006, stage 6). `evals` import no store's constants.

### D5. Tests of the contract

- One suite holds the contract through the interface alone, over a `store_factory` fixture parametrised by store: a write applied as a unit, a file's chunks replaced by path, removed files, stored vectors and their width, both channels, every kind of scope, two stores, as two processes, on one directory. About 25 of `test_lancedb_store.py`'s 36 tests move there.
- `IndexDirectory` has tests of its own: the sweep, the stamps, the lock, 0.4's leftovers.
- What only LanceDB does (folding, compaction, its fragments; about 10 tests) is tested apart, for LanceDB alone, and leaves with it if D7 takes SQLite.
- The indexing and search tests take their store from the factory; the two that edit tables directly are rewritten to what they mean; the cancelled write's fake wraps any store.
- Any text a question can be gets an answer from every store: a property test of the contract, as a question wholly in quotes failed LanceDB's, read as a phrase its index cannot answer (fixed before these steps, the keyword channel now handing LanceDB a match of words).
- An import-linter contract keeps stores from importing one another.

The factory comes first, before D2 changes how a store is built: 12 of the 21 places that build a store build the class itself or subclass it (9 more go through `open_index_store`, whose signature D2 can keep), and the contract suite starts from the factory anyway.

### D6. Versions held by what makes their output

The version tests of ADR 006 hold the cases a hand-picked sample shows, and a review found nine chunker rules and the full-text index's settings outside it. Each version constant instead records a digest of what makes its output: for chunks, the chunkers' modules and the grammars' versions; for stored rows, `search_text.py`, the store's columns and its full-text index's parameters (as LanceDB's `index_stats` or FTS5's options give them); for vectors, the embedder's module and the versions of onnxruntime and tokenizers. The digest reads the chunkers' modules, and the domain's classes they read (`Chunk`, `ChunkKind`, `Language`), as syntax trees without docstrings and comments: whole modules and what they import would have moved it on 23 of this branch's 133 commits, which touched the domain and no chunker (a review's count). A gate under `scripts/` fails when a digest moves without a new version, or without a recorded "output unchanged" for that digest: a rule about how code is changed is a gate, not a test. The samples stay, as the evidence of what changed. D6 follows D7, as its digest of stored rows reads the full-text index's parameters from the store that stays.

### D7. SQLite, tried behind the contract

`SqliteIndexStore`, a second implementation:
- one database file in WAL mode; a write is one transaction (the files' chunks replaced, their records, the manifest), so the guard every status pays today, reading the chunks' paths again for a write stopped between chunks and records, costs it nothing;
- the keyword channel on FTS5 over `search_text`, its tokenizer `unicode61`, with `porter` where it holds, checked to split identifiers as `search_text` expects and to keep stop words (ADR 006, stage 6);
- vectors as float32 blobs, searched by a matrix kept in the process and refreshed when `PRAGMA data_version` says another connection wrote (below); `sqlite-vec` only if that fails a measure, as loading extensions is not in every Python build;
- the scope as SQL, globs through a REGEXP function on Python's `re`.

Known before the probe, and built in from its start:
- FTS5 joins a query's words by AND, where LanceDB's full-text query joins them by OR (`MatchQuery`'s default), so a question would find almost nothing. The keyword channel splits the query into tokens by `unicode61`'s rule, letters and digits, and joins them by OR, each quoted, which also keeps FTS5's own syntax out of an agent's question (`-x` in one failed with `no such column: x`, probed on SQLite 3.50.4). A word with punctuation quoted whole would be a phrase (`"foo.bar"` matches adjacent tokens only, and fails where the index keeps no positions).
- FTS5 scores every row that matches, and an OR of every word of a question, stop words kept, matches nearly every row: at 100k chunks it took 107-142 ms a question against 7-12 ms for LanceDB 0.30.2 on the same rows, and 4-34 ms with the stop words left out of FTS5's query (a review's probe, on a machine under load). So the query's form is settled first, on the harness, the stop words left out of the query and kept in the index among the forms; and the criterion measures the keyword channel and a whole search at 100k chunks, not only on the corpora, where no such cost shows.
- The matrix and the full-text query read one state: the store reads `PRAGMA data_version` inside the transaction its full-text query runs in, and refreshes the matrix there when another connection wrote; read after `BEGIN`, it pins the snapshot (the probe). The value does not move for the connection's own writes, so a write marks its own matrix stale.
- The matrix is refreshed by rows, for any writer, not read whole: the database keeps a generation and the chunk ids each write changed, read inside the snapshot, since another session's server writes the same index as often as this one. Reading it whole took 223 ms at 100k chunks (the probe), against a target of 250 ms for a search after an edit.
- The scope applies to the matrix before it cuts, as a mask from one SQL query over the chunks' columns; a REGEXP costs less over the files' distinct paths than once per chunk.
- To settle in the probe: Python's `sqlite3` keeps a read snapshot open until it commits, unless in autocommit; WAL's own locks want a busy timeout beside the file lock; LanceDB stems by Snowball's English and drops tokens over 40 characters, FTS5's `porter` is Porter's and keeps them.
- uv's CPython 3.14.3 on Linux has FTS5 (SQLite 3.50.4, probed); CI runs on Linux alone, so the macOS and Windows builds are checked before code is written.

It is built as a second implementation behind the contract, on a branch or worktree, and reaches the main line only if taken. It passes the contract suite first. Then `evals quality` and `evals speed` run in turns against LanceDB through the server, on a cooled machine: `evals speed` gains the runs in turns this needs (ABBA, each begun as cool and idle as the machine gets within a cap), over the setting of D3, where stage 7 ran them from shell scripts outside the repository. SQLite is taken when, by ADR 006's criterion, neither kind of question nor either wording group loses 1.0 or more in chunk reciprocal-rank sum, bare names lose less than 1.0 and the off-topic questions get fewer than 10 results more; when warm search, a search after an edit and a full index are no slower beyond the spread of the turns, on the corpora and at 100k chunks; when agents find their answers as often, at no more tokens, as with LanceDB (step 1); and when the install and the memory a server holds after its work are smaller. Then LanceDB, pylance and pyarrow go (the install, by estimate, from about 600 MB to about 250), the import at start grows lighter, an index opens in any `sqlite3`, and the store's kind names new directories. Otherwise LanceDB stays, and D1 to D6 stand on their own. The probe is bounded by these criteria, not run until SQLite wins, so the release does not drift.

### D7's result: SQLite turned down (2026-10-27)

A throwaway spike at 100k chunks (windows of 12 lines of the installed packages' Python, the harness's 66 questions, SQLite 3.50.4, `unicode61`):
- The keyword channel: FTS5 with every word of a question joined by OR took 96 ms at the median, 135 at p90 and 385 at most, against 7, 11 and 16 for LanceDB 0.30.2 on the same rows; with the stop words left out of the query, 14.5, 40 and 77; with the words of more than 5% of the chunks left out, 33, 68 and 112. On one query the engines rank alike: FTS5's first 20 with every word shared 91% with LanceDB's, the fast forms 62-63% with FTS5's own.
- The fast form costs ranking: leaving the stop words out of the query, tried on LanceDB through `evals quality`, lost 1.75 in chunk reciprocal-rank sum on code questions (37.09 to 35.33) and 2.5 on documentation (12.0 to 9.5), and the near off-topic questions got 26 results against 19, failing ADR 006's criterion on both kinds.
- The matrix held: read whole in 263 ms, refreshed by rows after another connection's edit of 50 chunks in 27 ms (most of it copying the matrix, which spare rows would avoid), searched in 6 ms, 12 within a scope of 30%.

So SQLite's keyword channel is as good and 13 times slower at 100k chunks, or as fast and worse: D7 ends, and LanceDB stays. What SQLite would bring beside speed, a write in one transaction, would spare the store's code a guard that already works (the chunks' paths read again at each status, 29 ms at 100k chunks) and nothing a user sees; the owner saw no reason in it either (2026-10-27). Not measured: the full form at the usual 2k to 20k chunks, where its cost, which grows with the rows a word matches, may fit the targets; with a slower worst case and no gain a user sees, it was not pursued.

With one store, D1 to D6 shrink: D2's directories need no kind in their names, D3's setting never comes, D5's suite runs over LanceDB from one fixture with no parameter of stores, and no contract between stores is needed; D6's digest of stored rows reads LanceDB's `index_stats`. D1's flag in the files table may replace the read of the chunks' paths, weighed in its step.

### Order

| Step | Content |
|---|---|
| 1 | Search judged by agents, in an ADR of its own: agents given questions whose answers the harness keys, with Grep alone and with this search, scored by their answers, tokens and tool calls, each question asked several times; a pilot of about ten sessions first, to price a run on the owner's subscription before any larger one. It comes first, as no change of the product is shown to help an agent without it |
| 2 | D5's fixture: every test takes its store from one fixture |
| 3 | D2: `IndexDirectory` out of the LanceDB store |
| 4 | D1 and D4: the fold inside a run's last write; `indexed_files()` with the unrecorded paths; the interface in the index's own terms; shared helpers out; evals without a store's constants |
| 5 | D5's split: the contract's tests apart from LanceDB's own |
| 6 | D6: versions held by digests |

The spike of D7 ran and ended it (above). The steps fall into chunks of work, each reviewed once, over its whole range, after its last commit: search judged by agents (1), the store's contract (2 to 5), the digests (6). Steps 2 to 5 change no behaviour but the fold after a cancelled run: every test still holds what it held, moved onto the fixture; the two that edited LanceDB's tables are rewritten to what they meant, and the cancelled write's fake wraps any store. By estimate, steps 2 to 5 take about a session. What comes after them before the release, and is not the store's, is in ADR 006's stages.

## Alternatives Considered

- **Keep LanceDB and the upkeep built around it.** It works and is measured; but every version of LanceDB is a new probe of its scan speed and its filtered search, pylance and pyarrow stay in the install, and the upkeep rules stay to keep.
- **Swap the store without the contract first.** Faster to try, but the comparison would rest on tests written for one implementation, and the index's life would be copied into the second.
- **DuckDB.** Its full-text index is built whole, not kept up row by row, and a database file takes one writing process at a time with readers kept out, where a project has a server per session.
- **Tantivy for words and a vector library beside it.** Two stores with no transaction between them bring back the ordering SQLite removes.
- **An ANN index (usearch, hnswlib) now.** A scan answers within the targets at the corpora's sizes; an ANN index waits for a project past a few hundred thousand chunks.

## Consequences

- A second store, should one come, is a class behind a tested contract, and `storage/lancedb.py` holds only what is LanceDB's.
- LanceDB stays, and with it its upkeep, its pin at 0.30.2, pylance and pyarrow in the install, ADR 006's open item on the upkeep counter and TODO's "LanceDB index tuning".
- A daemon per user, which would give each index one writer and a file watcher, is a decision of its own after the release.
