# 007: The index store behind a contract, and SQLite tried behind it

**Status**: proposed
**Date**: 2026-10-27
**Continues**: 006 (D4 storage), before its release

## Context

ADR 006 built the index on LanceDB, and a noticeable part of its stages 5 and 6 went into working around how LanceDB is made rather than into search: rows a write adds score on another scale until upkeep folds them into the full-text index, so every index run ends in upkeep; rows a write replaced keep weighing in that index's word statistics, so upkeep compacts once in 20 and at once past 5%; a reader failed when upkeep pruned the version it read, so versions live 10 s; pylance came in for the folding alone; LanceDB is held at 0.30.2 exactly (0.27's filtered full-text search lost matches, 0.32 to 0.40 scan vectors two to three times slower, 0.30.0 built in another Lance); and a write orders chunks, records and manifest, while every status reads the chunks' paths again to catch a write stopped between them. LanceDB is made for large columnar datasets written in versions; this index is thousands to about 100k rows, replaced a file at a time, read by other processes, and must give a reader a file's old chunks or its new ones. That is what SQLite is made for: a write in one transaction, readers on a snapshot beside one writer (WAL), and FTS5's full-text index kept exact row by row.

Trying SQLite should be a matter of putting a second implementation behind the store's interface. Today it is not, though the store is injected (`StoreOpener`, chosen in `app.py`), import-linter keeps `lancedb` inside `storage/`, and the services, RRF fusion, the similarity floors and the name pass work on ranks and cosine similarity, so they would not change:
- The interface speaks LanceDB: `IndexStore.keep_up()` is described as folding into the index's own indexes, pruning old versions and compacting.
- The index's life is inside the LanceDB store: the directory of each identity and layout, the write lock beside it, the manifest, the use stamp, the sweep of unused indexes and of 0.4's leftovers, about 200 of `storage/lancedb.py`'s 810 lines, none of them about LanceDB. A second store would copy them.
- The tests hold the implementation, not the contract: `tests/integration/test_lancedb_store.py` builds the class, the indexing and search tests build it too, two tests edit its tables through `lancedb`, the fake of a cancelled write subclasses it, and unit tests import `scope_filter` and `cosine_similarities` from its module; evals import one of its constants.

The release that follows ADR 006 builds every user's index again once, as the embedder's version is part of an index's identity. A change of the storage's schema made before it costs users nothing more; made after it, every user builds again. So the plan puts this work before the release. The owner asked for this record (2026-10-27); its steps wait for his word.

## Decision

### D1. A contract in the index's own terms

`IndexStore` keeps its duties: the manifest, the files' records, the counts, the chunks' paths, stored vectors by chunk id, a write applied as a unit, and the candidates of both channels read from one consistent state. `keep_up()` becomes `end_run()`: the indexing service says what it knows, that a run has written its last slice, and each store decides what that asks of it. The hook stays, rather than folding into `write`, as LanceDB must fold rows after a run's last slice (ADR 006, stage 6: rows left unfolded score on another scale). The interface's documentation names no version, fragment or compaction.

### D2. The index's life in a component of its own

`storage/index_directories.py` holds `IndexDirectory`: the directory of an index, named by its store's schema version and the identity's digest as today (`schema-<version>-<digest>`), and by a part naming the store's kind for any store but LanceDB; the write lock beside it; the manifest, written atomically; the stamp of use; the sweep of the project's unused indexes and 0.4's leftovers, under the rules ADR 006 settled (seven days, the lock taken without waiting, the rename out of reach). A store is built on a ready `IndexDirectory` and owns only its tables and its search. Each store's class names its kind and schema version, so another store, or another schema, is another directory, and the old one goes by the sweep: no migration. LanceDB's directories keep their names exactly, or every index would be built again for nothing. Import-linter's contract that keeps the write lock (filelock) in the store names `storage/index_directories.py` instead, and AGENTS.md's architecture lines change with it.

### D3. The composition root picks the store

`app.py` builds the opener from `IndexDirectory` and a store's class; swapping stores is that class. A setting that picks the store comes only with a second store, to run `evals` through the real server in turns.

### D4. What no store owns, out of any store's module

Cosine similarity moves to `storage/similarity.py`. The reading of globs stays in `domain/globs.py`, and `storage/globs.py`'s regex serves any store with a regex function, LanceDB's `regexp_like` or a REGEXP on Python's `re`. Turning a `SearchScope` into a filter stays with each store, as each speaks its own SQL, and each applies it to both channels before they cut (ADR 006, stage 6). `evals` import no store's constants.

### D5. Tests of the contract

- One suite holds the contract through the interface alone, over a `store_factory` fixture parametrised by store: a write applied as a unit, a file's chunks replaced by path, removed files, stored vectors and their width, both channels, every kind of scope, two stores, as two processes, on one directory. About 25 of `test_lancedb_store.py`'s 36 tests move there.
- `IndexDirectory` has tests of its own: the sweep, the stamps, the lock, 0.4's leftovers.
- What only LanceDB does (folding, compaction, its fragments; about 10 tests) is tested apart, for LanceDB alone, and leaves with it if D7 takes SQLite.
- The indexing and search tests take their store from the factory; the two that edit tables directly are rewritten to what they mean; the cancelled write's fake wraps any store.
- An import-linter contract keeps stores from importing one another.

### D6. Versions held by what makes their output

The version tests of ADR 006 hold the cases a hand-picked sample shows, and a review found nine chunker rules and the full-text index's settings outside it. Each version constant instead records a digest of what makes its output: for chunks, the chunkers' modules and the grammars' versions; for stored rows, `search_text.py`, the store's columns and its full-text index's parameters (as LanceDB's `index_stats` or FTS5's options give them); for vectors, the embedder's module and the versions of onnxruntime and tokenizers. A test fails when a digest moves without a new version, or without a recorded "output unchanged" for that digest. The samples stay, as the evidence of what changed.

### D7. SQLite, tried behind the contract

`SqliteIndexStore`, a second implementation:
- one database file in WAL mode; a write is one transaction (the files' chunks replaced, their records, the manifest), so the guard every status pays today, reading the chunks' paths again for a write stopped between chunks and records, costs it a plain query;
- the keyword channel on FTS5 over `search_text`, its tokenizer `unicode61`, with `porter` where it holds, checked to split identifiers as `search_text` expects and to keep stop words (ADR 006, stage 6);
- vectors as float32 blobs, searched by a matrix kept in the process and read again when `PRAGMA data_version` says another connection wrote; `sqlite-vec` only if that fails a measure, as loading extensions is not in every Python build;
- the scope as SQL, globs through a REGEXP function on Python's `re`.

It is built as a second implementation behind the contract, on a branch or worktree, and reaches the main line only if taken. It passes the contract suite first. Then `evals quality` and `evals speed` run in turns against LanceDB through the server, on a cooled machine. SQLite is taken when, by ADR 006's criterion, neither kind of question nor either wording group loses 1.0 or more in chunk reciprocal-rank sum, bare names lose less than 1.0 and the off-topic questions get fewer than 10 results more; when warm search, a search after an edit and a full index are no slower beyond the spread of the turns; and when the install and the memory a server holds after its work are smaller. Then LanceDB, pylance and pyarrow go (the install, by estimate, from about 600 MB to about 250), the import at start grows lighter, an index opens in any `sqlite3`, and the store's kind names new directories. Otherwise LanceDB stays, and D1 to D6 stand on their own. The probe is bounded by these criteria, not run until SQLite wins, so the release does not drift.

### Order

| Step | Content |
|---|---|
| 1 | D2: `IndexDirectory` out of the LanceDB store |
| 2 | D1 and D4: `end_run`, the interface in the index's own terms, shared helpers out, evals without a store's constants |
| 3 | D5: the contract suite and the factory |
| 4 | D6: versions held by digests |
| 5 | D7: SQLite behind the contract, measured in turns; the choice recorded here |
| 6 | Markdown sections after a setext heading take their parent (ADR 006, open; a new `CHUNKER_VERSION`) |
| 7 | A baseline of `evals speed` on the final code, on an idle machine |
| 8 | The release, on the owner's word: the `pypi` environment's required reviewer, the version (0.5.0 proposed), the tag |

Each step is a chunk of work reviewed before the next. Steps 1 to 3 change no behaviour: every test still holds what it held, moved onto the factory; the two that edited LanceDB's tables are rewritten to what they meant, and the cancelled write's fake wraps any store. By estimate, steps 1 to 3 take about a session, the SQLite probe half of one, and a full swap one to two.

## Alternatives Considered

- **Keep LanceDB and the upkeep built around it.** It works and is measured; but every version of LanceDB is a new probe of its scan speed and its filtered search, pylance and pyarrow stay in the install, and the upkeep rules stay to keep.
- **Swap the store without the contract first.** Faster to try, but the comparison would rest on tests written for one implementation, and the index's life would be copied into the second.
- **DuckDB.** Its full-text index is built whole, not kept up row by row, and a database file takes one writing process at a time with readers kept out, where a project has a server per session.
- **Tantivy for words and a vector library beside it.** Two stores with no transaction between them bring back the ordering SQLite removes.
- **An ANN index (usearch, hnswlib) now.** A scan answers within the targets at the corpora's sizes; an ANN index waits for a project past a few hundred thousand chunks.

## Consequences

- A second store, or a third, is a class behind a tested contract, and `storage/lancedb.py` holds only what is LanceDB's.
- With SQLite, each server process holds its index's vectors in memory: about 150 MB at 100k chunks in float32 (38 MB in int8), 3 to 30 MB at the usual 2k to 20k. Past a few hundred thousand chunks a scan grows slow, and an ANN index comes in.
- The keyword channel's ranking moves with the engine: only the harness says whether it holds.
- Done before the release, a new kind of store costs users nothing beyond the one rebuild the release brings.
- If SQLite is taken, ADR 006's open item on the upkeep counter and TODO's "LanceDB index tuning" fall away. A daemon per user, which would give each index one writer and a file watcher, is a decision of its own after the release.
