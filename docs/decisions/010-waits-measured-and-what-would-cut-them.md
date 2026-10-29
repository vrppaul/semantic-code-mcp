# 010: The waits measured after words first, and what would cut them

**Status**: proposed (2026-10-28); built after ADR 009's other steps, as the owner put the optimizations last
**Date**: 2026-10-28
**Continues**: 009 (D4, D6, D9, D12, its step 7)

## Context

ADR 009's step 6 made the default search one by words. What a user waits for was then split by phase on Sentry (8,546 files, 87,694 chunks), the machine at a load of 0.4 to 1: through a server started fresh, every request profiled with pyinstrument (`SEMANTIC_CODE_MCP_PROFILE`) and its memory read from `/proc`; the stale check and a first index in process, under cProfile and with timers around each part of the write; the memory of a first index under tracemalloc. The pinned corpora (32 to 133 files) through `evals speed`.

| What a user waits for, on Sentry | Now | Where it goes |
|---|---|---|
| A first answer on a repository never indexed | 12.5 s | the process's start and handshake 0.6 s; chunking about 7 s, two thirds of it tree-sitter's parse in C, a third Python walking the trees and making chunks; the writes 4.5 s; the search 0.03 s |
| A warm search | 340-400 ms | the stale check 260-345 ms, 85 to 90% of it; the search by words 34 ms |
| `index_status` | 260-310 ms | the stale check |
| A search after an edit of one file | 800-1,140 ms | the stale check 336 ms; the index run's own discovery and plan again 358 ms; the write 64 ms; the search 34 ms |
| The server's memory | 123 MB at its start, 383 at its peak, 265-277 after a first index | at the peak every chunk of the run at once: their text 101 MB, their pydantic objects 91 MB |

- **The stale check**, 190-235 ms in process, is Python's, not the disk's: about 21,700 `lstat` calls, 2.5 a file; hashing `pathlib` paths in the plan, about a fifth; about 25,600 pydantic `SourceFile` models built each time (the index's records and the files found); git about 40 ms of it.
- **A first index's writes**, 4.7 s in 318 writes of 256 chunks: FTS5's inserts 1.38 s; each write's own cost (locks, a connection, a transaction, the manifest's rename) 1.40 s in all, about 4.4 ms a write; the search text 0.99 s; the chunks' inserts 0.71 s; the manifests 0.11 s. A probe that varied the slice beside a rising load showed nothing; this split shows the writes' count costs 1.4 s.
- **The server's start**: importing `mcp` 0.41 s, not ours; onnxruntime, tokenizers and huggingface_hub 80 ms and 22 MB, which a search by words never uses.
- An earlier reading of 140 ms "inside the server" beside the stale check was the load's: at a load of 0.4 the server spends on `index_status` what the stale check spends in process.
- On the pinned corpora a first answer takes about 1.1 s: 0.7 s the process's start, 0.1-0.3 s the index, 2 ms the search.

Not measured yet: the search by meaning at Sentry's size on step 6's tables, either store (an hour of embedding; a probe can add random vectors), and its memory with the model and the held matrix (135 MB at 88k chunks, ADR 009).

## Decision (proposed)

Each change is built with the phases split before and after, on an idle machine, in alternating turns, as ADR 009's measurements were; the figures above are what it is held to.

### O1. Watch the files instead of scanning them before every search

The server keeps a watcher of each project's files (`watchfiles`, Rust's notify: inotify, FSEvents, Windows' own), and a search's stale check reads its queue of changes: the work of the changes alone, under a millisecond. A full scan stays for the first use, a queue that overflowed, and now and then as a check. Expected: a warm search on Sentry about 40 ms instead of 340, `index_status` about 5 ms, a search after an edit about 100 ms. It costs a dependency (the owner's word first), Linux's limit of inotify watches, and the care missed events, renamed directories and a checkout of thousands of files need. A change of principle: less work, not the same work faster.

### O2. A lean scan for what scanning stays

The first scan and the checks O1 keeps: paths as strings rather than `Path`, one `lstat` a file, the files' records not built as pydantic models in the loop, the plan over a dictionary of strings. Expected: 200 ms to 60-80 ms, git 40 of it. It touches `discovery.py`, `plan_delta` and both stores' reading of records.

### O3. One pass for a search after an edit

The search's stale check hands what it found to the index run, which scans and plans again today: 300 ms less at once, with or without O1. It touches `SearchService` and `IndexingService`.

Done (2026-10-29): the stale check (`IndexingService.stale_check`) hands the run (`index_from_check`) the files its scan found, and the run plans once it holds the index. The scan is what the run took again for nothing; the plan is what another run's write since the check makes wrong, so it is made again under the hold: of two searches that checked at once, the second finds the first's work done, and one that waited for another session's first build does not build it twice. Handing the plan too, trusted while the index's manifest stayed the same, would save the plan too, 250 ms at the median in process at a load of 5 (the index's records read in 77 ms, compared in 177), but it trusts a view of the index taken before the hold for a cost O2 cuts. An edit that lands between the check and the run is the next search's. `evals speed` on Sentry by words, the old build and the new in alternating turns at a load of 0.9 to 1.7: a search after an edit 468-471 ms against 677-700, at its slowest 531-554 against 703-988; every other figure the same. Its target at 100k chunks is now 700 ms, 1 s at its slowest (`evals/targets.py`).

### O4. A first index in a pool, streamed, in few writes

Chunking in a pool of processes (ADR 009's step 7, moved here): about 7-10 s to 3-4 s by D12's probe. Streamed: a slice of files chunked, then written, then the next, instead of every chunk held before the first write: the peak from 383 MB to about 150, and what the server holds after it as low, as Python does not hand freed memory back. A whole first build (ADR 009, D12) written in large transactions: about 1.3 s less. Together Sentry's first answer about 6-7 s instead of 12.5. It needs `__main__` that does nothing on import, as a pool's workers start from a forkserver on Linux.

### O5. The model's libraries imported with the model

onnxruntime, tokenizers and huggingface_hub imported when a search by meaning first loads the model: 22 MB and 80 ms less for every server. Done (2026-10-28) with ADR 009's extra `semantic-code-mcp[meaning]`, which installs them, imported only where meaning is asked for, when a model loads or a server set to search by meaning starts: a server's imports by words 89 MB at their peak and 0.655 s at the median of twelve alternating turns, against 113 MB and 0.79 s with them.

### O6. The search by meaning's own costs, with ADR 009's step 9

The chunks without vectors handed out in pages, not all 87,694 at once, which would bring back the peak O4 removes; the held matrix in float16 (67 MB against 135 at 88k chunks), held to ADR 006's criterion on the harness.

### Order

O3 (done, as O5); then O2; then O1 on O2's figures and the owner's word on the dependency; then O4; O6 with ADR 009's step 9.

## Alternatives Considered

- **A stale check from git** (`status` and `diff` from a recorded commit): about 40 ms, turned down in ADR 009 for what git shows only with care; O1 sees every change, tracked or not.
- **A stale check cached for some seconds**: an agent that edits and searches at once would meet the old index.
- **Larger slices for every run**: a run that is not a first build keeps each slice it wrote when cancelled; only the whole build gains nothing from small ones.
- **A faster `mcp` import**: not ours to change.

## Consequences

- A warm search on a large repository costs what its search does, not what its scan does; a search after an edit, what the edit's file costs.
- A first answer on Sentry halves, and a server's memory stays near its start.
- O1 adds a part with a life of its own (a watcher per project, started and stopped with the registry) and a dependency.
