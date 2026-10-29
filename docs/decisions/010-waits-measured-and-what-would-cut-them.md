# 010: The waits measured after words first, and what would cut them

**Status**: proposed (2026-10-28); built after ADR 009's other steps, as the owner put the optimizations last: O5 built (2026-10-28), O3 and O2 (2026-10-29); O1 built, rejected by its review and reverted, its construction again waiting for the owner; O4 and O6 to come
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

Built on watchfiles (`15901da`, 2026-10-29, on the owner's word for the dependency) and reverted (`b183058`) after its review (REJECT). It was faster (Sentry: `index_status` 34-36 ms against 148-152, a warm search 111-116 against 194-243, a fresh server's first answer 0.2 s slower) and wrong:
- every search failed on a project with one directory name that is not UTF-8 (`RustNotify` raised on its path and nothing caught it);
- edits were lost when reads filled inotify's queue: notify 8 asks for an event at every file opened (`IN_OPEN`), one `rg` over Sentry queued 27,905 events against a queue of 16,384, and watchfiles drops the overflow, which carries no path, so a read named fewer than 1,000 paths and was trusted (two `rg` at once lost 2 edits of 10, six lost 9);
- a directory made while the watcher started was never watched, as the directories were listed before the watches existed;
- `git ls-files --others --ignored --directory` names a directory whose files are all ignored though the directory is not (`git check-ignore` says so), and such a directory went unwatched (the committed property failed with `--hypothesis-seed=1`);
- a linked worktree's index lies outside `.git`, a file; a link's target outside what is watched was not followed, as the construction said it would be; a root made again and a nested repository's `.git` removed sent nothing the watcher kept;
- every git command that rewrites the index made the next look a full scan and a new watcher of 3,132 directories, 227 ms against the 91 a scan cost, as `RustNotify` adds no watch to a live watcher;
- tests could not fail with its 1,000-path rule, the submodule rule or the registry's close broken, and no end-to-end test ran the default setting.

`RustNotify` gives no mask, no overflow and no watch added to a live watcher, so none of the second, third and sixth can be fixed within it. The review's step back, recorded as the construction that would hold, waits for the owner's word, as it drops the dependency he approved and watches on Linux alone:
- **inotify read directly** (its system calls through `ctypes`, nothing to install): one inotify instance per project, its file descriptor non-blocking, read to its end at each check. The kernel queues an event inside the call that changed the file, so a read at the check holds every change made before it: no thread, no markers, no deadline.
- **A mask of changes alone** (`IN_CREATE`, `IN_DELETE`, `IN_MODIFY`, `IN_CLOSE_WRITE`, `IN_ATTRIB`, `IN_MOVED_FROM`, `IN_MOVED_TO`, `IN_DELETE_SELF`, `IN_MOVE_SELF`), never `IN_OPEN` or `IN_ACCESS`, so reading files queues nothing; `IN_Q_OVERFLOW` seen, and a full scan follows it.
- **Watched, then listed**: each directory's watch is added before its entries are read, so what is made meanwhile sends an event; a new directory gets its watch on the live descriptor and is listed, with no restart; `IN_ISDIR` says what is a directory.
- **What decides git's listing**, found by `git rev-parse --absolute-git-dir` and `--git-common-dir` (a linked worktree's index and excludes); directories pruned only where `git check-ignore` confirms them; a directory name the scan cannot hold (not UTF-8) left out as the scan leaves its files out; a link the scan keeps has its target watched; the root's own going, and a nested repository's directory watched alone for its `.git`.
- **Elsewhere a scan**: macOS and Windows scan at each search until a watcher there is measured.
- **Tests that can fail**: the property with the cases the review found, each rule broken and its test seen failing, and an end-to-end search after an edit with the default setting.

### O2. A lean scan for what scanning stays

The first scan and the checks O1 keeps: paths as strings rather than `Path`, one `lstat` a file, the files' records not built as pydantic models in the loop, the plan over a dictionary of strings. Expected: 200 ms to 60-80 ms, git 40 of it. It touches `discovery.py`, `plan_delta` and both stores' reading of records.

Done (2026-10-29), in two parts. First, a directory is resolved from where its parent really lives, one `lstat` each, where `Path.resolve` took one a component: on Sentry 10,000 `lstat` calls a scan instead of 21,700, and the scan 102 ms instead of 136 at the median of nine in process, the machine at a load of 0.5. Split after it, the stale check takes 183 ms: git 40 (its listing of untracked files 28, as it walks the work tree; the two listings run at once would save 3 ms, so they stay one after another), the extensions and ignore patterns 9, the files' `lstat` and their models 47, the index's records 21 and the plan 60. What was left was a `Path` a file on either side, the scan's and the index's, which every hash and comparison of the plan turned into text again. Second, on the owner's word (2026-10-29), the records went by their paths' text: the scan and both stores give each file's `FileStamp` (its modification time and size, a pydantic model, as he chose over a slotted dataclass that would cost about 20 ms less at Sentry's size) under its path from the project root as POSIX text, the plan compares those, and a `SourceFile`, now a path and its stamp, is made only for the files to index. In process the stale check takes 118 ms: the scan 96 (git 41), the index's records 15, the plan 7 where it took 60. `evals speed` on Sentry by words, before O2 and after in alternating turns at a load of 0.9 to 1.4: `index_status` 147-148 ms against 281-283, a warm search 233-243 at the median against 370-374 and 261-278 at p95 against 402-428, a narrowed search 227-232 against 359, a question of 40 words 292-293 against 444-461, a search after an edit 288-294 against 477-478 and at its slowest 340-373 against 512-542, a fresh server's first answer 1.09-1.14 s against 1.24-1.26; the first answer the same, the peak 4 MB higher (359-361 MB, a `SourceFile` the plan now makes per file of a first index), the server holding 228-230 MB after against 238-240. `evals quality` by words gives what its parent gave, byte for byte; its baseline, saved before `a394852` read a question's words once, differed by that change alone (this repository's documentation, chunk MRR 0.671 against 0.686) and is saved again. The targets at 100k chunks these moved are set again from these runs (`evals/targets.py`), the stale check's at 220 ms, below D9's 300.

### O3. One pass for a search after an edit

The search's stale check hands what it found to the index run, which scans and plans again today: 300 ms less at once, with or without O1. It touches `SearchService` and `IndexingService`.

Done (2026-10-29): the stale check (`IndexingService.stale_check`) hands the run (`index_from_check`) the files its scan found, and the run plans once it holds the index. The scan is what the run took again for nothing; the plan is what another run's write since the check makes wrong, so it is made again under the hold: of two searches that checked at once, the second finds the first's work done, and one that waited for another session's first build does not build it twice. The review of O3 and O2 found the scan wrong too in that case: a file made after the check and indexed by another run was taken back out by the run from the check, and one edited after it recorded again with its older stamp, where the scan under the hold had done neither. So the check keeps the index's manifest, read before anything else of the index, which every write that changes the index records anew, and a run that finds another one under its hold scans again: one file read, the usual path keeping the saving. Handing the plan too, trusted while the index's manifest stayed the same, would save the plan too, 250 ms at the median in process at a load of 5 (the index's records read in 77 ms, compared in 177), but it trusts a view of the index taken before the hold for a cost O2 cuts. An edit that lands between the check and the run is the next search's. `evals speed` on Sentry by words, the old build and the new in alternating turns at a load of 0.9 to 1.7: a search after an edit 468-471 ms against 677-700, at its slowest 531-554 against 703-988; every other figure the same. Its targets at 100k chunks were set from these runs, then again from O2's (below).

### O4. A first index in a pool, streamed, in few writes

Chunking in a pool of processes (ADR 009's step 7, moved here): about 7-10 s to 3-4 s by D12's probe. Streamed: a slice of files chunked, then written, then the next, instead of every chunk held before the first write: the peak from 383 MB to about 150, and what the server holds after it as low, as Python does not hand freed memory back. A whole first build (ADR 009, D12) written in large transactions: about 1.3 s less. Together Sentry's first answer about 6-7 s instead of 12.5. It needs `__main__` that does nothing on import, as a pool's workers start from a forkserver on Linux.

Measured again before it is built (2026-10-29, after O3 and O2, in process, a load of 1.3, two runs): a first index of Sentry by words takes 13.3-15.3 s, chunking 9.2-10.8 of it, 318 writes 3.8-4.1 (10-11 ms each at the median, the search text and FTS5's inserts among them), the scan and the plan 0.3; `evals speed`'s cold first answer through a server is 15.1-15.4 s. The construction to build, in this order of risk:
- **The pool's life**: `concurrent.futures.ProcessPoolExecutor` on a forkserver context, at most four workers and half the cores (D12: four chunked Sentry in 2.7 s, eight in 4.1), made for a run of at least a number of files a probe sets (Sentry's 8,546 gain; algotrading's 1,399, where the pool's first start of 1.2 s eats the gain, may not) and shut down after it, so no idle worker holds memory between runs; small runs and the tests' changing chunkers chunk in process as now.
- **A task is a batch of files**, a few dozen, chunked by a chunker the task builds from a factory the composition root hands down (a function of a module, which pickles by name): no state of a module in the workers.
- **The protocol's pipe kept clean**: a worker's file descriptor 1 is the MCP session's stdout, so the pool's initializer points it at stderr before anything runs; one stray line from a library would break the session. An end-to-end test with a chunker that prints holds it.
- **Streamed**: batches come back in the files' order; the run groups their files into write slices and writes each as it fills, in a worker thread while the pool chunks on, so a first index holds the batches in flight and a slice, not every chunk (the peak D12 measured at 383 MB).
- **Few writes for a whole first build**: where the index is empty, slices of thousands of chunks, as a build that stops keeps nothing anyway and each write costs its own few milliseconds; a run that is not a whole build keeps today's slices, each kept if it stops.
- **`__main__` imports nothing** but under its guard, as a forkserver imports the main module; a cancelled run shuts the pool down with its futures cancelled.
- **Tests**: a first index through the pool gives the chunks and records one in process gives; a run below the threshold makes no pool; a cancelled run leaves no worker behind; what a worker prints reaches stderr only.

### O5. The model's libraries imported with the model

onnxruntime, tokenizers and huggingface_hub imported when a search by meaning first loads the model: 22 MB and 80 ms less for every server. Done (2026-10-28) with ADR 009's extra `semantic-code-mcp[meaning]`, which installs them, imported only where meaning is asked for, when a model loads or a server set to search by meaning starts: a server's imports by words 89 MB at their peak and 0.655 s at the median of twelve alternating turns, against 113 MB and 0.79 s with them.

### O6. The search by meaning's own costs, with ADR 009's step 9

The chunks without vectors handed out in pages, not all 87,694 at once, which would bring back the peak O4 removes; the held matrix in float16 (67 MB against 135 at 88k chunks), held to ADR 006's criterion on the harness.

### Order

O3 and O2 (done, as O5); then O1 on the owner's word for its new construction (above), or O4 first if he puts it after; O6 with ADR 009's step 9.

## Alternatives Considered

- **A stale check from git** (`status` and `diff` from a recorded commit): about 40 ms, turned down in ADR 009 for what git shows only with care; O1 sees every change, tracked or not.
- **A stale check cached for some seconds**: an agent that edits and searches at once would meet the old index.
- **Larger slices for every run**: a run that is not a first build keeps each slice it wrote when cancelled; only the whole build gains nothing from small ones.
- **A faster `mcp` import**: not ours to change.

## Consequences

- A warm search on a large repository costs what its search does, not what its scan does; a search after an edit, what the edit's file costs.
- A first answer on Sentry halves, and a server's memory stays near its start.
- O1 adds a part with a life of its own (a watcher per project, started and stopped with the registry); built on watchfiles it added a dependency, which its construction on inotify read directly would drop.
