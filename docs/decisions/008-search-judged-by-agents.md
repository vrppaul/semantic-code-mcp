# 008: Search judged by agents

**Status**: proposed
**Date**: 2026-10-27
**Continues**: 006 (D12, the harness), 007 (step 1)

## Context

`evals quality` ranks chunks: it says where a question's answer lands among the hits of `search_code`. It cannot say whether an agent that has the server finds its answer sooner, or at all, than one with Read, Grep and Glob alone, and that is what the server is for. Several changes ahead are judged by nothing else: compact hits change what an agent reads after a search, not the ranking; the skill changes which tool an agent picks and how it asks (ADR 006 lists it as never run with and without); and a store or a model could rank alike and still cost an agent more turns. The owner asked for search judged by agents themselves (2026-10-27), run on his subscription.

## Decision

### D1. Tasks with keys

The harness's chosen questions, code and documentation, each asked as a task: find what does this in the repository, and end the reply with one line of JSON naming the file from the repository's root and the function, method, class or section. The answer is scored against the question's keys (`evals/corpora/*.toml`) without a judge: the file is right when a key names it, the answer right when a key names that file and that symbol, its qualifier aside (`ProgressBar.generator` reads as `generator`). An answer no key names is listed for judging, as the harness pools its keys, and goes into the keys once judged. The held-out questions stay held out.

### D2. Arms

Each task runs in each arm with the same prompt, which names no tool:
- **own tools**: Claude Code's Read, Grep and Glob, and its shell for reading (`rg`, `grep`, `find`, `ls`, `cat`, `head`, `tail`, `sed -n`, `wc`, `git grep`, `git log`);
- **with search**: the same, and this server's `search_code` and `index_status`;
- **with search and the skill**, once the first two are measured: the skill loaded from the repository (`--plugin-dir`).

No arm edits, fetches from the web or starts subagents, so a session's cost is its own.

### D3. Sessions alike, and apart from the owner's own

Each session is Claude Code headless (`claude -p`, `--output-format stream-json`), started in the corpus's checkout. It runs in a config directory of its own, holding only a link to the owner's login, so none of the owner's instructions, memory, hooks, plugins or MCP servers reach it: his `CLAUDE.md` asks an agent to wait for his word before acting, which would end a task before it starts. `--bare` would isolate it too, but takes an API key, not a subscription's login. The arm's MCP config is the only one (`--strict-mcp-config`). The server is the checkout of this repository under test (`uv run --directory`), its index built before the first session into the run's own cache, so no session pays a first index. Each session runs under a memory cap.

### D4. What a session is scored by

Whether its file and its answer are right; its tokens (input, cache read, cache creation, output) and the cost Claude Code reports; its turns; its tool calls, by tool; its time. The figures of an arm are each task's, and their sum and spread over the tasks.

### D5. Noise

An agent's path through a task varies from run to run, so each task runs several times in each arm, and a difference counts only beyond the spread of those runs. How many runs, and the criterion a change is held to, are set from the pilot's spread (below).

### D6. Where it lives

`evals agents` beside `quality` and `speed`, in `evals/agents.py`: a run names its model and arms, writes its sessions to `evals/results/`, and saves a baseline (`evals/baseline/agents.json`) when asked. It measures the product, so it is code of the repository, tested where it scores.

### D7. Pilots (2026-10-27, Sonnet, each question once per arm)

1. Five of click's code questions, own tools against with search: 10 of 10 right, each in 2 turns and one Grep, `search_code` connected and never called. The agent wrote names it knew (`def (echo_via_pager|_pipepager|...)`): models know click, so its questions test memory, not search.
2. Four of this repository's questions whose wording holds none of the answer's words, two of code and two of documentation, in all three arms: 12 of 12 right, 2 or 3 turns, one Grep and at most one Read each; `search_code` never called, the skill loaded and never used. The agent turned each question into an alternation of likely words (`ChromaDB|Chroma|FAISS|Qdrant|Milvus|pgvector`, `code block|fenced|snippet`), which a repository of 61 files answers at once.
3. The nine questions again, the prompt telling the agent to start with `search_code`: 9 of 9 right, each from one search, a Read never needed; but each session took 3 turns, as Claude Code defers an MCP server's tools and the agent loads them first (`ToolSearch`), and the search's answer, ten chunks of up to 50 lines, is longer than Grep's lines. Per session on click: $0.067-0.076 against $0.043-0.051 with own tools; on this repository $0.073-0.080 against $0.057-0.075.

4. The same, the server's MCP config marked `alwaysLoad`, so Claude Code puts its tools in the prompt rather than behind tool search: 9 of 9 right, 8 of them in 2 turns and one search, no `ToolSearch`. Per session on click $0.056-0.064 (the first $0.118, as its cache was new), on this repository $0.062-0.077: within the own tools' range here, a quarter above it on click. Claude Code reads a tool's own `_meta` too (`anthropic/alwaysLoad`, `anthropic/searchHint`), so the server can ask for it without the user's config; loaded always, the tools' descriptions sit in every request of every session, searched or not.

5. The owner's algotrading repository, private, so no model has seen it: 1,399 files indexed, 19,432 chunks. Fifteen questions written from chunks sampled at random (eleven of code, four of documentation), worded without their names, each once in three arms, the server marked `alwaysLoad`:

   | | own tools | with search | search first |
   |---|---|---|---|
   | right | 12 of 15 | 14 of 15 | 14 of 15 |
   | turns, mean (range) | 4.7 (3-9) | 2.1 (2-3) | 2.0 |
   | context read from cache, mean | 107k | 49k | 49k |
   | cost, mean | $0.116 | $0.103 | $0.103 |
   | time, mean | 15.8 s | 11.7 s | 10.7 s |
   | sessions that searched | 0 | 15 | 15 |

   Unprompted, the agent took `search_code` in every session: here the search halved the turns and the context read, and cut the cost by a tenth and the time by a quarter. The one answer scored wrong in the search arms, `StrategyRepository.get_enabled_by_setup_id` for a question that asks for the enabled strategies, is right by its words and goes into the key; of the own tools' three, one took a method of the right class and one another section of the right file.

6. The same fifteen, with search, its tools deferred as Claude Code defers them by default: 14 of 15 right, the search taken in 10 of 15 sessions (Grep in the rest), 2.7 turns and 74k of context read on average. So an unknown repository is what makes an agent search; a tool in the prompt makes it search every time and spares the turn of looking the tool up. The server now marks `search_code` so (`_meta` `anthropic/alwaysLoad`), and three of those questions asked again with no `alwaysLoad` in the config each went to the search at once, in 2 turns.

   A session's cost as Claude Code reports it is a third to a half the writing of the prompt into the cache, at twice the price of input, and how much a session writes depends on the order sessions run in: about 24k tokens each in pilot 5, whose arms took turns, against 11.5k in pilot 6, which ran one arm. Costs are compared within one run alone; across runs, the turns and the context read are.

The first 31 sessions cost about $1.8 as Claude Code reports cost, at 7 to 22 s each, and moved the five-hour window by about two points with the author's own work beside them. So on repositories of 60 to 130 files an agent with Grep finds what these questions ask in one search of its own and never picks this server, skill or not; made to use it, the server finds the same answers in one call, at the same cost or more. What remains to measure is where Grep's guesses fail: a repository of thousands of files, whose words a model cannot guess and does not know, with questions keyed for it; and what compact hits save of the search's answer.

## Alternatives Considered

- **The owner's own configuration.** His instructions, hooks and plugins would reach every session, and his `CLAUDE.md` asks the agent to wait for his word.
- **`--bare` with an API key.** Isolates the session with no link to a login, but bills an API account; the owner chose his subscription.
- **Coding tasks graded by their tests (SWE-bench's way).** Nearer to real work, but each needs a task, a failing test and a sandbox, and costs many times more per session; locating code is where the server acts, and the harness's keys already exist.
- **An LLM judging every answer.** The keys judge most answers for free; a judge reads only the answers outside them, as the harness's pooling does.

## Consequences

- A change of the server can be shown to help, or not, an agent's work, not only the rank of a chunk.
- Each run spends the owner's subscription; the pilot sets its size.
- Models know click and fd from their training and may answer from memory, which favours the own-tools arm alike in both; this repository and a corpus newer than the models' training show the difference cleaner, and the results are reported by corpus.
- The keys were pooled from this server's own hits, so an agent that finds a right answer by other means scores as wrong until it is judged; answers outside the keys are listed in every run.
