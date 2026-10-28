---
paths:
  - "src/semantic_code_mcp/server/*.py"
---

# MCP Server Rules

## Progress Notifications
- Use MCP progress notifications for long operations (model loading, indexing)
- Keep user informed during lazy model initialization

## Tool Design
- Tools should be idempotent where possible
- Return structured data (Pydantic models serialized to dict)
- Include enough context in results for Claude to use them effectively

## Performance Targets
`uv run python -m evals speed` measures them on the machine `evals/baseline/speed.json` names, at the median unless a target says otherwise (ADR 009, D9).
- At 20k and 100k chunks means algotrading and Sentry, timed by `--corpus-file <file> --words-only`; their question files are kept outside the repository.
- A change that crosses a target says why, with its figures.
- The first answer on a repository never indexed: < 3 s on the pinned corpora, < 10 s at 20k chunks, < 60 s at 100k.
- A warm search by words at p95, about the slowest question: < 150 ms on the pinned corpora, < 500 ms at 100k chunks.
- `index_status`, the stale check each search makes first: < 50 ms on the pinned corpora, < 300 ms at 100k chunks.
- A search after an edit of one file, on the pinned corpora: < 250 ms; the slowest: < 500 ms.
- A fresh server's first answer on an indexed project: < 3 s.
- A server by words, its peak memory and what it holds once its work is done: < 200 MB on the pinned corpora, < 600 MB at 100k chunks.
- A server set to search by meaning, on the pinned corpora: the model's load < 1 s; its peak memory < 1 GB; what it holds once its work is done < 500 MB.
