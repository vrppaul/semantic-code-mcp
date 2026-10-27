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
`uv run python -m evals speed` measures them on the pinned corpora, on the machine `evals/baseline/speed.json` names; stage 7 set each from its slowest corpus, with room for run-to-run spread (ADR 006).
- A change that crosses a target says why, with its figures.
- Warm search, plain and narrowed, median: < 80 ms; its p95, about the slowest question: < 150 ms
- Search after an edit of one file, median: < 250 ms; the slowest: < 500 ms
- `index_status`, median: < 50 ms
- A full index: > 20 chunks a second
- A fresh server's first answer on an indexed project: < 3 s; the model's load: < 1 s
- The server's peak memory: < 1 GB; what it holds once its work is done: < 800 MB
