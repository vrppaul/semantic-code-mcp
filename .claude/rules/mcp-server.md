---
paths:
  - "src/semantic_code_mcp/server.py"
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
Unmeasured until the benchmark of ADR 006 (D12) exists; v0.4.0 misses all four (ADR 006, Appendix B).
- Search latency: < 100ms (excluding cold start)
- Index speed: > 50 files/sec
- Cold start (model load): < 3s
- Memory: < 500MB during search
