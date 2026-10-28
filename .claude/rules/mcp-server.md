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
