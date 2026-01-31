---
paths:
  - "CLAUDE.md"
  - "AGENTS.md"
---

# Agent Instruction File Maintenance Rules

## Line Limits
- **CLAUDE.md** must stay under 80 lines
- **AGENTS.md** has no hard limit but should stay concise and scannable

## Content Principles
- **CLAUDE.md**: Only universally-applicable content that loads every session. Domain-specific rules go in `.claude/rules/` with glob patterns.
- **AGENTS.md**: Cross-agent compatible. No Claude-specific features (rules references, etc.).
- **Never duplicate what linters/formatters enforce** — if ruff catches it, don't instruct it here.
- **No static project trees** — they go stale. Describe structure in prose or let the agent explore.
- **Pointer references** (`file:line`) over embedded code snippets where possible.

## Edit Discipline
- **Prune on every edit**: Before adding content, review existing lines and remove anything outdated or no longer preventing mistakes.
- **Ask the test**: For each line, "Would removing this cause the agent to make a mistake?" If no, cut it.
- **Keep in sync**: When updating commands, tech stack, or boundaries in one file, check and update the other.

## What Belongs Where
| Content | Location |
|---------|----------|
| Commands, stack, boundaries | CLAUDE.md + AGENTS.md (both) |
| Python coding standards | `.claude/rules/python.md` |
| Testing philosophy | `.claude/rules/testing.md` |
| MCP-specific rules | `.claude/rules/mcp-server.md` |
| Documentation flow | `.claude/rules/documentation.md` |
| Dev process | `.claude/rules/development-process.md` |
| Code style examples | AGENTS.md only (for cross-agent use) |
