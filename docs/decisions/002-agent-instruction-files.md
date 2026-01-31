# 002: Agent Instruction Files (CLAUDE.md & AGENTS.md)

**Status**: implemented
**Date**: 2026-01-30

## Context

Our CLAUDE.md had grown to 112 lines and included content that was domain-specific (project structure tree, data flow diagrams), stale-prone (static file trees), or redundant (listing MCP tools Claude already knows about, a rules table for auto-loaded rules). Research into best practices revealed that bloated instruction files cause LLMs to ignore instructions wholesale rather than selectively filtering irrelevant content.

Additionally, we had no AGENTS.md file for cross-agent compatibility (GitHub Copilot, OpenAI Codex, Cursor, etc.), and no rules constraining how these files should be maintained over time.

## Research Findings

### Sources

- [Claude Code Best Practices — Anthropic](https://www.anthropic.com/engineering/claude-code-best-practices)
- [Writing a Good CLAUDE.md — HumanLayer](https://www.humanlayer.dev/blog/writing-a-good-claude-md)
- [AGENTS.md Outperforms Skills in Agent Evals — Vercel](https://vercel.com/blog/agents-md-outperforms-skills-in-our-agent-evals)
- [How to Write a Great agents.md — GitHub Blog](https://github.blog/ai-and-ml/github-copilot/how-to-write-a-great-agents-md-lessons-from-over-2500-repositories/)
- [AGENTS.md Specification](https://agents.md/)
- [Impact of AGENTS.md on AI Coding Agent Efficiency — arxiv (Jan 2026)](https://arxiv.org/html/2601.20404)

### Key Data Points

- **Vercel evals**: Static AGENTS.md context achieved 100% pass rate vs 79% with dynamic skill retrieval. Agents fail 56% of the time when they must decide to fetch context.
- **Academic research (124 PRs, 10 repos)**: AGENTS.md reduces output tokens ~20% and completion time 20-28%.
- **GitHub analysis (2,500+ repos)**: Most agent files fail because they're too vague. Effective ones use specific roles, real code examples, and three-tier boundaries.
- **Anthropic/HumanLayer**: Frontier LLMs handle ~150-200 instructions consistently. Non-relevant content causes wholesale dismissal of all instructions, not selective filtering.

### Best Practices Discovered

**CLAUDE.md:**
- Keep under 300 lines (ideally <100). HumanLayer keeps theirs under 60.
- Structure as WHAT (stack, structure) / WHY (purpose) / HOW (workflows).
- Only universally-applicable content — it loads every session.
- Use progressive disclosure: domain-specific docs in separate files or `.claude/rules/`.
- Don't duplicate linters. Don't tell Claude about tools it already has.
- Pointer references (`file:line`) over embedded code snippets.
- Self-referential maintenance notes prevent bloat.

**AGENTS.md:**
- Six effective sections: Role, Project Knowledge, Commands, Code Style, Boundaries, Git Workflow.
- Three-tier boundaries: Always / Ask First / Never.
- Real code examples beat prose descriptions.
- Commands placed early for frequent reference.
- Specificity over vagueness ("React 18 with TypeScript" not "React project").

**Maintenance:**
- Prune on every edit — review and remove outdated content.
- Treat as code: review when things go wrong, test by observing behavior.
- Don't auto-generate — manually curate.
- No static project trees (go stale quickly).

## Decision

1. **Rewrite CLAUDE.md** to ~70 lines: keep overview, commands, tech stack; add boundaries and maintenance section; remove project tree, data flow, MCP tools list, rules table, architecture decisions, and documentation structure sections.

2. **Create AGENTS.md** at repo root following the cross-agent convention with overview, build/test commands, tech stack, code style example, conventions, and boundaries.

3. **Create `.claude/rules/agent-docs.md`** glob-scoped to CLAUDE.md and AGENTS.md with maintenance rules: line limits, prune discipline, sync requirements, no static trees.

4. **Update `.claude/rules/documentation.md`** to include CLAUDE.md/AGENTS.md in the documentation flow.

## Alternatives Considered

**Keep CLAUDE.md as-is**: Rejected because it contained stale-prone content (static tree), redundant info (MCP tools list, rules table), and domain-specific content that doesn't need to load every session.

**Only use AGENTS.md, drop CLAUDE.md**: Rejected because CLAUDE.md supports Claude-specific features (rules references, progressive disclosure via `.claude/rules/`) that AGENTS.md doesn't.

**No maintenance rules**: Rejected because without constraints, these files grow monotonically. The research consistently identifies bloat as the primary failure mode.

## Consequences

- CLAUDE.md is 40% smaller and every line prevents a concrete mistake.
- Cross-agent compatibility via AGENTS.md (GitHub Copilot, Codex, Cursor).
- Maintenance rules prevent future bloat by enforcing prune-on-edit discipline.
- Domain-specific content lives in `.claude/rules/` with glob activation, not in every session.
- Two files to keep in sync (CLAUDE.md and AGENTS.md) — mitigated by the agent-docs rule requiring sync checks.
