# Documentation System

## File Purposes
- **TODO.md** - High-level epics with rationale ("we need X because Y")
- **CHANGELOG.md** - Completed work, version-based (Keep a Changelog format)
- **docs/decisions/** - Implementation plans and architectural decisions

## Flow
TODO.md (what & why) → decisions/ (how) → task list (doing) → CHANGELOG.md (done)

## TODO.md Format
```markdown
## Pending

### Epic Title
Why we need this. What problem it solves. Business/technical rationale.

## In Progress

### Another Epic
Rationale here...
```

## CHANGELOG.md Format
Follow https://keepachangelog.com:
```markdown
## [Unreleased]

### Added
- New feature description

## [0.1.0] - 2024-01-24

### Added
- Initial feature
```

## docs/decisions/ Convention
- Filename: `NNN-short-title.md` (e.g., `001-initial-architecture.md`)
- Sequential numbering, never reuse numbers

## Decision Document Structure
```markdown
# NNN: Title

**Status**: proposed | accepted | implemented | superseded
**Date**: YYYY-MM-DD

## Context
What problem are we solving? Why now?

## Decision
What did we decide?

## Alternatives Considered
What else did we consider and why was it rejected?

## Consequences
What are the implications of this decision?
```

## Agent Instruction Files
- **CLAUDE.md** - Claude Code context, loaded every session. Keep under 80 lines.
- **AGENTS.md** - Cross-agent instructions (GitHub Copilot, Codex, Cursor). Keep in sync with CLAUDE.md.
- See `.claude/rules/agent-docs.md` for detailed maintenance rules.

## When to Create/Update
- New epic idea → add to TODO.md
- Starting implementation planning → create decisions/ doc
- Completing work → move from TODO.md to CHANGELOG.md
- Changing commands, stack, or boundaries → update both CLAUDE.md and AGENTS.md
- Always keep these in sync with actual state
