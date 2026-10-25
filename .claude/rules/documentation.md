---
paths:
  - "TODO.md"
  - "CHANGELOG.md"
  - "docs/decisions/*.md"
---

# Documentation System

## File Purposes
- **TODO.md** - High-level epics with rationale ("we need X because Y")
- **CHANGELOG.md** - Completed work, version-based (Keep a Changelog format)
- **docs/decisions/** - Implementation plans and architectural decisions

## Flow
TODO.md (what & why) → decisions/ (how) → task list (doing) → CHANGELOG.md (done)

## TODO.md Format
An item done is removed, never ticked off.
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

## [0.1.0] - 2026-01-24

### Added
- Initial feature
```

## docs/decisions/ Convention
- Filename: NNN-short-title.md, as `docs/decisions/001-initial-architecture.md`
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

## When to Create/Update
- New epic idea → add to TODO.md
- Starting implementation planning → create decisions/ doc
- Completing work → remove it from TODO.md, record it in CHANGELOG.md under `[Unreleased]`
- A plan that changes → change its decision document; a decision replaced → mark it superseded and name its successor
- Changing commands, stack, or boundaries → update AGENTS.md
- Always keep these in sync with actual state
