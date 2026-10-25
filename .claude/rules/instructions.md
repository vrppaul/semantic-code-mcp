---
paths:
  - "AGENTS.md"
  - "CLAUDE.md"
  - ".claude/rules/*.md"
---

# The instruction files

`AGENTS.md` and the rules of each area under `.claude/rules/`. What `scripts/check_instruction_files.py` holds them to, and why, is said in its docstring.

- A new rule is a new bullet a reviewer can check, in the file of its area: never a sentence added to a paragraph.
- How a thing is built is said in its module's docstring, not in these files.
- What a tool can hold is a gate, not a rule: write the gate.
- `CLAUDE.md` is a symlink to `AGENTS.md`: edit `AGENTS.md`.
