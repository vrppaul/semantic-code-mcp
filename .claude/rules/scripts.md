---
paths:
  - "scripts/*"
---

# The scripts

The gates of our own, which no linter here can be told to keep. How each is built is said at its top.

- A script says at its top why it exists and how it is run, what it prints and every code it exits with.
- It is run in the repository's root and does its whole job in one process start.
- A gate taken from claude-review says so at its top, and what it left out.
- `scripts/check_instruction_files.py` is claude-review's, taken whole: it is changed there and copied here, and its tests are there.
