---
paths:
  - "tests/**/*.py"
---

# Tests

## When a test is written

- A test holds a behaviour someone can name: it does one thing — a call, a scan, an index run, a search — and checks what came of it.
- Its name states that behaviour: `tests/test_discovery.py` is the standard.
- First find the test that already holds the behaviour and extend it; a new test only where none does.
- It is written before its code and seen failing. A refactor gets no new test: the old ones pass unchanged.
- A test rewritten is seen failing too: with the line of code it holds broken, then restored.
- No test for a declaration (a field's bound, a default), a library's own behaviour, a private method, a log line or a progress message.
- A rule about how code is written is a gate in pre-commit, never a test.

## At which level

- One level, the cheapest that sees the behaviour: a chunker on a string, discovery on a directory, the index on a real LanceDB in `tmp_path`.
- What git decides is tested against a real repository (`tests/helpers.py` builds one), never a mocked subprocess.
- The real embedding model only where a test is about ranking or what is stored; it is loaded once per session (`tests/conftest.py`).

## How a test is written

- A test and its helpers hold no branch and no loop: no `if`, no `for`, no conditional expression. Compare whole lists instead.
- Cases of one rule are one test, or one parametrized test, that names every case.
- Time is set, never waited for: `os.utime` with explicit values, not `time.sleep`.
- A fake keeps what it was handed and is asserted on that; a mock's call count is never the assertion.
- The tests are held to `ruff`; no type checker or spelling check reads them.

## Runs

- Iterating: the tests of the code in hand. A commit runs the gates and collects every test.
- At a chunk's end the whole suite runs once, `uv run pytest`, before the review.
