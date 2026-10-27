"""What the server tells a client about its tools when a session starts.

A client shows these instructions beside the tools whether or not the agent loaded the
skill, so they hold what an agent needs before its first call: when to search by meaning
and when to Grep, what `project_path` names, which languages are indexed, and that
indexing needs no call of its own. `scripts/check_skill.py` holds the languages and the
tools they name to the server's own.
"""

SERVER_INSTRUCTIONS = """\
Semantic code search over a project's Python, Rust and Markdown files: `search_code` finds \
functions, classes, methods and document sections by what they do, when you do not know \
their names.

- Use `search_code` for behaviour or a concept you cannot name ("where failed requests are \
retried"). Use Grep for an exact identifier, a string or a regex, and for every call site: \
`search_code` returns only its best matches.
- `project_path` is absolute: a repository's root, or a directory inside it, which keeps \
the search to that directory. A repository keeps one index, whichever directory you name. \
Left out, it is the client's workspace root: where Claude Code was started.
- Indexing is automatic: the first search indexes the project, each later one the files \
changed since. Call `index_codebase` with `force` only after mass renames or when results \
point at code that is gone; `index_status` reports an index without changing it.
- A result is a line range of a file: Read it before quoting when `truncated` is true.
"""
