"""What the server tells a client about its tools when a session starts.

A client shows these instructions beside the tools whether or not the agent loaded the
skill, so they hold what an agent needs before its first call: how to word a search and
when to Grep instead, what `project_path` names, which languages are indexed, and that
indexing needs no call of its own. `scripts/check_skill.py` holds the
languages and the tools they name to the server's own.
"""

SERVER_INSTRUCTIONS = """\
Code search over a project's Python, Rust and Markdown files: `search_code` finds \
functions, classes, methods and document sections by the words their code uses.

- Use `search_code` to find where behaviour or a concept lives, or a definition by its name: \
write the identifiers and terms the code would use ("retry backoff delay"); on a miss, other \
words. Use Grep for an exact string or a regex, and for every call site: `search_code` \
returns only its best matches.
- `project_path` is absolute: a repository's root, or a directory inside it, which keeps \
the search to that directory. A repository keeps one index, whichever directory you name. \
Left out, it is the client's workspace root, where the client names one and it lies in a \
git repository: the directory Claude Code was started in. Give it whenever the work lies \
outside that directory, or that directory lies in no repository.
- Indexing is automatic: the first search indexes the project, each later one the files \
changed since, renames and deletions among them. Call `index_codebase` with `force` only \
when results point at code that is gone; `index_status` reports an index without changing it.
- A result is a line range of a file: Read it before quoting when `truncated` is true.
"""
