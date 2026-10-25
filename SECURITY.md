# Security Policy

## Reporting a Vulnerability

Report a vulnerability privately through [GitHub Security Advisories](https://github.com/vrppaul/semantic-code-mcp/security/advisories/new). Do **not** open a public issue for it.

## Scope

semantic-code-mcp is a local MCP server that a coding agent starts over stdio. What it touches:

- **Reads** the files of the projects it is asked to search: files with an indexed extension (`.py`, `.rs`, `.md`) that git lists, or that a file-system walk finds, minus the ignore patterns. It never follows a symlinked directory or a file that resolves outside the project root.
- **Writes** its index under `~/.cache/semantic-code-mcp/` (or `SEMANTIC_CODE_MCP_CACHE_DIR`), or under `.semantic-code/` in the project when `SEMANTIC_CODE_MCP_LOCAL_INDEX` is set.
- **Network:** none of its own. On first use sentence-transformers downloads the embedding model from Hugging Face.
- **Processes:** it runs `git ls-files` with a fixed argument list, never through a shell.
- The index holds the text of indexed files. Whatever a project keeps in an indexed file — a token pasted into a Markdown note, say — is stored in the index on the same machine.
