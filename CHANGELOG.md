# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- MCP server with three tools: `search_code`, `index_codebase`, `index_status`
- Semantic code search using sentence-transformers embeddings (all-MiniLM-L6-v2)
- LanceDB vector storage for embeddings
- Tree-sitter based AST chunking for Python (functions, classes, methods)
- Incremental indexing with mtime-based change detection
- Debug timing info in search results (status_check_ms, embedding_ms, search_ms)
- Project documentation structure (CLAUDE.md, README.md, TODO.md, CHANGELOG.md)
- Claude Code rules in `.claude/rules/`
- Architecture decision records in `docs/decisions/`
- Pre-commit hooks (ruff, bandit, conventional commits)

### Fixed
- Force reindex now clears vector store to prevent duplicate results

### Changed
- File scanning uses `git ls-files` for 100x speedup (falls back to os.walk for non-git repos)
