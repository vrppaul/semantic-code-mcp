# semantic-code-mcp

MCP server that provides semantic code search for Claude Code. Instead of iterative grep/glob, it indexes your codebase with embeddings and returns ranked results by meaning.

Supports **Python**, **Rust**, and **Markdown** — more languages planned.

## How It Works

```
Claude Code ──(MCP/STDIO)──▶ semantic-code-mcp server
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              AST Chunker      Embedder        LanceDB
             (tree-sitter)  (sentence-trans)  (vectors)
```

1. **Chunking** — tree-sitter parses source files into functions, classes, methods, structs, traits, markdown sections, etc.
2. **Embedding** — sentence-transformers encodes each chunk (all-MiniLM-L6-v2, 384d)
3. **Storage** — vectors stored in LanceDB (embedded, like SQLite)
4. **Search** — hybrid semantic + keyword search with recency boosting

Indexing is incremental (mtime-based). In a git repository, git lists the files to index — tracked ones and untracked ones `.gitignore` does not exclude; elsewhere the server walks the file system. The embedding model loads lazily on first query.

## Installation

### macOS / Windows

PyPI ships CPU-only torch on these platforms, so no extra flags are needed (~1.7GB install).

```bash
uvx semantic-code-mcp
```

**Claude Code integration:**

```bash
claude mcp add --scope user semantic-code -- uvx semantic-code-mcp
```

### Linux

> [!IMPORTANT]
> Without the `--index` flag, PyPI installs CUDA-bundled torch (~3.5GB). Unless you need GPU acceleration (you don't — embeddings run on CPU), use the command below to get the CPU-only build (~1.7GB).

```bash
uvx --index pytorch-cpu=https://download.pytorch.org/whl/cpu semantic-code-mcp
```

**Claude Code integration:**

```bash
claude mcp add --scope user semantic-code -- \
  uvx --index pytorch-cpu=https://download.pytorch.org/whl/cpu semantic-code-mcp
```

<details>
<summary>Claude Desktop / other MCP clients (JSON config)</summary>

```json
{
  "mcpServers": {
    "semantic-code": {
      "command": "uvx",
      "args": ["--index", "pytorch-cpu=https://download.pytorch.org/whl/cpu", "semantic-code-mcp"]
    }
  }
}
```

On macOS/Windows you can omit the `--index` and `pytorch-cpu` args.

</details>

### Updating

`uvx` caches the installed version. To get the latest release:

```bash
uvx --upgrade semantic-code-mcp
```

Or pin a specific version in your MCP config:

```bash
claude mcp add --scope user semantic-code -- uvx semantic-code-mcp@0.2.0
```

## MCP Tools

### `search_code`

Search code by meaning, not just text matching. Auto-indexes on first search.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | required | Natural language description of what you're looking for |
| `project_path` | `str` | required | Absolute path to the project root |
| `limit` | `int` | `10` | Maximum number of results, at most 50 |
| `include_debug` | `bool` | `False` | Also return timings and counts of the search |

Returns ranked results with `file_path`, `line_start`, `line_end`, `name`, `chunk_type`, `content` (cut to 50 lines, `truncated` says so) and `score`. A failure — a missing or relative path, an index that cannot be written — comes back as a tool error with its cause.

### `index_codebase`

Index a codebase for semantic search. Only processes new and changed files unless `force=True`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project_path` | `str` | required | Absolute path to the project root |
| `force` | `bool` | `False` | Re-index all files regardless of changes |

### `index_status`

Check indexing status for a project.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project_path` | `str` | required | Absolute path to the project root |

Returns `is_indexed`, `last_updated`, `files_count`, `chunks_count` and `stale_files_count`.

## Configuration

All settings are environment variables with the `SEMANTIC_CODE_MCP_` prefix (via pydantic-settings):

| Variable | Default | Description |
|----------|---------|-------------|
| `SEMANTIC_CODE_MCP_CACHE_DIR` | `~/.cache/semantic-code-mcp` | Where indexes are stored |
| `SEMANTIC_CODE_MCP_LOCAL_INDEX` | `false` | Store index in `.semantic-code/` within each project |
| `SEMANTIC_CODE_MCP_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `SEMANTIC_CODE_MCP_DEBUG` | `false` | Enable debug logging |
| `SEMANTIC_CODE_MCP_PROFILE` | `false` | Enable pyinstrument profiling (a development checkout only: pyinstrument is a dev dependency) |
| `SEMANTIC_CODE_MCP_USE_GITIGNORE` | `true` | Let `.gitignore` decide what is indexed. Set `false` for a repo whose `.gitignore` hides source on purpose (an allowlist such as `/*`); the server then walks the file system and applies the ignore patterns only |
| `SEMANTIC_CODE_MCP_GIT_TIMEOUT_SECONDS` | `10` | How long git may take to list the files before discovery walks the file system instead |
| `SEMANTIC_CODE_MCP_IGNORE_PATTERNS` | caches, virtualenvs, `node_modules`, top-level `build/`, `dist/`, `target/` | JSON list of patterns skipped in every mode. A pattern without `/` matches a name at any depth (`node_modules`); one with `/` matches from the project root (`docs/**`) |

Pass environment variables via the `env` field in your MCP config:

```json
{
  "mcpServers": {
    "semantic-code": {
      "command": "uvx",
      "args": ["semantic-code-mcp"],
      "env": {
        "SEMANTIC_CODE_MCP_DEBUG": "true",
        "SEMANTIC_CODE_MCP_LOCAL_INDEX": "true"
      }
    }
  }
}
```

Or with Claude Code CLI:

```bash
claude mcp add --scope user semantic-code \
  -e SEMANTIC_CODE_MCP_DEBUG=true \
  -e SEMANTIC_CODE_MCP_LOCAL_INDEX=true \
  -- uvx semantic-code-mcp
```

## Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| MCP Framework | mcp 2.x (`MCPServer`) | The official Python SDK, STDIO transport |
| Embeddings | sentence-transformers | Local, no API costs, good quality |
| Vector Store | LanceDB | Embedded (like SQLite), no server needed |
| Chunking | tree-sitter | AST-based, respects code structure |

## Development

```bash
uv sync                            # Install dependencies
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
uv run python -m semantic_code_mcp # Run server
uv run pytest                      # Run tests
uv run pre-commit run --all-files  # Every gate a commit runs
```

Pre-commit hooks run ruff, ty, bandit, deptry, typos, a check that every test module loads and a check of the agent instruction files, and hold commit messages to [Conventional Commits](https://www.conventionalcommits.org/). CI runs the same hooks and the tests on every push to `master` and every pull request. See [CONTRIBUTING.md](CONTRIBUTING.md).

### Releasing

Versions are derived from git tags automatically (`hatch-vcs`) — there's no hardcoded version in `pyproject.toml`.

```bash
git tag v0.2.0
git push origin v0.2.0
```

CI runs the whole suite on the tagged commit, then builds the package, publishes to PyPI, and creates a GitHub Release with auto-generated notes. A required reviewer on the `pypi` environment adds the owner's approval before publishing.

### Adding a New Language

The chunker system is designed to make adding languages straightforward. Each language needs:

1. **A tree-sitter grammar package** (e.g. `tree-sitter-javascript`)
2. **A chunker subclass** that walks the AST and extracts meaningful chunks

Steps:

```bash
uv add tree-sitter-mylang
```

Create `src/semantic_code_mcp/chunkers/mylang.py`:

```python
from enum import StrEnum, auto
from typing import override

import tree_sitter_mylang as tsmylang
from tree_sitter import Language, Node

from semantic_code_mcp.chunkers.base import BaseTreeSitterChunker, ChunkCollector, field_text
from semantic_code_mcp.domain.models import ChunkType


class NodeType(StrEnum):
    function_definition = auto()
    # ... other node types


class FieldName(StrEnum):
    name = auto()


class MyLangChunker(BaseTreeSitterChunker):
    language = Language(tsmylang.language())
    extensions = (".ml",)

    @override
    def _extract_chunks(self, root: Node, collector: ChunkCollector) -> None:
        for node in root.children:
            match node.type:
                case NodeType.function_definition:
                    name = field_text(node, FieldName.name)
                    if name is not None:
                        collector.add(node, node, ChunkType.function, name)
                # ... other node types
```

Register it in `src/semantic_code_mcp/services/projects.py`:

```python
from semantic_code_mcp.chunkers.mylang import MyLangChunker

def build_chunker() -> CompositeChunker:
    return CompositeChunker([PythonChunker(), RustChunker(), MarkdownChunker(), MyLangChunker()])
```

### Project Structure

- `src/semantic_code_mcp/server/` — the MCP tools, their answers and presenters
- `src/semantic_code_mcp/services/` — indexing, search, the registry of projects
- `src/semantic_code_mcp/discovery.py` — which files get indexed (git or a file-system walk, ignore patterns)
- `src/semantic_code_mcp/chunkers/` — language chunkers (`base.py`, `composite.py`, `python.py`, `rust.py`, `markdown.py`)
- `src/semantic_code_mcp/embedding.py`, `src/semantic_code_mcp/storage/` — embeddings and the LanceDB store
- `src/semantic_code_mcp/domain/` — models and interfaces
- `evals/` — search quality and speed on pinned corpora (`uv run python -m evals quality`)
- `docs/decisions/` — architecture decision records
- `TODO.md` — epics and planning
- `CHANGELOG.md` — completed work (Keep a Changelog format)
- `.claude/rules/` — context-specific coding rules for AI agents

## License

MIT
