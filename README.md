# semantic-code-mcp

MCP server that provides code search for Claude Code. Instead of iterative grep/glob, it indexes your codebase's words in seconds and returns ranked functions, classes and sections, a definition first when you ask its name. A setting adds a search by meaning, which embeds the code; agents measured with and without it found no gain from it.

Supports **Python**, **Rust**, and **Markdown** — more languages planned.

## How It Works

```
Claude Code ──(MCP/STDIO)──▶ semantic-code-mcp server
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              AST Chunker      Embedder         SQLite
             (tree-sitter)   (onnxruntime)  (words, vectors)
```

1. **Chunking** — tree-sitter parses source files into functions, classes, methods, structs, traits, markdown sections, etc.; a repository of 8 MiB of source or more is parsed in up to four processes beside the server, half its cores, for the seconds of its first index
2. **Storage** — the chunks and a full-text index of their words in one SQLite database per index (LanceDB, an optional extra, in its place)
3. **Search** — by words (BM25), the chunks named as the query first
4. **Meaning, by a setting** — with the extra `semantic-code-mcp[meaning]` and `SEMANTIC_CODE_MCP_SEARCH_MODE=words_and_meaning` the server embeds each chunk once with onnxruntime and the model's ONNX export (all-MiniLM-L6-v2, 384d, no torch), keeps each model's vectors beside the chunks, and fuses every search by meaning with one by words by rank

Indexing is incremental: the index records each file's modification time and size, and re-indexes the files that differ. Each chunker version and storage layout keeps an index of its own, so a new version builds a new index once, and switching back finds the old one; each model's vectors are kept apart inside it, so another model embeds again without chunking again; an index no session used for a week is removed. In a git repository, git lists the files to index — tracked ones and untracked ones `.gitignore` does not exclude; elsewhere the server walks the file system. The embedding model loads on the first search of a server set to search by meaning, never for a search by words.

## Installation

One command on every platform, about 95 MB installed. A search by meaning needs the model's libraries, the extra `semantic-code-mcp[meaning]` (190 MB in all: `uvx --from 'semantic-code-mcp[meaning]' semantic-code-mcp`), and `SEMANTIC_CODE_MCP_SEARCH_MODE=words_and_meaning`; such a server fetches the model's ONNX file (90 MB) from the Hugging Face hub once, on its first search:

```bash
uvx semantic-code-mcp
```

**Claude Code, as a plugin** — the server and the skill that tells the agent how to word a search and when to Grep instead, in one install (Claude Code 2.1.275 or later):

```
/plugin install semantic-code --marketplace vrppaul/semantic-code-mcp
```

On an earlier Claude Code, add the marketplace first:

```
/plugin marketplace add vrppaul/semantic-code-mcp
/plugin install semantic-code@semantic-code-marketplace
```

**Claude Code, the server alone:**

```bash
claude mcp add --scope user semantic-code -- uvx semantic-code-mcp
```

**The skill alone**, for an agent that has the server some other way:

```bash
npx skills add vrppaul/semantic-code-mcp
```

<details>
<summary>Claude Desktop / other MCP clients (JSON config)</summary>

```json
{
  "mcpServers": {
    "semantic-code": {
      "command": "uvx",
      "args": ["semantic-code-mcp"]
    }
  }
}
```

</details>

### Updating

`uvx` caches the installed version. To get the latest release:

```bash
uvx --upgrade semantic-code-mcp
```

Or pin a specific version in your MCP config:

```bash
claude mcp add --scope user semantic-code -- uvx semantic-code-mcp@<version>
```

## MCP Tools

### `search_code`

Search code by the words it uses. Auto-indexes on first search.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | required | The words to find, as code would write them, or a name (`Stack.push`), whose chunks come first |
| `project_path` | `str` | the client's workspace root, inside a git repository | Absolute path to the project, or to a directory inside its git repository: the repository is indexed once, and a search keeps to the directory; not a repository that ignores the directory or holds the user's home directory, where the directory is a project of its own |
| `limit` | `int` | `10` | Maximum number of results, at most 50 |
| `chunk_kinds` | `list[str]` | all | Only chunks of these kinds: `module`, `function`, `method`, `class`, `section` |
| `languages` | `list[str]` | all | Only chunks of files in these languages: `python`, `rust`, `markdown` |
| `path_glob` | `str` | none | Only files this glob names, as in a `.gitignore`: without a slash a name at any depth below the directory `project_path` names (`*.rs`, and `tests/`, as a trailing slash does not count), with one before a name a path from that directory (`src/**/*.py`, `/tests`); `*` stays within a directory, `**` crosses any number, even inside a name (`src**.py`, where a `.gitignore` reads it as `*`), `?` is one character, a directory takes what it holds; a character class (`[ab]`), braces, escapes, a leading `!`, `..` or more than 1000 characters is refused with the reason |
| `exclude_glob` | `str` | none | No file this glob names, read as `path_glob` is |
| `include_debug` | `bool` | `False` | Also return timings and counts of the search |

Returns `project_root` (the project's root: its repository's, or the directory itself outside git; given as `project_path`, it searches all of it), `searched_directory` (the directory the search kept to: `project_root`, or a directory under it) and ranked results with `file_path` (absolute), `line_start`, `line_end`, `name`, `kind` (`module`, `function`, `method`, `class`, `section`; Rust's structs, enums, traits and impls are classes), `parent` (the class, impl, module or heading the chunk sits in, as `Stack` for `Stack.push`), `content` (cut to 50 lines, `truncated` says so); results come best first. A failure — a missing or relative path, an index that cannot be written — comes back as a tool error with its cause.

### `index_codebase`

Index a codebase's words for search; it embeds nothing, and a server set to search by meaning embeds the chunks on its next search. Only processes new and changed files unless `force=True`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project_path` | `str` | the client's workspace root, inside a git repository | Absolute path to the project, or to a directory inside its git repository: the repository is indexed once, and a search keeps to the directory; not a repository that ignores the directory or holds the user's home directory, where the directory is a project of its own |
| `force` | `bool` | `False` | Re-index all files regardless of changes |

Returns `project_root` (a directory inside a repository indexes the repository from its root), `files_indexed`, `chunks_indexed`, `files_deleted` and `duration_seconds`.

### `index_status`

Check indexing status for a project.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project_path` | `str` | the client's workspace root, inside a git repository | Absolute path to the project, or to a directory inside its git repository: the repository is indexed once, and a search keeps to the directory; not a repository that ignores the directory or holds the user's home directory, where the directory is a project of its own |

Returns `project_root` (a directory inside a repository reports the repository's root), `is_indexed`, `last_updated`, `files_count`, `chunks_count`, `embedded_chunks_count` (the chunks with the server's model's vectors, which a server set to search by meaning reads without embedding first), `model` (the model, its revision and the embedder version whose vectors `embedded_chunks_count` counts), `stale_files_count`, `manifest` (the chunker and storage versions that built the index) and `being_built` (a run, this session's or another's, holds the index now: until it ends, the counts may be what it has written so far).

## Configuration

All settings are environment variables with the `SEMANTIC_CODE_MCP_` prefix (via pydantic-settings):

| Variable | Default | Description |
|----------|---------|-------------|
| `SEMANTIC_CODE_MCP_CACHE_DIR` | `~/.cache/semantic-code-mcp` | Where indexes are stored |
| `SEMANTIC_CODE_MCP_INDEX_STORE` | `sqlite` | What keeps the indexes: `sqlite`, or `lancedb`, whose libraries come with the extra `semantic-code-mcp[lancedb]` (`uvx --from 'semantic-code-mcp[lancedb]' semantic-code-mcp`). Each keeps indexes of its own, so a change builds the index again once |
| `SEMANTIC_CODE_MCP_LOCAL_INDEX` | `false` | Store index in `.semantic-code/` at the root of each project's repository; it holds a `.gitignore`, so git never shows it |
| `SEMANTIC_CODE_MCP_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | A sentence-transformers model that pools by the mean, with an ONNX export in one file (`onnx/model.onnx`), by hub name or directory. Only the default model's vectors are tested against sentence-transformers'; another is read from its configs as well as the embedder can, and refused where it can see the vectors would differ |
| `SEMANTIC_CODE_MCP_DEBUG` | `false` | Enable debug logging |
| `SEMANTIC_CODE_MCP_PROFILE` | `false` | Enable pyinstrument profiling (a development checkout only: pyinstrument is a dev dependency) |
| `SEMANTIC_CODE_MCP_USE_GITIGNORE` | `true` | Let `.gitignore` decide what is indexed. Set `false` for a repo whose `.gitignore` hides source on purpose (an allowlist such as `/*`); the server then walks the file system and applies the ignore patterns only |
| `SEMANTIC_CODE_MCP_GIT_TIMEOUT_SECONDS` | `10` | How long git may take to list the files before discovery walks the file system instead |
| `SEMANTIC_CODE_MCP_WATCH_FILES` | `true` | Follow each repository's files with the system's own watcher, so a search reads what changed since the last one instead of every file, with a full scan every 5 minutes as a check: inotify on Linux, which takes a watch for each directory git does not ignore, from the limit your editor shares (`fs.inotify.max_user_watches`), and an inotify instance per project; FSEvents on macOS, a stream for the project's root and one for each of git's directories outside it; ReadDirectoryChangesW on Windows, a handle of the project's root, which keeps a directory holding it from being moved meanwhile. Past those limits, outside a repository, with submodules, and on a network or FUSE file system (on macOS and Windows, the one the project's root lies on) each search scans the files, as with `false` |
| `SEMANTIC_CODE_MCP_SEARCH_MODE` | `words` | How every search ranks: `words`, BM25 over the chunks' words, no model loaded; or `words_and_meaning`, the model's vectors searched too and fused with the words by rank. The first search of a project by meaning embeds every chunk inside the call, minutes at 20k chunks and about an hour at 90k; later ones embed what changed. Meaning needs the extra `semantic-code-mcp[meaning]`: without it a server set to it stops as it starts, naming the extra |
| `SEMANTIC_CODE_MCP_MIN_SIMILARITY` | `0.35` | In a search by meaning, how close in meaning to the query, a cosine similarity, a result no keyword matched must be to come back; set for the default model, whose scale another model does not share. Below 0.25, lower `MIN_SIMILARITY_WHEN_WORDS_MATCH` with it, or the server refuses to start |
| `SEMANTIC_CODE_MCP_MIN_SIMILARITY_WHEN_WORDS_MATCH` | `0.25` | The same, for a result whose words match the query's; at most `MIN_SIMILARITY`. A result named as the query comes back however far |
| `SEMANTIC_CODE_MCP_IGNORE_PATTERNS` | caches, virtualenvs, `node_modules`, top-level `build/`, `dist/`, `target/` | JSON list of patterns skipped in every mode. A pattern without `/` matches a name at any depth (`node_modules`); one with `/` matches from the project root (`docs/**`) |

Every session on one project should run with the same `IGNORE_PATTERNS` and `USE_GITIGNORE`: an index keeps the files its last run found, so two sessions that find different files delete each other's at every search.

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
| Embeddings | onnxruntime over the model's ONNX export | Local, no API costs, no torch |
| Index | SQLite with FTS5 | In Python's standard library: one file, a write in one transaction, no server; LanceDB as an optional extra |
| Chunking | tree-sitter | AST-based, respects code structure |

## Development

```bash
uv sync                            # Install dependencies
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
uv run python -m semantic_code_mcp # Run server
uv run pytest                      # Run tests
uv run pre-commit run --all-files  # Every gate a commit runs
```

The gates are one list, `.pre-commit-config.yaml`: ruff (lint and format), ty (types), import-linter (the package's layers), vulture (code nothing calls), the project's code rules (nested functions, short names, models taken apart by field names), bandit (security), deptry (dependencies as declared), typos, a check that every test module loads, a check of the agent instruction files, the skill's gate (the skill and the server's instructions name what the server has), validate-pyproject, and the file checks of pre-commit-hooks; commit messages are held to [Conventional Commits](https://www.conventionalcommits.org/). CI runs the same hooks and the tests on every push to `master` and every pull request. See [CONTRIBUTING.md](CONTRIBUTING.md).

### Releasing

Versions are derived from git tags automatically (`hatch-vcs`) — there's no hardcoded version in `pyproject.toml`.

```bash
git tag v0.2.0
git push origin v0.2.0
```

CI runs the whole suite on the tagged commit, then builds the package, publishes to PyPI, and creates a GitHub Release with auto-generated notes. A required reviewer on the `pypi` environment would add the owner's approval before publishing; until the repository's settings name one, a green CI publishes at once.

### Adding a New Language

The chunker system is designed to make adding languages straightforward. Each language needs:

1. **A tree-sitter grammar package** (e.g. `tree-sitter-javascript`)
2. **A chunker subclass** that walks the AST and extracts meaningful chunks

Steps:

```bash
uv add tree-sitter-mylang
```

Add the language to `Language` in `src/semantic_code_mcp/domain/models.py`, and the separator its qualified names are written with (`Stack.push`, `outer::Stack`) to `PARENT_SEPARATORS` beside it; then create `src/semantic_code_mcp/chunkers/mylang.py`:

```python
from enum import StrEnum, auto
from typing import override

import tree_sitter
import tree_sitter_mylang as tsmylang
from tree_sitter import Node

from semantic_code_mcp.chunkers.base import BaseTreeSitterChunker, ChunkCollector, field_text
from semantic_code_mcp.domain.models import ChunkKind, Language


class NodeType(StrEnum):
    function_definition = auto()
    # ... other node types


class FieldName(StrEnum):
    name = auto()


class MyLangChunker(BaseTreeSitterChunker):
    language = Language.mylang
    grammar = tree_sitter.Language(tsmylang.language())
    extensions = (".ml",)

    @override
    def _extract_chunks(self, root: Node, collector: ChunkCollector) -> None:
        for node in root.children:
            match node.type:
                case NodeType.function_definition:
                    name = field_text(node, FieldName.name)
                    if name is not None:
                        collector.add(node, node, ChunkKind.function, name, parent=None)
                # ... other node types
```

Register it in `src/semantic_code_mcp/app.py`, and bump `CHUNKER_VERSION` in `src/semantic_code_mcp/chunkers/composite.py`, so indexes built before it are built again:

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
- `src/semantic_code_mcp/embedding/`, `src/semantic_code_mcp/storage/` — the embedder and the model it loads from the `meaning` extra, and the stores (SQLite, LanceDB) behind one contract
- `src/semantic_code_mcp/domain/` — models and interfaces
- `evals/` — search quality and speed on pinned corpora (`uv run python -m evals quality`, `speed`), and agents' sessions with and without the search (`agents`, on corpora it names)
- `docs/decisions/` — architecture decision records
- `TODO.md` — epics and planning
- `CHANGELOG.md` — completed work (Keep a Changelog format)
- `.claude/rules/` — context-specific coding rules for AI agents

## License

MIT
